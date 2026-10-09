"""Read-only bounded OSC ingress history; never performs file or hardware I/O."""
from __future__ import annotations

import copy
import threading
import time
from collections import deque


class OscInputJournal:
    def __init__(self, capacity: int = 4096) -> None:
        self.lock = threading.Lock()
        self.rows = deque(maxlen=capacity)
        self.revision = 0
        self.session_id = None
        self.source = "external"

    def bind(self, session_id: str, source: str) -> None:
        with self.lock:
            self.session_id, self.source = session_id, source or "external"

    def context(self, session: dict, output_mode: str, connected: bool) -> dict:
        with self.lock:
            source = self.source if session.get("id") == self.session_id else "external"
        return {"control_source": source, "session_id": session.get("id"),
                "client_id": session.get("client_id"), "connected": bool(connected),
                "execution_mode": session.get("execution_mode"), "output_mode": output_mode}

    def append(self, body: dict, context: dict, received_ns: int, *, accepted: bool, error=None) -> None:
        with self.lock:
            self.revision += 1
            self.rows.append({"revision": self.revision, "received_perf_counter_ns": received_ns,
                "completed_perf_counter_ns": time.perf_counter_ns(),
                "context": copy.deepcopy(context),
                "command": {key: copy.deepcopy(body.get(key)) for key in
                            ("session_id", "client_id", "sequence", "type", "payload")},
                "tcp_semantics": "absolute_pose_robot_base", "accepted": bool(accepted), "error": error})

    def read(self, after_revision: int, max_items: int = 512) -> dict:
        with self.lock:
            rows = [row for row in self.rows if row["revision"] > after_revision][:max(1, min(max_items, 4096))]
            oldest = self.rows[0]["revision"] if self.rows else self.revision + 1
            return {"revision": self.revision, "lost_events": max(0, oldest - after_revision - 1),
                    "events": copy.deepcopy(rows)}
