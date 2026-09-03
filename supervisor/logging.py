from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
from typing import Any, Iterable

from shared.schemas import jsonable, now_iso


class AsyncJsonlTraceLogger:
    """Non-blocking trace writer used by the high-rate PICO input path."""

    def __init__(self, path: Path | str, metadata: dict[str, Any] | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=20000)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._writer, name="pico-trace-writer", daemon=True)
        self._thread.start()
        self.append({"record_type": "metadata", "timestamp": now_iso(), "metadata": metadata or {}})

    def append(self, record: dict[str, Any]) -> None:
        try:
            self._queue.put_nowait(dict(record))
        except queue.Full:
            pass

    def _writer(self) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            while not self._stop.is_set() or not self._queue.empty():
                try:
                    record = self._queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                if record is not None:
                    handle.write(json.dumps(jsonable(record), ensure_ascii=False) + "\n")
                    handle.flush()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


class JsonlExperimentLogger:
    def __init__(self, path: Path | str, metadata: dict[str, Any] | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata = metadata or {}
        if not self.path.exists():
            self.append({"record_type": "metadata", "timestamp": now_iso(), "metadata": self.metadata})

    def append(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(jsonable(record), ensure_ascii=False) + "\n")

    def frame(
        self,
        observation: Any = None,
        requested_action: dict[str, Any] | None = None,
        executed_action: Any = None,
        task_stage: str | None = None,
        success_label: bool | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.append(
            {
                "record_type": "frame",
                "timestamp": now_iso(),
                "task_stage": task_stage,
                "observation": observation,
                "requested_action": requested_action,
                "executed_action": executed_action,
                "success_label": success_label,
                "extra": extra or {},
            }
        )


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def replay_summary(records: Iterable[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for idx, record in enumerate(records):
        record_type = record.get("record_type", "unknown")
        if record_type == "metadata":
            lines.append(f"{idx}: metadata {record.get('metadata', {})}")
            continue
        action = record.get("requested_action")
        executed = record.get("executed_action") or {}
        ok = executed.get("ok") if isinstance(executed, dict) else None
        stage = record.get("task_stage")
        lines.append(f"{idx}: frame stage={stage} action={action} ok={ok}")
    return lines
