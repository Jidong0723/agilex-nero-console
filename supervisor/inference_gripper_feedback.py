"""Read-only, shared freshness accounting for SDK receive-cache samples."""
from __future__ import annotations

import copy
import math
import threading
import time


def sdk_marker(sample):
    raw = sample.get('raw') or {}
    value = raw.get('value') if isinstance(raw, dict) else None
    stamp = value.get('timestamp') if isinstance(value, dict) else None
    return stamp if isinstance(stamp, (float, int)) and math.isfinite(stamp) and stamp > 0 else None


def require_fresh(diagnostic, now_ns=None):
    now_ns = time.perf_counter_ns() if now_ns is None else now_ns
    if diagnostic.get('error'):
        raise RuntimeError(diagnostic['error'])
    stamp = diagnostic.get('received_perf_ns')
    if diagnostic.get('sdk_timestamp') is None or not isinstance(stamp, int) or stamp <= 0:
        raise RuntimeError('等待带时间戳的实测夹爪反馈')
    if not 0 <= now_ns - stamp <= 150_000_000:
        raise RuntimeError('实测夹爪SDK反馈已停更超过150 ms；重复读取不会刷新反馈时间')


class GripperFeedbackCache:
    """No commands, queries or session ownership; the reader only reads SDK RAM.

    This SDK uses the CAN receive timestamp (Unix seconds). Convert its age to
    QPC once per new frame, never timestamp an old cached frame as a new arrival.
    Session changes do not reset physical sample age. Clock changes invalidate
    the sample until a new frame with a plausible source time is observed.
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.sample = None
        self.marker = None
        self.received_ns = 0
        self.revision = 0
        self.clock_offset = None

    def read(self, reader, session_id=None):
        with self.lock:
            started = time.perf_counter_ns()
            diagnostic = {'source': 'sdk_receive_cache', 'session_id': session_id,
                          'new_sdk_feedback': False}
            try:
                candidate = reader()
                now_ns, wall = time.perf_counter_ns(), time.time()
                marker = sdk_marker(candidate)
                diagnostic['observed_sdk_timestamp'] = marker
                diagnostic['sdk_reported_hz'] = ((candidate.get('raw') or {}).get('value') or {}).get('hz')
                width = candidate.get('width_m')
                if marker is None or not isinstance(width, (float, int)) or not math.isfinite(width):
                    raise ValueError('SDK gripper cache has no valid timestamp/width')
                offset = wall - now_ns / 1e9
                clock_changed = self.clock_offset is not None and abs(offset-self.clock_offset) > .05
                self.clock_offset = offset
                if clock_changed:
                    self.received_ns = 0
                if marker != self.marker:
                    # Same width with a different RX timestamp is a new frame.
                    # A reconnect may reset the timestamp: accept only a fresh
                    # source frame, not a regressed stale cache.
                    age = wall-marker
                    if age < 0 or age > .15:
                        raise ValueError('SDK夹爪反馈源时间过期或与主机时钟不一致；等待真实新帧')
                    self.sample = copy.deepcopy(candidate)
                    self.marker = marker
                    self.received_ns = now_ns-int(age*1e9)
                    self.revision += 1
                    diagnostic['new_sdk_feedback'] = True
                diagnostic.update(sdk_timestamp=self.marker, received_perf_ns=self.received_ns,
                                  feedback_revision=self.revision,
                                  width_m=(self.sample or {}).get('width_m'))
                require_fresh(diagnostic, now_ns)
            except Exception as exc:
                diagnostic.update(error=str(exc), sdk_timestamp=self.marker,
                                  received_perf_ns=self.received_ns, feedback_revision=self.revision)
            now_ns = time.perf_counter_ns()
            diagnostic['feedback_age_ms'] = ((now_ns-self.received_ns)/1e6 if self.received_ns else None)
            diagnostic['cache_read_ms'] = (now_ns-started)/1e6
            return copy.deepcopy(self.sample), diagnostic
