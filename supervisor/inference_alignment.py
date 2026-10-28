"""Inference-only histories on genuine QPC sample times; no hardware access."""
from collections import deque
import threading

import numpy as np


class AlignmentPending(RuntimeError):
    pass


class SampleHistory:
    def __init__(self):
        self.lock = threading.Lock()
        self.key = None
        self.samples = deque(maxlen=128)

    def append(self, key, stamp, values):
        if not isinstance(stamp, int) or stamp <= 0:
            return
        values = np.asarray(values, dtype=float)
        if not np.isfinite(values).all():
            return
        with self.lock:
            if key != self.key:
                self.key = key
                self.samples.clear()
            # A repeated SDK sample never acquires a fresh timestamp.
            if self.samples and stamp <= self.samples[-1][0]:
                return
            self.samples.append((stamp, values.copy()))

    def at(self, key, anchor_ns, max_gap_ns=100_000_000, interpolate=None):
        with self.lock:
            samples = list(self.samples) if key == self.key else []
        left = next((s for s in reversed(samples) if s[0] <= anchor_ns), None)
        right = next((s for s in samples if s[0] >= anchor_ns), None)
        if left is None or right is None:
            raise AlignmentPending('RTC alignment waiting for genuine bracketing feedback')
        gap = right[0]-left[0]
        if gap > max_gap_ns:
            raise RuntimeError('RTC alignment feedback gap exceeds 100 ms; extrapolation prohibited')
        fraction = (anchor_ns-left[0])/gap if gap else 0.
        values = (interpolate(left[1], right[1], fraction) if interpolate else
                  left[1]+fraction*(right[1]-left[1]))
        return values, {'source_perf_ns': [left[0], right[0]], 'fraction': fraction,
                        'method': 'linear_measured_feedback', 'anchor_perf_ns': anchor_ns}


def next_grid_anchor(origin_ns, now_ns, period_ns):
    """First regular request is on-grid; later requests advance by E ticks."""
    return origin_ns + max(0, (now_ns-origin_ns+period_ns-1)//period_ns)*period_ns


def nearest_frames(resource, anchor_ns):
    """Select existing RGB frames without changing their real timestamps."""
    frames, stamps = [], []
    with resource.lock:
        history = getattr(resource, 'frame_history', {})
        for key in ('external', 'wrist'):
            choices = history.get(key) or []
            if not choices and resource.frames.get(key) is not None:
                choices = [(resource.frame_times_ns.get(key, 0), resource.frames[key])]
            if not choices:
                raise AlignmentPending('RTC alignment waiting for camera history')
            stamp, frame = min(choices, key=lambda item: (abs(item[0]-anchor_ns), item[0]))
            stamps.append(stamp)
            frames.append(frame.copy())
    if max(abs(t-anchor_ns) for t in stamps) > 150_000_000 or abs(stamps[0]-stamps[1]) > 75_000_000:
        raise RuntimeError('RTC aligned image/TCP skew exceeds existing freshness limits')
    return frames, stamps
