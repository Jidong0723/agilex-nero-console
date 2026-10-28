"""Read-only, shared freshness accounting for SDK receive-cache samples."""
from __future__ import annotations

import copy
import math
import threading
import time
from .inference_alignment import SampleHistory


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
        self.history = SampleHistory()

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
                diagnostic['reported_mode'] = candidate.get('mode')
                diagnostic['value_semantics'] = 'sdk_reported_width_not_independent_physical_measurement'
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
                self.history.append(session_id, self.received_ns, [self.sample['width_m']])
            except Exception as exc:
                diagnostic.update(error=str(exc), sdk_timestamp=self.marker,
                                  received_perf_ns=self.received_ns, feedback_revision=self.revision)
            now_ns = time.perf_counter_ns()
            diagnostic['feedback_age_ms'] = ((now_ns-self.received_ns)/1e6 if self.received_ns else None)
            diagnostic['cache_read_ms'] = (now_ns-started)/1e6
            return copy.deepcopy(self.sample), diagnostic


class GripperFeedbackFeed:
    """Parallel cache sampling, not a hardware writer or feedback-query sender.

    The reader uses the existing dedicated sensor IPC. Its thread is independent
    of UI/state polling and command RPC. Repeated SDK frames do not add samples.
    """
    def __init__(self, reader, context, stop_event, cache, event=None):
        self.reader, self.context, self.stop_event, self.cache = reader, context, stop_event, cache
        self.event = event
        self.lock = threading.Lock()
        self.value = {}
        self.thread = threading.Thread(target=self.run, name='nero-model-gripper-feedback', daemon=True)

    @staticmethod
    def identity(snapshot):
        return tuple((snapshot.get('session') or {}).get(k) for k in ('id','client_id','execution_mode'))

    def attach(self, snapshot):
        if (snapshot.get('session') or {}).get('execution_mode') == 'shadow':
            return snapshot
        key = self.identity(snapshot)
        with self.lock:
            value = copy.deepcopy(self.value)
        now = time.perf_counter_ns()
        if value.get('identity') != key:
            snapshot['gripper_observation'] = {'source':'parallel_sdk_receive_cache',
                'error':'waiting for current-session parallel gripper feedback'}
            snapshot['gripper'] = None
            return snapshot
        diagnostic = value['diagnostic']
        diagnostic.update(source='parallel_sdk_receive_cache',
            producer_read_ms=value['read_ms'], producer_poll_interval_ms=value.get('poll_interval_ms'),
            producer_cache_age_ms=(now-value['published_ns'])/1e6)
        stamp = diagnostic.get('received_perf_ns')
        diagnostic['feedback_age_ms'] = (now-stamp)/1e6 if stamp else None
        try:
            require_fresh(diagnostic, now)
        except RuntimeError as exc:
            diagnostic['error'] = str(exc)
        snapshot['gripper'] = value['sample']
        snapshot['gripper_observation'] = diagnostic
        return snapshot

    def run(self):
        previous = None
        while not self.stop_event.is_set():
            started = time.perf_counter_ns()
            key = self.identity(self.context())
            if key[2] != 'shadow':
                sample, diagnostic = self.cache.read(self.reader, key[0])
                completed = time.perf_counter_ns()
                if self.identity(self.context()) == key:
                    with self.lock:
                        self.value = {'identity':key,'sample':sample,'diagnostic':diagnostic,
                            'published_ns':completed, 'read_ms':(completed-started)/1e6,
                            'poll_interval_ms':(started-previous)/1e6 if previous else None}
                    if self.event and diagnostic.get('new_sdk_feedback'):
                        try:
                            self.event('gripper_source_sample', session_id=key[0],
                                sdk_timestamp=diagnostic.get('sdk_timestamp'),
                                received_perf_ns=diagnostic.get('received_perf_ns'),
                                published_perf_ns=completed, sdk_reported_width_m=(sample or {}).get('width_m'),
                                mode=(sample or {}).get('mode'),
                                force_n=(sample or {}).get('force_n'),
                                feedback_revision=diagnostic.get('feedback_revision'),
                                source_message_type=(((sample or {}).get('raw') or {}).get('value') or {}).get('msg_type'),
                                read_ms=(completed-started)/1e6,
                                physical_opening_verified=False, raw_can_bytes_available=False)
                        except Exception as exc:
                            with self.lock:
                                self.value['diagnostic']['logging_error'] = str(exc)
            previous = started
            # Best-effort 100 Hz polling of the SDK cache; never claim 100 Hz
            # independent frames and never catch up in a burst after a stall.
            self.stop_event.wait(max(0.,.01-(time.perf_counter_ns()-started)/1e9))
