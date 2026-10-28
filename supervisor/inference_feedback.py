"""Model-source-only measured TCP cache, independent of state/gripper/log RPCs."""
from __future__ import annotations

import copy
import threading
import time
from .inference_alignment import SampleHistory


def identity(snapshot):
    return tuple((snapshot.get('session') or {}).get(k) for k in ('id','client_id','execution_mode'))


class MeasuredTCPFeed:
    def __init__(self, source, context, stop_event):
        self.source, self.context, self.stop_event = source, context, stop_event
        self.lock = threading.Lock()
        self.value = {}
        self.history = SampleHistory()
        self.thread = threading.Thread(target=self.run, name='nero-model-measured-tcp', daemon=True)

    def status(self):
        with self.lock:
            value = self.value
            observation = value.get('observation') or {}
            now = time.perf_counter_ns()
            stamp = observation.get('perf_ns')
            return {'source': observation.get('source'), 'sample_perf_ns': stamp,
                'sample_age_ms': (now-stamp)/1e6 if isinstance(stamp,int) else None,
                'cache_age_ms': (now-value['published_perf_ns'])/1e6 if value.get('published_perf_ns') else None,
                'poll_interval_ms': value.get('poll_interval_ms'), 'refresh_ms': value.get('refresh_ms'),
                'sensor_path': (observation.get('timing') or {}).get('sensor_path'),
                'sensor_ipc': copy.deepcopy((observation.get('timing') or {}).get('sensor_ipc')),
                'error': value.get('error'), 'errors': value.get('errors',0), 'thread_alive':self.thread.is_alive()}

    def attach(self, snapshot):
        with self.lock:
            cached = self.value
            if cached.get('identity') != identity(snapshot):
                snapshot.pop('inference_observation', None)
                snapshot['inference_observation_error'] = 'waiting for current-session measured TCP'
                return snapshot
            observation = cached.get('observation')
            if observation is not None:
                snapshot['inference_observation'] = copy.deepcopy(observation)
                snapshot['inference_cache_perf_ns'] = cached['published_perf_ns']
            else:
                snapshot.pop('inference_observation', None)
            snapshot['tcp_feedback_diagnostic'] = {k: v for k,v in cached.items() if k not in ('observation','identity')}
            if cached.get('error'):
                snapshot['inference_observation_error'] = cached['error']
            else:
                snapshot.pop('inference_observation_error', None)
        return snapshot

    def run(self):
        previous_started = None
        errors = 0
        while not self.stop_event.is_set():
            started = time.perf_counter_ns()
            snapshot = self.context()
            key = identity(snapshot)
            try:
                result = self.source.refresh(snapshot, tcp_only=True)
                completed = time.perf_counter_ns()
                observation = result['inference_observation']
                if observation.get('joint_position_rad') is not None and identity(self.context()) == key:
                    self.history.append(key, observation['perf_ns'], observation['joint_position_rad'])
                observation['timing'].update(published_perf_ns=completed,
                    sample_age_at_publish_ms=(completed-observation['perf_ns'])/1e6)
                value = {'identity': key, 'observation': observation,
                    'published_perf_ns': completed, 'error': None, 'errors': errors,
                    'refresh_ms': (completed-started)/1e6,
                    'poll_interval_ms': (started-previous_started)/1e6 if previous_started else None}
            except Exception as exc:
                completed = time.perf_counter_ns()
                errors += 1
                with self.lock:
                    prior = self.value
                    observation = prior.get('observation') if prior.get('identity') == key else None
                    if observation and not 0 <= completed-observation['perf_ns'] <= 100_000_000:
                        observation = None
                    value = dict(prior if prior.get('identity') == key else {},
                        identity=key, observation=observation, error=str(exc), errors=errors)
            # A completed read must not cross into another session's cache.
            if identity(self.context()) == key:
                with self.lock:
                    self.value = value
            previous_started = started
            # No catch-up burst and no extra skipped slot after a slow read.
            self.stop_event.wait(max(0., .02-(time.perf_counter_ns()-started)/1e9))
