"""RTC strategy owned exclusively by the model inference input source.

No OSC internals or hardware interfaces are imported here. All output uses the
adapter's existing client; one row lock is the stop/dispatch ordering barrier.
"""
from __future__ import annotations

import copy
import math
import threading
import time
from dataclasses import dataclass

import numpy as np

from .inference_contract import action_rows
from .inference_rtc_timing import RTCTiming, preserve_delay_prefix
from .inference_alignment import SampleHistory, AlignmentPending, nearest_frames, next_grid_anchor
from .inference_feedback import identity

VERSION = "nero.rtc.v2"
H = 20
PERIOD = .05
PERIOD_NS = 50_000_000
TIME_CONTRACT = "observation_time_plus_row_index_times_period"


def orientation_distance(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    dot = abs(float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b))))
    return 2*math.acos(min(1.0, dot))


def compatibility(metadata, execution):
    if metadata.get("rtc_version") != VERSION or metadata.get("ready") is not True:
        raise RuntimeError("RTC requires a ready nero.rtc.v2 server; no normal-mode fallback")
    if metadata.get("state_dim") != 7 or metadata.get("valid_action_dim") != 7 or metadata.get("action_horizon") != H:
        raise RuntimeError("RTC server must declare state7/action7/H20")
    if metadata.get("period_s") != PERIOD or metadata.get("action_contract") != "base_relative_observation_start_rotvec_gripper_absolute":
        raise RuntimeError("RTC server action/time contract mismatch")
    if metadata.get('action_time_contract') != TIME_CONTRACT or metadata.get('action_row0_offset_s') != 0:
        raise RuntimeError("RTC requires action row 0 at observation time, not response time")
    if metadata.get('rtc_bootstrap_mode') != 'hold_prefix_v1':
        raise RuntimeError('RTC requires a server with HOLD-prefix bootstrap support')
    if metadata.get('rtc_schedule') != 'fixed_delay_execute_horizon_v1':
        raise RuntimeError('RTC server must support fixed inference_delay / execute_horizon scheduling')
    if metadata.get('rtc_bootstrap_attention') != 'full_hold_v2':
        raise RuntimeError('RTC server must support full stationary HOLD bootstrap attention')
    if execution["expected_chunk_steps"] != H or execution.get("period_s", PERIOD) != PERIOD:
        raise RuntimeError("RTC requires H20 and 50 ms")
    timing=RTCTiming.from_execution(execution)
    if metadata.get('rtc_requires_execution_within_prefix'):
        if timing.inference_delay+2*timing.execute_horizon > H:
            raise RuntimeError('当前 RTC 采样配置要求整段执行位于引导重叠区：D+2E≤20；已验证配置为 inference_delay=6、execute_horizon=6')


@dataclass(frozen=True)
class Target:
    position: tuple
    quaternion: tuple
    gripper: float

    @classmethod
    def from_decoded(cls, decoded):
        pose, gripper = decoded
        return cls(tuple(pose["position_m"]), tuple(pose["orientation_xyzw"]), float(gripper))

    def pose(self):
        return {"position_m": list(self.position), "orientation_xyzw": list(self.quaternion)}

    def absolute_row(self):
        return [*self.position, *self.quaternion, self.gripper]


def interpolate_target(a, b, fraction):
    """Linear position/gripper and shortest-arc quaternion interpolation."""
    qa, qb = np.asarray(a.quaternion, float), np.asarray(b.quaternion, float)
    if np.dot(qa, qb) < 0:
        qb = -qb
    dot = float(np.clip(np.dot(qa, qb), -1, 1))
    if dot > .9995:
        quat = qa+(qb-qa)*fraction
    else:
        angle = math.acos(dot)
        quat = (math.sin((1-fraction)*angle)*qa+math.sin(fraction*angle)*qb)/math.sin(angle)
    quat /= np.linalg.norm(quat)
    return Target(tuple(np.asarray(a.position)+(np.asarray(b.position)-a.position)*fraction),
                  tuple(quat), a.gripper+(b.gripper-a.gripper)*fraction)


def sample_rows(rows, origin_ns, target_ns):
    index, remainder = divmod(target_ns-origin_ns, PERIOD_NS)
    if index < 0 or index >= len(rows) or (remainder and index+1 >= len(rows)):
        raise RuntimeError('RTC cannot extrapolate outside timestamped action rows')
    return rows[index] if not remainder else interpolate_target(rows[index], rows[index+1], remainder/PERIOD_NS)


def _terminal_bridge(rows, first, end, hold):
    """Quintic from stationary HOLD through three unchanged future samples.

    Fit the continuation rather than forcing zero terminal velocity before a
    moving model row. All samples retain their original observation timestamps.
    The caller still checks the entire seam against the same safety limits.
    """
    def product(a, b):
        va, vb = np.asarray(a[:3],float), np.asarray(b[:3],float)
        return np.r_[a[3]*vb+b[3]*va+np.cross(va,vb), a[3]*b[3]-np.dot(va,vb)]

    inverse = np.r_[-np.asarray(hold.quaternion[:3]), hold.quaternion[3]]
    values = []
    for target in rows[end:end+3]:
        q = product(target.quaternion, inverse)
        q /= np.linalg.norm(q)
        if q[3] < 0:
            q = -q
        norm = np.linalg.norm(q[:3])
        rotation = q[:3]*(2*math.atan2(norm,q[3])/norm) if norm > 1e-12 else np.zeros(3)
        values.append(np.r_[np.subtract(target.position,hold.position), rotation, target.gripper-hold.gripper])
    n = end-first
    times = np.asarray([1.,1+1/n,1+2/n])
    coefficients = np.linalg.solve(np.stack([times**3,times**4,times**5],axis=1),np.asarray(values))
    candidate = list(rows)
    for tick in range(first,end):
        s = (tick-first)/n
        value = np.asarray([s**3,s**4,s**5])@coefficients
        angle = np.linalg.norm(value[3:6])
        delta = np.r_[value[3:6]*(math.sin(angle/2)/angle),math.cos(angle/2)] if angle > 1e-12 else np.asarray([0.,0.,0.,1.])
        q = product(delta,hold.quaternion)
        q /= np.linalg.norm(q)
        candidate[tick] = Target(tuple(np.asarray(hold.position)+value[:3]),tuple(q),float(hold.gripper+value[6]))
    return candidate


def bootstrap_bridge(rows, first, hold, limits):
    """Join HOLD to a future model row on its ORIGINAL observation time grid.

    Only startup slots are locally generated; model rows are never time-shifted.
    The actual bridge becomes the prefix sent to subsequent RTC requests.
    """
    linear = min(.05, float(limits.get('linear_speed_m_s', .05)))
    angular = min(.2, float(limits.get('angular_speed_rad_s', .2)))
    if linear <= 0 or angular <= 0:
        raise RuntimeError('RTC bootstrap requires positive speed limits')
    def safe(candidate, end):
        # A terminal fit can overshoot: validate all generated points, not just
        # the destination. Keep the existing envelope and gripper limits.
        if any(not np.isfinite(t.absolute_row()).all()
               or np.linalg.norm(np.subtract(t.position,hold.position)) > .02
               or orientation_distance(t.quaternion,hold.quaternion) > math.radians(5)
               or not 0 <= t.gripper <= 1 for t in candidate[first:end+1]):
            return False
        ordered = [hold, *candidate[first:]]
        if any(np.linalg.norm(np.subtract(b.position,a.position)) > linear*PERIOD+1e-6
               or orientation_distance(a.quaternion,b.quaternion) > angular*PERIOD+1e-6
               or abs(b.gripper-a.gripper) > 2*PERIOD+1e-6
               for a,b in zip(ordered,ordered[1:])):
            return False
        velocities = [np.subtract(b.position,a.position)/PERIOD for a,b in zip(ordered,ordered[1:])]
        if any(np.linalg.norm(b-a) > .3*PERIOD+1e-6 for a,b in zip(velocities,velocities[1:])):
            return False
        angular_velocities = []
        for a,b in zip(ordered,ordered[1:]):
            va,wa = np.asarray(a.quaternion[:3]), a.quaternion[3]
            vb,wb = np.asarray(b.quaternion[:3]), b.quaternion[3]
            vector, scalar = wa*vb-wb*va-np.cross(vb,va), wa*wb+np.dot(va,vb)
            if scalar < 0:
                vector, scalar = -vector, -scalar
            norm = np.linalg.norm(vector)
            angular_velocities.append(vector*(2*math.atan2(norm,scalar)/norm/PERIOD) if norm > 1e-12 else np.zeros(3))
        if any(np.linalg.norm(b-a) > PERIOD+1e-6 for a,b in zip(angular_velocities,angular_velocities[1:])):
            return False
        return True

    # Leave two unchanged original rows as runway for the first replan.
    for end in range(first+6, H-2):
        destination = rows[end]
        distance = float(np.linalg.norm(np.subtract(destination.position, hold.position)))
        angle = orientation_distance(destination.quaternion, hold.quaternion)
        duration = (end-first)*PERIOD
        if distance > .02 or angle > math.radians(5):
            continue
        # These analytic peaks apply only to the zero-terminal smoothstep,
        # not the continuation fit; both must pass the actual seam checks.
        if (1.875*distance/duration <= linear and 5.78*distance/duration**2 <= .3
                and 1.875*angle/duration <= angular and 5.78*angle/duration**2 <= 1.
                and 1.875*abs(destination.gripper-hold.gripper)/duration <= 2.):
            candidate = list(rows)
            for tick in range(first, end+1):
                s = (tick-first)/(end-first)
                candidate[tick] = interpolate_target(hold, destination, 10*s**3-15*s**4+6*s**5)
            if safe(candidate,end):
                return candidate, end
        candidate = _terminal_bridge(rows,first,end,hold)
        if safe(candidate,end):
            return candidate,end
    raise RuntimeError('RTC bootstrap cannot safely join HOLD within remaining H20 horizon; restart required')


class Timeline:
    """Small atomic timeline; immutable prefix ownership remains LOCAL."""
    def __init__(self, epoch, origin_ns=0):
        self.epoch = epoch
        self.revision = 0
        self.next_tick = 0
        self.committed_until = 0
        self.targets = {}
        self.history = {}
        self.origin_ns = origin_ns
        self.lock = threading.RLock()

    def seed(self, targets, now_ns=None, hold_target=None):
        if len(targets) != H:
            raise ValueError("Expected full H20 seed")
        with self.lock:
            first = 0 if now_ns is None else (now_ns-self.origin_ns)//PERIOD_NS+1
            if not 0 <= first < H:
                raise RuntimeError('RTC bootstrap has no future rows')
            self.next_tick = self.committed_until = first
            self.targets = {i: target for i, target in enumerate(targets) if i >= first}
            # Past bootstrap slots were HOLD, never model actions executed late.
            self.history = {i: hold_target for i in range(first)} if hold_target else {}
            self.revision += 1

    def context(self, observation_tick, delay, request_id, *, observation_ns=None, now_ns=None, minimum_freeze_tick=0, execute_horizon=None):
        with self.lock:
            observation_ns = self.origin_ns+observation_tick*PERIOD_NS if observation_ns is None else observation_ns
            observation_tick = (observation_ns-self.origin_ns)//PERIOD_NS
            # Delay is relative to the observation, never the request/response
            # clock. Round only the absolute boundary onto the execution grid.
            freeze = max(self.next_tick, self.committed_until, minimum_freeze_tick,
                (observation_ns+delay*PERIOD_NS-self.origin_ns+PERIOD_NS-1)//PERIOD_NS)
            prefix = []
            known = {**self.history, **self.targets}
            for i in range(H):
                tick, rem = divmod(observation_ns+i*PERIOD_NS-self.origin_ns, PERIOD_NS)
                if tick not in known or (rem and tick+1 not in known):
                    break
                target = known[tick] if not rem else interpolate_target(known[tick], known[tick+1], rem/PERIOD_NS)
                prefix.append(target.absolute_row())
            frozen_rows = (self.origin_ns+freeze*PERIOD_NS-observation_ns+PERIOD_NS-1)//PERIOD_NS
            if len(prefix) < frozen_rows:
                raise RuntimeError("RTC runway insufficient for another inference")
            result = {"version": VERSION, "phase": "replan", "run_epoch": self.epoch,
                    "request_id": request_id, "observation_tick": observation_tick,
                    "plan_revision": self.revision, "freeze_until": freeze,
                    "observation_perf_ns": observation_ns, "timeline_origin_perf_ns": self.origin_ns,
                    "freeze_until_perf_ns": self.origin_ns+freeze*PERIOD_NS,
                    "action_row0_offset_s": 0,
                    "period_s": PERIOD, "absolute_prefix": prefix}
            if execute_horizon is not None:
                result.update(inference_delay=delay, execute_horizon=execute_horizon,
                              prefix_attention_horizon=H-execute_horizon)
            return result

    def merged(self, context, targets, now_ns=None):
        """Build a candidate while locked; caller validates it before publish."""
        if context["run_epoch"] != self.epoch or context["plan_revision"] != self.revision:
            raise RuntimeError("Obsolete RTC plan")
        observation_ns = context['observation_perf_ns']
        last = (observation_ns+(H-1)*PERIOD_NS-self.origin_ns)//PERIOD_NS
        cut = max(context["freeze_until"], self.committed_until, self.next_tick)
        if now_ns is not None:
            cut = max(cut, (now_ns-self.origin_ns)//PERIOD_NS+1)
        if len(targets) != H or cut > last:
            raise RuntimeError("RTC response has no usable future")
        result = {t: v for t, v in self.targets.items() if self.next_tick <= t < cut}
        result.update({tick: sample_rows(targets, observation_ns, self.origin_ns+tick*PERIOD_NS)
                       for tick in range(cut, last+1)})
        if any(t not in result for t in range(self.next_tick, last+1)):
            raise RuntimeError("RTC splice would create a queue gap")
        return result

    def commit_next(self):
        with self.lock:
            tick = self.next_tick
            if tick not in self.targets:
                raise RuntimeError("RTC queue exhausted")
            self.committed_until = tick+1
            return tick, self.targets[tick]

    def merge_response(self, context, targets, now_ns, metadata):
        """Replies may replace only continuous, uncommitted future slots.

        D defines the original frozen prefix, not a blanket response timeout.
        Keep the observation epoch and sample at the actual replacement time.
        For a late reply, require an explicitly supported guided profile,
        surviving old targets, and millimetre-scale same-time continuity.
        """
        candidate = self.merged(context, targets, now_ns=now_ns)
        cut = max(context['freeze_until'], self.committed_until, self.next_tick,
                  (now_ns-self.origin_ns)//PERIOD_NS+1)
        late_ns = max(0, now_ns-context['freeze_until_perf_ns'])
        evidence = dict(replacement_tick=cut, response_lateness_ms=late_ns/1e6)
        if late_ns:
            bounded_late = (metadata.get('rtc_late_response_policy') == 'guided_same_time_continuity_v1'
                            or metadata.get('rtc_requires_execution_within_prefix'))
            if (metadata.get('prefix_attention_schedule') not in {'ones', 'linear', 'exp'}
                    or not bounded_late):
                raise RuntimeError('RTC inference_delay exceeded without supported future guidance')
        row = (self.origin_ns+cut*PERIOD_NS-context['observation_perf_ns'])/PERIOD_NS
        last_guided = min(len(context['absolute_prefix']), context['prefix_attention_horizon'])-1
        old = self.targets.get(cut)
        if old is None or row > last_guided:
            raise RuntimeError('RTC late response has no guided old-plan continuation' if late_ns
                               else 'RTC response has no same-time old-plan continuation')
        position_gap = float(np.linalg.norm(np.subtract(candidate[cut].position, old.position)))
        angle_gap = orientation_distance(candidate[cut].quaternion, old.quaternion)
        if position_gap > .002 or angle_gap > math.radians(.5):
            raise RuntimeError(f'RTC response does not safely match the old plan: {position_gap*1000:.3f} mm / {math.degrees(angle_gap):.3f} deg')
        evidence.update(same_time_position_gap_m=position_gap, same_time_orientation_gap_rad=angle_gap,
                        continuity_checked=True)
        return candidate, evidence

    def complete(self, tick):
        with self.lock:
            if tick != self.next_tick:
                raise RuntimeError("RTC dispatch order mismatch")
            self.history[tick] = self.targets.pop(tick)
            self.history = {t: v for t, v in self.history.items() if t >= tick-H}
            self.next_tick += 1


class ObservationAssembler:
    def __init__(self, adapter):
        self.a = adapter
        self._gripper_marker = None
        self._gripper_received_ns = 0
        self._gripper_has_advanced = False
        self._gripper_lock = threading.Lock()
        self._simulated_history = SampleHistory()
        self._simulated_gripper = SampleHistory()

    def read(self):
        a = self.a
        resource = a.camera_resource
        if resource is None or not hasattr(resource, "frame_times_ns"):
            raise RuntimeError("RTC requires timestamped shared camera frames")
        with resource.lock:
            frames = [resource.frames.get(k) for k in ("external", "wrist")]
            stamps = [resource.frame_times_ns.get(k, 0) for k in ("external", "wrist")]
        now = time.perf_counter_ns()
        if any(f is None for f in frames) or any(not 0 <= now-t <= 150_000_000 for t in stamps):
            raise RuntimeError("RTC camera feedback absent or older than 150 ms")
        if abs(stamps[0]-stamps[1]) > 75_000_000:
            raise RuntimeError("RTC camera pair skew exceeds 75 ms")
        with a.lock:
            osc = a._inference_snapshot() if hasattr(a, '_inference_snapshot') else copy.deepcopy(a._osc_snapshot)
            published = osc.get('inference_cache_perf_ns', a._log_feedback_snapshot.get("cache_published_perf_ns", 0))
        monotonic_now = time.monotonic_ns()
        now = time.perf_counter_ns()
        session = osc.get("session") or {}
        if session.get("state") != "ACTIVE" or (session.get("id"), session.get("client_id")) != (a.state.get("session_id"), a.state.get("client_id")):
            raise RuntimeError("RTC control session no longer belongs to this run")
        execution = osc.get("execution") or {}
        measured = osc.get('inference_observation')
        if session.get('execution_mode') == 'hardware' and callable(getattr(a.osc,'sensor_sample',None)) and not measured:
            raise RuntimeError('RTC measured TCP unavailable: '+str(osc.get('inference_observation_error') or 'waiting for measured joint FK'))
        pose = measured['pose'] if measured else execution.get("measured_tcp_pose")
        if not pose or not 0 <= now-published <= 100_000_000:
            raise RuntimeError("RTC requires fresh measured TCP; target fallback is prohibited")
        stamp = execution.get("sample_monotonic_ns")
        # OSC timestamps use monotonic_ns; camera/cache timestamps use
        # perf_counter_ns. On Windows Python <=3.12 these have different
        # epochs (GetTickCount64 vs QPC), so compare only after conversion.
        if isinstance(stamp, int):
            stamp += now-monotonic_now
        if measured:
            stamp = measured['perf_ns']
            execution = dict(execution, feedback_age_s=measured['feedback_age_s'], measured_tcp_pose=pose)
            osc['execution'] = execution
            if 'gripper_width_m' in measured:
                osc['gripper'] = dict(osc.get('gripper') or {}, width_m=measured['gripper_width_m'])
        if session.get("execution_mode") == "hardware":
            health = {"read_perf_ns": now, "sample_perf_ns": stamp,
                      "sample_age_ms": (now-stamp)/1e6 if isinstance(stamp, int) else None,
                      "cache_age_ms": (now-published)/1e6,
                      "feedback_age_ms": float(execution.get("feedback_age_s", math.inf))*1000,
                      "source": measured.get('source') if measured else 'osc_execution',
                      "observation_timing": measured.get('timing') if measured else None}
            a._event('rtc_observation_health', **health)
            with a.lock:
                a.state['rtc_observation_health'] = health
            if not isinstance(stamp, int) or not 0 <= now-stamp <= 100_000_000:
                raise RuntimeError("RTC measured TCP timestamp missing or stale")
            if float(execution.get("feedback_age_s", math.inf)) > .1:
                raise RuntimeError("RTC hardware feedback is stale")
            diagnostic = osc.get("gripper_observation") or {}
            # Use the producer's actual source age, not when this consumer first
            # noticed a marker. A fresh first frame needs no extra inference cycle.
            from .inference_gripper_feedback import require_fresh
            require_fresh(diagnostic, now)
        else:
            stamp = stamp or published
            # CPV HOLD freezes its TCP sample while the shadow plant continues
            # publishing feedback. Reuse that pose only with fresh evidence
            # that the simulated joints are stationary at the sampled state.
            diagnostics = osc.get("diagnostics") or {}
            plant = diagnostics.get("shadow_transport") or {}
            sampled_q = execution.get("measured_joint_state_rad")
            applied_q = plant.get("applied_joint_state_rad")
            applied_qd = plant.get("applied_joint_velocity_rad_s")
            if (session.get("execution_mode") == "shadow"
                    and diagnostics.get("trajectory_state") == "HOLD_READY"
                    and plant.get("enabled") is True
                    and all(isinstance(v, (list, tuple)) and len(v) == 7
                            for v in (sampled_q, applied_q, applied_qd))):
                values = np.asarray([sampled_q, applied_q, applied_qd], dtype=float)
                if (np.isfinite(values).all()
                        and np.max(np.abs(values[0]-values[1])) <= 1e-9
                        and np.max(np.abs(values[2])) <= 1e-9):
                    stamp = published
        if max(abs(stamp-t) for t in stamps) > 150_000_000:
            raise RuntimeError(f"RTC image/TCP skew exceeds 150 ms (TCP age={(now-stamp)/1e6:.1f} ms, "
                               f"image ages={[round((now-t)/1e6, 1) for t in stamps]} ms)")
        observation = a._observation(osc, *frames)
        # TCP sample time is the reference epoch; retain camera exposure times
        # separately rather than pretending capture and cache-read are identical.
        osc['rtc_observation_times'] = {'observation_perf_ns': stamp,
                                       'image_perf_ns': list(stamps), 'read_perf_ns': now}
        if session.get('execution_mode') == 'shadow':
            self._simulated_history.append(identity(osc), stamp,
                [*pose['position_m'], *pose['orientation_xyzw']])
            self._simulated_gripper.append(identity(osc), stamp,
                [float(observation['observation/state'][6])])
        return observation, copy.deepcopy(pose), osc, stamp

    def read_aligned(self, anchor_ns, cancel):
        """Real feedback brackets a scheduled tick; no retimestamped latest pose.

        Waiting is bounded by the existing 100 ms TCP freshness budget, and
        remains inside D, not an extra allowance to dispatch expired rows.
        """
        deadline = anchor_ns+100_000_000
        while not cancel.is_set():
            _, _, osc, raw_stamp = self.read()  # Existing health/session checks.
            try:
                key = identity(osc)
                feed = getattr(self.a, '_tcp_feed', None)
                if feed is not None:
                    joints, state_evidence = feed.history.at(key, anchor_ns)
                    pose = self.a._observation_source.pose_for_joints(joints)
                    state_evidence['joint_position_rad'] = joints.tolist()
                elif (osc.get('session') or {}).get('execution_mode') == 'shadow':
                    # Direct fake clients have explicit simulated poses but no
                    # hardware FK producer. Never use this for real hardware.
                    def interpolate_pose(a, b, f):
                        t = interpolate_target(Target(tuple(a[:3]),tuple(a[3:]),0),
                                               Target(tuple(b[:3]),tuple(b[3:]),0),f)
                        return np.asarray([*t.position,*t.quaternion])
                    row, state_evidence = self._simulated_history.at(key, anchor_ns,
                        interpolate=interpolate_pose)
                    pose = {'position_m': row[:3].tolist(), 'orientation_xyzw': row[3:].tolist()}
                    state_evidence['method'] = 'simulated_pose_interpolation'
                else:
                    raise RuntimeError('RTC aligned hardware observation requires measured joint history')
                if key[2] == 'hardware':
                    cache = getattr(self.a, '_measured_gripper_cache', None)
                    if cache is None:
                        raise AlignmentPending('RTC alignment waiting for measured gripper history')
                    width, grip_evidence = cache.history.at(key[0], anchor_ns)
                    width = float(width[0])
                else:
                    ratio, grip_evidence = self._simulated_gripper.at(key, anchor_ns)
                    bounds = self.a.config['gripper']
                    width = bounds['min_width_m']+float(ratio[0])*(bounds['max_width_m']-bounds['min_width_m'])
                frames, stamps = nearest_frames(self.a.camera_resource, anchor_ns)
                now = time.perf_counter_ns()
                if not 0 <= now-anchor_ns <= 100_000_000:
                    raise RuntimeError('RTC alignment anchor exceeds existing TCP freshness budget')
                if any(not 0 <= now-t <= 150_000_000 for t in stamps):
                    raise RuntimeError('RTC aligned camera feedback absent or older than 150 ms')
                observed = copy.deepcopy(osc.get('inference_observation') or {})
                observed.update(pose=pose, perf_ns=anchor_ns, gripper_width_m=width,
                                source='grid_aligned_measured_feedback')
                osc['inference_observation'] = observed
                evidence = {'observation_perf_ns': anchor_ns, 'raw_latest_tcp_perf_ns': raw_stamp,
                    'image_perf_ns': stamps, 'image_offset_ms': [(t-anchor_ns)/1e6 for t in stamps],
                    'read_perf_ns': now, 'state_alignment': state_evidence,
                    'gripper_alignment': grip_evidence, 'alignment_wait_ms': (now-anchor_ns)/1e6}
                osc['rtc_observation_times'] = evidence
                observation = self.a._observation(osc, *frames)
                self.a._event('rtc_observation_alignment', **evidence)
                return observation, copy.deepcopy(pose), osc, anchor_ns
            except AlignmentPending:
                if time.perf_counter_ns() >= deadline:
                    raise RuntimeError('RTC alignment missing bracketing feedback within 100 ms; no extrapolation')
                if cancel.wait(.002):
                    break
        return None


class RtcRun:
    def __init__(self, adapter, epoch, session_id, client_id, policy):
        self.a, self.epoch, self.session, self.client, self.policy = adapter, epoch, session_id, client_id, policy
        self.cancel = threading.Event()
        self.finished = threading.Event()
        self.timeline = Timeline(epoch)
        self.assembler = ObservationAssembler(adapter)
        self.request_id = 0
        self.t0 = None
        self.executor = None
        self.health = None
        self.hold_confirmed = False
        self.authority_released = False
        self._hold_lock = threading.Lock()
        self.last_dispatch_started = None

    def live(self):
        with self.a.lock:
            return not self.cancel.is_set() and self.epoch == self.a._run_revision and bool(self.a.state.get("execution_enabled") or self.a.state.get("priming"))

    @property
    def stop_confirmed(self):
        return self.hold_confirmed or self.authority_released

    def publish(self, **fields):
        with self.a.lock:
            if self.epoch == self.a._run_revision:
                self.a.state.setdefault("rtc", {}).update(fields)

    def stop(self, reason):
        self.cancel.set()
        # A timed-out caller never releases a background writer. Wait for the
        # one synchronous command to settle before HOLD, and block restart.
        with self._hold_lock:
            if self.stop_confirmed:
                return
            with self.a._row_lock:
                current = self.a.osc.state().get("session") or {}
                if (current.get("id"), current.get("client_id")) != (self.session, self.client):
                    self.authority_released = True  # Don't send HOLD to another source.
                    with self.timeline.lock:
                        self.timeline.targets.clear()
                    return
                result = self.a.osc.hold(self.session, self.client, self.a._next_sequence(), reason)
                if not result.get("ok"):
                    raise RuntimeError(f"RTC HOLD not confirmed: {result}")
                self.hold_confirmed = True
                with self.timeline.lock:
                    self.timeline.targets.clear()

    def fail(self, reason):
        if self.cancel.is_set():
            return
        self.cancel.set()
        with self.a.lock:
            if self.epoch != self.a._run_revision:
                return
            self.a.state.update(state="ERROR", execution_enabled=False, priming=False, last_error=reason)
            self.a.state.setdefault("rtc", {}).update(state="HOLD_PENDING", reason=reason, manual_restart_required=True)
        self.a._event("rtc_fault", reason=reason)
        self.a._invalidate_policy(self.policy, reason)
        try:
            self.stop(reason)
            self.publish(state="RELEASED" if self.authority_released else "HOLD", hold_confirmed=self.hold_confirmed,
                         authority_released=self.authority_released)
        except Exception as exc:
            self.publish(state="HOLD_UNCONFIRMED", reason=f"{reason}; {exc}", hold_confirmed=False)

    def validate_targets(self, targets, osc):
        decoded = [(t.pose(), t.gripper) for t in targets]
        self.a._validate_execution_targets(decoded, len(decoded), osc)
        angular_limit = (osc.get("limits") or {}).get("angular_speed_rad_s")
        linear_limit = (osc.get("limits") or {}).get("linear_speed_m_s")
        for previous, target in zip(targets, targets[1:]):
            if linear_limit is not None and np.linalg.norm(np.subtract(target.position, previous.position)) > float(linear_limit)*PERIOD+1e-6:
                raise RuntimeError("RTC splice exceeds published Cartesian speed limit")
            angle = orientation_distance(target.quaternion, previous.quaternion)
            if angular_limit is not None and angle > float(angular_limit)*PERIOD+1e-6:
                raise RuntimeError("RTC splice exceeds published angular speed limit")

    def infer(self, observation, base, osc, context):
        self.request_id += 1
        context["request_id"] = self.request_id
        started = time.perf_counter()
        self.a._event("rtc_request", context=context, observation_state=observation["observation/state"], base_tcp=base,
                      observation_times=osc.get('rtc_observation_times'))
        response = self.policy.infer(dict(observation, rtc_context=context))
        if not self.live():
            return None
        elapsed = time.perf_counter()-started
        echo = response.get("rtc", {})
        for key in ("version", "phase", "run_epoch", "request_id", "observation_tick", "plan_revision", "freeze_until",
                    "observation_perf_ns", "timeline_origin_perf_ns", "freeze_until_perf_ns", "action_row0_offset_s"):
            if echo.get(key) != context[key]:
                raise RuntimeError(f"RTC response identity mismatch: {key}")
        for key in ('inference_delay', 'execute_horizon', 'prefix_attention_horizon'):
            if key in context and echo.get(key) != context[key]:
                raise RuntimeError(f'RTC response timing mismatch: {key}')
        expected_sampler = "bootstrap_hold_vjp" if context["phase"] == "bootstrap_hold" else "rtc_vjp"
        if echo.get("sampler") != expected_sampler:
            raise RuntimeError("RTC sampler mismatch; normal fallback is prohibited")
        rows = action_rows(response, H)
        targets = [Target.from_decoded(self.a._target_from_action(row, base, action_index=i)) for i, row in enumerate(rows)]
        self.validate_targets(targets, osc)
        diagnostics = {key: echo[key] for key in ("sampler_config", "prefix_residuals",
                                                   "prefix_position_rmse_m", "prefix_gripper_rmse",
                                                   "hard_prefix_position_rmse_m", "soft_prefix_position_rmse_m")
                       if key in echo}
        with self.a.lock:
            if not self.live():
                return None
            self.a.state.update(action_chunk=rows, absolute_tcp_chunk=[t.pose() for t in targets],
                                chunk_source="control", chunk_updated_at=time.time(),
                                inference_base_tcp=base, action_chunk_length=H, last_response_steps=H,
                                inference_ms=elapsed*1000, chunk_sequence=self.a.state["chunk_sequence"]+1,
                                rtc_diagnostics=diagnostics,
                                protocol_compatibility={"status": "compatible", "message": "RTC 7D/H20 响应验证通过"})
        self.a._event("rtc_response", context=context, actions=rows, absolute_targets=[t.absolute_row() for t in targets],
                      elapsed_ms=elapsed*1000, server_timing=response.get("policy_timing"), rtc_diagnostics=diagnostics,
                      row_target_perf_ns=[context['observation_perf_ns']+i*PERIOD_NS for i in range(H)])
        self.publish(end_to_end_ms=elapsed*1000, sampler=echo["sampler"])
        return targets, elapsed

    def _health_loop(self):
        while not self.cancel.wait(.05):
            try:
                self.assembler.read()
                self.a.osc.heartbeat(self.client, self.session)
            except Exception as exc:
                self.fail(f"RTC feedback/heartbeat failed: {exc}")
                return

    def _execute(self):
        try:
            while self.live():
                with self.timeline.lock:
                    tick = self.timeline.next_tick
                deadline = self.t0+tick*PERIOD
                wait_started_ns = time.perf_counter_ns()
                if self.cancel.wait(max(0, deadline-time.perf_counter())):
                    return
                wake_ns = time.perf_counter_ns()
                wake_lateness_ms = (wake_ns/1e9-deadline)*1000
                self.a._event("rtc_executor_timing", tick=tick, stage="woke",
                              deadline_perf_s=deadline, wait_started_perf_ns=wait_started_ns,
                              wake_perf_ns=wake_ns, wake_lateness_ms=wake_lateness_ms)
                if time.perf_counter() >= deadline+PERIOD:
                    raise RuntimeError("RTC dispatch missed its 50 ms slot")
                dispatch_timing = {"tick": tick, "deadline_perf_s": deadline,
                                   "wake_perf_ns": wake_ns,
                                   "wake_lateness_ms": wake_lateness_ms,
                                   "row_lock_requested_perf_ns": time.perf_counter_ns(),
                                   "stage": "waiting_for_row_lock"}
                try:
                    with self.a._row_lock:
                        dispatch_timing["row_lock_acquired_perf_ns"] = time.perf_counter_ns()
                        dispatch_timing["row_lock_wait_ms"] = (
                            dispatch_timing["row_lock_acquired_perf_ns"]-
                            dispatch_timing["row_lock_requested_perf_ns"])/1e6
                        dispatch_timing["stage"] = "row_lock_acquired"
                        if not self.live():
                            return
                        if time.perf_counter() >= deadline+PERIOD:
                            raise RuntimeError('RTC target expired while waiting for output lock')
                        if self.last_dispatch_started is not None and time.perf_counter()-self.last_dispatch_started < PERIOD/2:
                            raise RuntimeError("RTC dispatch jitter would require a catch-up burst")
                        tick, target = self.timeline.commit_next()
                        started = time.perf_counter()
                        dispatch_timing["dispatch_started_perf_ns"] = time.perf_counter_ns()
                        dispatch_timing["stage"] = "track_tcp"
                        self.last_dispatch_started = started
                        sequence_started_ns = time.perf_counter_ns()
                        sequence = self.a._next_sequence()
                        sequence_finished_ns = time.perf_counter_ns()
                        dispatch_timing["next_sequence_started_perf_ns"] = sequence_started_ns
                        dispatch_timing["next_sequence_finished_perf_ns"] = sequence_finished_ns
                        dispatch_timing["next_sequence_ms"] = (sequence_finished_ns-sequence_started_ns)/1e6
                        dispatch_timing["osc_client_call_started_perf_ns"] = time.perf_counter_ns()
                        result = self.a.osc.track_tcp(self.session, self.client, sequence, target.pose())
                        dispatch_timing["osc_client_call_returned_perf_ns"] = time.perf_counter_ns()
                        dispatch_timing["osc_client_call_ms"] = (dispatch_timing["osc_client_call_returned_perf_ns"]-
                                                                   dispatch_timing["osc_client_call_started_perf_ns"])/1e6
                        dispatch_timing["track_tcp_returned_perf_ns"] = time.perf_counter_ns()
                        dispatch_timing["track_tcp_ms"] = (dispatch_timing["track_tcp_returned_perf_ns"]-
                                                             dispatch_timing["dispatch_started_perf_ns"])/1e6
                        timing = result.get("track_tcp_timing") if isinstance(result, dict) else None
                        if isinstance(timing, dict):
                            self.a._event("osc_track_tcp_timing", tick=tick, **timing)
                        if not result.get("ok"):
                            raise RuntimeError(f"RTC TCP rejected: {result}")
                        if not self.live():
                            return
                        if time.perf_counter() >= deadline+PERIOD:
                            raise RuntimeError("RTC TCP acknowledgement exceeded its slot")
                        dispatch_timing["stage"] = "gripper_enqueue"
                        self.a._send_gripper_if_needed(self.session, self.client, target.gripper)
                        if time.perf_counter() >= deadline+PERIOD:
                            raise RuntimeError("RTC control dispatch exceeded its slot")
                        self.timeline.complete(tick)
                        dispatch_timing["stage"] = "completed"
                        self.a._event("rtc_dispatch", tick=tick, deadline_perf_s=deadline, started_perf_s=started,
                                      completed_perf_s=time.perf_counter(), target_tcp=target.pose(), gripper_ratio=target.gripper)
                        with self.a.lock:
                            self.a.state["executed_steps"] += 1
                finally:
                    dispatch_timing["finished_perf_ns"] = time.perf_counter_ns()
                    dispatch_timing["total_ms"] = (dispatch_timing["finished_perf_ns"]-
                                                    dispatch_timing["row_lock_requested_perf_ns"])/1e6
                    self.a._event("rtc_dispatch_timing", **dispatch_timing)
                with self.timeline.lock:
                    remaining = len(self.timeline.targets)
                self.publish(next_tick=tick+1, queue_remaining_ms=remaining*50, state="RUNNING")
        except Exception as exc:
            self.fail(str(exc))

    def run(self):
        try:
            compatibility(self.policy.metadata, self.a.config["execution"])
            timing = RTCTiming.from_execution(self.a.config['execution'])
            self.publish(state="BOOTSTRAP", reason=None, manual_restart_required=False, hold_confirmed=False,
                inference_delay=timing.inference_delay, execute_horizon=timing.execute_horizon,
                prefix_attention_horizon=H)
            # Startup publishes the session immediately; the independent SDK
            # cache poll may need one cycle before measured gripper is ready.
            ready_by = time.perf_counter()+1
            while self.live():
                try:
                    observation, base, osc, stamp = self.assembler.read()
                    break
                except RuntimeError:
                    if time.perf_counter() >= ready_by:
                        raise
                    if self.cancel.wait(.02):
                        return
            else:
                return
            self.health = threading.Thread(target=self._health_loop, name="nero-model-rtc-health", daemon=True)
            self.health.start()
            self.timeline.origin_ns = stamp
            self.t0 = stamp/1e9
            initial_hold = Target.from_decoded((base, float(observation['observation/state'][6])))
            self.timeline.seed([initial_hold]*H)
            # The robot is actually holding while inference is in flight.
            # Guide startup against that real stationary plan, rather than
            # generating an unconstrained moving chunk and joining mid-motion.
            context = {"version": VERSION, "phase": "bootstrap_hold", "run_epoch": self.epoch, "request_id": 0,
                       "observation_tick": 0, "plan_revision": self.timeline.revision, "freeze_until": timing.inference_delay,
                       "observation_perf_ns": stamp, "timeline_origin_perf_ns": stamp,
                       "freeze_until_perf_ns": stamp+timing.inference_delay*PERIOD_NS, "action_row0_offset_s": 0,
                       "inference_delay": timing.inference_delay, "execute_horizon": timing.execute_horizon,
                       "prefix_attention_horizon": H,
                       "period_s": PERIOD, "absolute_prefix": [initial_hold.absolute_row() for _ in range(H)]}
            result = self.infer(observation, base, osc, context)
            if result is None:
                return
            targets, elapsed = result
            if time.perf_counter_ns() > context['freeze_until_perf_ns']:
                raise RuntimeError('RTC bootstrap inference_delay exceeded; remain in HOLD and increase configured delay')
            current_observation, current, _, _ = self.assembler.read()
            if np.linalg.norm(np.subtract(base["position_m"], current["position_m"])) > .002 or orientation_distance(base["orientation_xyzw"], current["orientation_xyzw"]) > math.radians(.5):
                raise RuntimeError("RTC bootstrap observation anchor moved; restart required")
            if time.perf_counter_ns() >= stamp+(H-1)*PERIOD_NS:
                raise RuntimeError("RTC bootstrap response has no future H20 targets")
            hold_target = Target.from_decoded((current, float(current_observation['observation/state'][6])))
            # Official splice: old HOLD, not approximately guided model rows,
            # owns all slots before inference_delay. Check the actual seam.
            targets = preserve_delay_prefix([hold_target]*H, targets, timing.inference_delay)
            now_ns = time.perf_counter_ns()
            self.timeline.seed(targets, now_ns, hold_target)
            join_tick = max(self.timeline.next_tick, timing.inference_delay)
            first_target = self.timeline.targets[join_tick]
            position_gap = float(np.linalg.norm(np.subtract(current['position_m'], first_target.position)))
            angle_gap = orientation_distance(current['orientation_xyzw'], first_target.quaternion)
            bridge_end = None
            if position_gap > .002 or angle_gap > math.radians(.5):
                targets, bridge_end = bootstrap_bridge(targets, join_tick, hold_target, osc.get('limits') or {})
                self.timeline.targets = {i: targets[i] for i in range(self.timeline.next_tick,H)}
            self.validate_targets([hold_target, *targets[self.timeline.next_tick:]], osc)
            self.publish(bootstrap_strategy='bounded_bridge' if bridge_end is not None else 'direct',
                         bootstrap_position_gap_m=position_gap, bootstrap_orientation_gap_rad=angle_gap,
                         bootstrap_bridge_end_tick=bridge_end)
            self.a._event('rtc_bootstrap_alignment', observation_perf_ns=stamp,
                          response_ready_perf_ns=now_ns, skipped_rows=self.timeline.next_tick,
                          position_gap_m=position_gap, orientation_gap_rad=angle_gap,
                          bridge_end_tick=bridge_end, scheduled_targets=[t.absolute_row() for t in targets],
                          first_target_perf_ns=stamp+self.timeline.next_tick*PERIOD_NS)
            # Replan immediately at the first future slot: bootstrap has already
            # consumed latency out of its H20 horizon, so do not wait another K.
            # First regular request uses the next global slot. Bootstrap's
            # original observation and bridge remain untouched.
            next_request_ns = next_grid_anchor(self.timeline.origin_ns, time.perf_counter_ns(), PERIOD_NS)
            with self.a.lock:
                if not self.live():
                    return
                self.a.state.update(priming=False, execution_enabled=True, state="RUNNING")
            self.executor = threading.Thread(target=self._execute, name="nero-model-rtc-executor", daemon=True)
            self.executor.start()
            while self.live():
                if next_request_ns is not None and self.cancel.wait(max(0, (next_request_ns-time.perf_counter_ns())/1e9)):
                    break
                request_started_ns = time.perf_counter_ns()
                aligned = self.assembler.read_aligned(next_request_ns, self.cancel)
                if aligned is None or not self.live():
                    break
                observation, base, osc, stamp = aligned
                observation_tick = (stamp-self.timeline.origin_ns)//PERIOD_NS
                context = self.timeline.context(observation_tick, timing.inference_delay, self.request_id+1,
                                                observation_ns=stamp, now_ns=time.perf_counter_ns(),
                                                minimum_freeze_tick=0 if bridge_end is None else bridge_end+1,
                                                execute_horizon=timing.execute_horizon)
                result = self.infer(observation, base, osc, context)
                if result is None:
                    break
                targets, elapsed = result
                with self.timeline.lock:
                    candidate, splice_evidence = self.timeline.merge_response(
                        context, targets, now_ns=time.perf_counter_ns(), metadata=self.policy.metadata)
                    ordered = [candidate[t] for t in sorted(candidate)]
                    previous = self.timeline.history.get(self.timeline.next_tick-1)
                    if previous is not None:
                        ordered.insert(0, previous)
                    self.validate_targets(ordered, osc)
                    self.timeline.targets = candidate
                    self.timeline.revision += 1
                self.a._event('rtc_splice', inference_delay=timing.inference_delay,
                    execute_horizon=timing.execute_horizon, prefix_attention_horizon=timing.prefix_attention_horizon,
                    observation_perf_ns=stamp, freeze_until_perf_ns=context['freeze_until_perf_ns'],
                    request_started_perf_ns=request_started_ns, next_request_perf_ns=stamp+timing.execute_horizon*PERIOD_NS,
                    first_future_tick=min(candidate), last_future_tick=max(candidate),
                    plan_revision=self.timeline.revision, **splice_evidence)
                next_request_ns = stamp+timing.execute_horizon*PERIOD_NS
                self.publish(inference_delay=timing.inference_delay, execute_horizon=timing.execute_horizon,
                    prefix_attention_horizon=timing.prefix_attention_horizon,
                    queue_remaining_ms=len(candidate)*50, plan_revision=self.timeline.revision)
        except Exception as exc:
            self.fail(f"RTC stopped: {exc}")
        finally:
            self.cancel.set()
            for thread in (self.executor, self.health):
                if thread and thread is not threading.current_thread():
                    thread.join(timeout=2)
            with self.a.lock:
                same_run = self.epoch == self.a._run_revision
            if same_run:
                self.a._finish_capture("RTC run exited")
            self.finished.set()
            # If a request/command is still running, start() rejects until this
            # run has exited and HOLD is confirmed. Never reuse its queue.


def run_adapter(adapter):
    """One connection owner alternates read-only previews and RTC control."""
    from .inference_preview import preview_once
    while not adapter.stop_event.wait(.02):
        with adapter.lock:
            active = bool(adapter.state.get("priming") or adapter.state.get("execution_enabled"))
            epoch = adapter._run_revision
            policy = adapter._policy
            session, client = adapter.state.get("session_id"), adapter.state.get("client_id")
            previous = adapter._rtc_run
        if not active:
            preview_once(adapter)
            continue
        if policy is None or (previous and previous.epoch == epoch):
            continue
        run = RtcRun(adapter, epoch, session, client, policy)
        with adapter.lock:
            if epoch != adapter._run_revision or not (adapter.state.get("priming") or adapter.state.get("execution_enabled")):
                continue
            adapter._rtc_run = run
        run.run()
