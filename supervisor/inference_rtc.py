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

VERSION = "nero.rtc.v1"
H = 20
PERIOD = .05


def orientation_distance(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    dot = abs(float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b))))
    return 2*math.acos(min(1.0, dot))


def compatibility(metadata, execution):
    if metadata.get("rtc_version") != VERSION or metadata.get("ready") is not True:
        raise RuntimeError("RTC requires a ready nero.rtc.v1 server; no normal-mode fallback")
    if metadata.get("state_dim") != 7 or metadata.get("valid_action_dim") != 7 or metadata.get("action_horizon") != H:
        raise RuntimeError("RTC server must declare state7/action7/H20")
    if metadata.get("period_s") != PERIOD or metadata.get("action_contract") != "base_relative_observation_start_rotvec_gripper_absolute":
        raise RuntimeError("RTC server action/time contract mismatch")
    if execution["expected_chunk_steps"] != H or execution.get("period_s", PERIOD) != PERIOD:
        raise RuntimeError("RTC requires H20 and 50 ms")


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


class Timeline:
    """Small atomic timeline; immutable prefix ownership remains LOCAL."""
    def __init__(self, epoch):
        self.epoch = epoch
        self.revision = 0
        self.next_tick = 0
        self.committed_until = 0
        self.targets = {}
        self.lock = threading.RLock()

    def seed(self, targets):
        if len(targets) != H:
            raise ValueError("Expected full H20 seed")
        with self.lock:
            self.targets = dict(enumerate(targets))
            self.revision += 1

    def context(self, observation_tick, delay, request_id):
        with self.lock:
            prefix = []
            tick = observation_tick
            while tick in self.targets and len(prefix) < H:
                prefix.append(self.targets[tick].absolute_row())
                tick += 1
            if len(prefix) <= delay:
                raise RuntimeError("RTC runway insufficient for another inference")
            return {"version": VERSION, "phase": "replan", "run_epoch": self.epoch,
                    "request_id": request_id, "observation_tick": observation_tick,
                    "plan_revision": self.revision, "freeze_until": observation_tick+delay,
                    "period_s": PERIOD, "absolute_prefix": prefix}

    def merged(self, context, targets):
        """Build a candidate while locked; caller validates it before publish."""
        if context["run_epoch"] != self.epoch or context["plan_revision"] != self.revision:
            raise RuntimeError("Obsolete RTC plan")
        start = context["observation_tick"]
        cut = max(context["freeze_until"], self.committed_until, self.next_tick)
        if len(targets) != H or cut >= start+H:
            raise RuntimeError("RTC response has no usable future")
        result = {t: v for t, v in self.targets.items() if self.next_tick <= t < cut}
        result.update({start+i: v for i, v in enumerate(targets) if start+i >= cut})
        if any(t not in result for t in range(self.next_tick, start+H)):
            raise RuntimeError("RTC splice would create a queue gap")
        return result

    def commit_next(self):
        with self.lock:
            tick = self.next_tick
            if tick not in self.targets:
                raise RuntimeError("RTC queue exhausted")
            self.committed_until = tick+1
            return tick, self.targets[tick]

    def complete(self, tick):
        with self.lock:
            if tick != self.next_tick:
                raise RuntimeError("RTC dispatch order mismatch")
            self.targets.pop(tick, None)
            self.next_tick += 1


class ObservationAssembler:
    def __init__(self, adapter):
        self.a = adapter
        self._gripper_marker = None
        self._gripper_received_ns = 0
        self._gripper_has_advanced = False
        self._gripper_lock = threading.Lock()

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
            osc = copy.deepcopy(a._osc_snapshot)
            published = a._log_feedback_snapshot.get("cache_published_perf_ns", 0)
        monotonic_now = time.monotonic_ns()
        now = time.perf_counter_ns()
        session = osc.get("session") or {}
        if session.get("state") != "ACTIVE" or (session.get("id"), session.get("client_id")) != (a.state.get("session_id"), a.state.get("client_id")):
            raise RuntimeError("RTC control session no longer belongs to this run")
        execution = osc.get("execution") or {}
        measured = osc.get('inference_observation')
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
            osc['gripper'] = dict(osc.get('gripper') or {}, width_m=measured['gripper_width_m'])
        if session.get("execution_mode") == "hardware":
            if not isinstance(stamp, int) or not 0 <= now-stamp <= 100_000_000:
                raise RuntimeError("RTC measured TCP timestamp missing or stale")
            if float(execution.get("feedback_age_s", math.inf)) > .1:
                raise RuntimeError("RTC hardware feedback is stale")
            diagnostic = osc.get("gripper_observation") or {}
            if diagnostic.get("error") or diagnostic.get("sdk_timestamp") is None:
                raise RuntimeError("RTC requires timestamped measured gripper")
            marker = diagnostic["sdk_timestamp"]
            with self._gripper_lock:
                if self._gripper_marker is None:
                    self._gripper_marker, self._gripper_received_ns = marker, now
                elif marker > self._gripper_marker:
                    self._gripper_has_advanced = True
                    self._gripper_marker, self._gripper_received_ns = marker, now
                if not self._gripper_has_advanced:
                    raise RuntimeError("RTC waiting for a new measured gripper sample")
                if now-self._gripper_received_ns > 150_000_000:
                    raise RuntimeError("RTC measured gripper has stopped updating")
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
        return observation, copy.deepcopy(pose), osc, now


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
        self.delay_budget = .3
        self.last_request_tick = 0
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
        self.a._event("rtc_request", context=context, observation_state=observation["observation/state"], base_tcp=base)
        response = self.policy.infer(dict(observation, rtc_context=context))
        if not self.live():
            return None
        elapsed = time.perf_counter()-started
        echo = response.get("rtc", {})
        for key in ("version", "phase", "run_epoch", "request_id", "observation_tick", "plan_revision", "freeze_until"):
            if echo.get(key) != context[key]:
                raise RuntimeError(f"RTC response identity mismatch: {key}")
        expected_sampler = "bootstrap_normal" if context["phase"] == "bootstrap" else "rtc_vjp"
        if echo.get("sampler") != expected_sampler:
            raise RuntimeError("RTC sampler mismatch; normal fallback is prohibited")
        rows = action_rows(response, H)
        targets = [Target.from_decoded(self.a._target_from_action(row, base, action_index=i)) for i, row in enumerate(rows)]
        self.validate_targets(targets, osc)
        with self.a.lock:
            if not self.live():
                return None
            self.a.state.update(action_chunk=rows, absolute_tcp_chunk=[t.pose() for t in targets],
                                chunk_source="control", chunk_updated_at=time.time(),
                                inference_base_tcp=base, action_chunk_length=H, last_response_steps=H,
                                inference_ms=elapsed*1000, chunk_sequence=self.a.state["chunk_sequence"]+1,
                                protocol_compatibility={"status": "compatible", "message": "RTC 7D/H20 响应验证通过"})
        self.a._event("rtc_response", context=context, elapsed_ms=elapsed*1000, server_timing=response.get("policy_timing"))
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
                if self.cancel.wait(max(0, deadline-time.perf_counter())):
                    return
                if time.perf_counter() >= deadline+PERIOD:
                    raise RuntimeError("RTC dispatch missed its 50 ms slot")
                with self.a._row_lock:
                    if not self.live():
                        return
                    if self.last_dispatch_started is not None and time.perf_counter()-self.last_dispatch_started < PERIOD/2:
                        raise RuntimeError("RTC dispatch jitter would require a catch-up burst")
                    tick, target = self.timeline.commit_next()
                    started = time.perf_counter()
                    self.last_dispatch_started = started
                    result = self.a.osc.track_tcp(self.session, self.client, self.a._next_sequence(), target.pose())
                    if not result.get("ok"):
                        raise RuntimeError(f"RTC TCP rejected: {result}")
                    if not self.live():
                        return
                    if time.perf_counter() >= deadline+PERIOD:
                        raise RuntimeError("RTC TCP acknowledgement exceeded its slot")
                    self.a._send_gripper_if_needed(self.session, self.client, target.gripper)
                    if time.perf_counter() >= deadline+PERIOD:
                        raise RuntimeError("RTC control dispatch exceeded its slot")
                    self.timeline.complete(tick)
                    self.a._event("rtc_dispatch", tick=tick, deadline_perf_s=deadline, started_perf_s=started,
                                  completed_perf_s=time.perf_counter(), target_tcp=target.pose(), gripper_ratio=target.gripper)
                    with self.a.lock:
                        self.a.state["executed_steps"] += 1
                with self.timeline.lock:
                    remaining = len(self.timeline.targets)
                self.publish(next_tick=tick+1, queue_remaining_ms=remaining*50, state="RUNNING")
        except Exception as exc:
            self.fail(str(exc))

    def run(self):
        try:
            compatibility(self.policy.metadata, self.a.config["execution"])
            self.publish(state="BOOTSTRAP", reason=None, manual_restart_required=False, hold_confirmed=False)
            # Startup publishes the session immediately; the independent SDK
            # cache poll may need one cycle before measured gripper is ready.
            ready_by = time.perf_counter()+1
            while self.live():
                try:
                    observation, base, osc, _ = self.assembler.read()
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
            context = {"version": VERSION, "phase": "bootstrap", "run_epoch": self.epoch, "request_id": 0,
                       "observation_tick": 0, "plan_revision": 0, "freeze_until": 0,
                       "period_s": PERIOD, "absolute_prefix": []}
            result = self.infer(observation, base, osc, context)
            if result is None:
                return
            targets, elapsed = result
            _, current, _, _ = self.assembler.read()
            if np.linalg.norm(np.subtract(base["position_m"], current["position_m"])) > .002 or orientation_distance(base["orientation_xyzw"], current["orientation_xyzw"]) > math.radians(.5):
                raise RuntimeError("RTC bootstrap observation anchor moved; restart required")
            self.delay_budget = max(.3, elapsed)
            initial_delay = math.ceil(self.delay_budget/PERIOD)+1
            if max(int(self.a.config["execution"].get("rtc_min_replan_steps", 6)), initial_delay)+initial_delay >= H:
                raise RuntimeError("RTC bootstrap latency leaves insufficient H20 runway")
            self.timeline.seed(targets)
            with self.a.lock:
                if not self.live():
                    return
                self.a.state.update(priming=False, execution_enabled=True, state="RUNNING")
            self.t0 = time.perf_counter()+PERIOD
            self.executor = threading.Thread(target=self._execute, name="nero-model-rtc-executor", daemon=True)
            self.executor.start()
            while self.live():
                delay = math.ceil(self.delay_budget/PERIOD)+1
                interval = max(int(self.a.config["execution"].get("rtc_min_replan_steps", 6)), delay)
                if interval+delay >= H:
                    raise RuntimeError("RTC measured latency leaves insufficient H20 runway")
                request_tick = self.last_request_tick+interval
                if self.cancel.wait(max(0, self.t0+request_tick*PERIOD-time.perf_counter())):
                    break
                observation, base, osc, stamp = self.assembler.read()
                observation_tick = max(request_tick, math.ceil((stamp/1e9-self.t0)/PERIOD))
                context = self.timeline.context(observation_tick, delay, self.request_id+1)
                actual_interval = observation_tick-self.last_request_tick
                self.last_request_tick = observation_tick
                result = self.infer(observation, base, osc, context)
                if result is None:
                    break
                targets, elapsed = result
                if time.perf_counter() >= self.t0+context["freeze_until"]*PERIOD:
                    raise RuntimeError("RTC response missed its frozen-prefix deadline")
                with self.timeline.lock:
                    candidate = self.timeline.merged(context, targets)
                    ordered = [candidate[t] for t in sorted(candidate)]
                    self.validate_targets(ordered, osc)
                    self.timeline.targets = candidate
                    self.timeline.revision += 1
                self.delay_budget = max(self.delay_budget, elapsed)
                self.publish(delay_slots=delay, replan_interval_steps=actual_interval, plan_revision=self.timeline.revision)
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
