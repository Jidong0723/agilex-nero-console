"""CPV reference owned and committed only by the successful output sender.

OSC observes snapshots and proposes a one-cycle step (velocity and interval),
never an independently integrated history. Shadow uses this same reference.
"""
from __future__ import annotations

import math
import threading


def _vector(values):
    if not isinstance(values, (list, tuple)) or len(values) != 7:
        raise ValueError("CPV reference requires seven values")
    result = [float(v) for v in values]
    if not all(math.isfinite(v) for v in result):
        raise ValueError("CPV reference values must be finite")
    return result


class CpvReference:
    def __init__(self):
        self._lock = threading.Lock()
        self._state = None

    def snapshot(self):
        with self._lock:
            return None if self._state is None else {
                k: list(v) if isinstance(v, list) else v for k, v in self._state.items()
            }

    def prepare(self, command, velocity, *, started_perf_ns, epoch,
                max_speed, max_acceleration):
        desired = _vector(velocity)
        state = self.snapshot()
        reset = state is None or state['epoch'] != epoch or state['reference_id'] != command['reference_id']
        proposal_dt = float(command['dt_s'])
        if not math.isfinite(proposal_dt) or not .001 <= proposal_dt <= .2:
            raise RuntimeError(f"invalid CPV proposal interval: {proposal_dt}")
        dt = proposal_dt if reset else (started_perf_ns-state['started_perf_ns'])/1e9
        if not math.isfinite(dt) or not 0 < dt <= .2:
            raise RuntimeError(f"invalid CPV reference interval: {dt}")
        q = _vector(command['anchor_position_rad']) if reset else state['position_rad']
        prior = [0.0]*7 if reset else state['velocity_rad_s']
        speed, acc = float(max_speed), float(max_acceleration)
        if not math.isfinite(speed) or not math.isfinite(acc) or speed <= 0 or acc <= 0:
            raise ValueError("CPV reference needs finite positive motion limits")
        # A delayed position batch is not permission to extrapolate the Pink
        # step over the whole gap. Bound its displacement to one proposal,
        # then apply acceleration against the actual successful-send history.
        step_scale = min(1.0, proposal_dt/dt)
        requested = [want*step_scale for want in desired]
        v = [max(old-acc*dt, min(old+acc*dt, max(-speed, min(speed, want))))
             for old, want in zip(prior, requested)]
        # The measured-state final safety gate must not be undone by shaping.
        mask = command.get('gate_mask', [False]*7)
        if len(mask) != 7:
            raise ValueError("invalid CPV reference safety mask")
        v = [want if gated else x for gated,want,x in zip(mask,requested,v)]
        if any(abs(x)>speed+1e-9 for x in v):
            raise RuntimeError("CPV gated reference exceeds speed limit")
        position = [p+x*dt for p,x in zip(q,v)]
        lower, upper = _vector(command['lower_rad']), _vector(command['upper_rad'])
        if any(lo>=hi for lo,hi in zip(lower,upper)):
            raise ValueError("invalid CPV reference position limits")
        if any((p<lo and p<old-1e-9) or (p>hi and p>old+1e-9)
               for p,old,lo,hi in zip(position,q,lower,upper)):
            raise RuntimeError("CPV reference moves outside joint limits")
        return dict(position_rad=position, velocity_rad_s=v,
                    acceleration_rad_s2=[(x-old)/dt for x,old in zip(v,prior)],
                    started_perf_ns=started_perf_ns, epoch=int(epoch),
                    reference_id=command['reference_id'], dt_s=dt,
                    limited=any(abs(x-y)>1e-9 for x,y in zip(v,desired)))

    def commit(self, candidate):
        with self._lock:
            self._state = {k: list(v) if isinstance(v,list) else v for k,v in candidate.items()}
