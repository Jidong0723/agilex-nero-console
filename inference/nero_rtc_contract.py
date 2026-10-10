"""Pure RTC payload math; shares no controller or model state."""
import math
import numpy as np

PROTOCOL = 'nero.rtc.chunk_start.v1'


def rotvec_quaternion(v):
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.array([0.,0.,0.,1.])
    return np.array([*(np.asarray(v)*math.sin(angle/2)/angle), math.cos(angle/2)])


def quaternion_product(a,b):
    x,y,z,w=a; X,Y,Z,W=b
    return np.array([w*X+x*W+y*Z-z*Y, w*Y-x*Z+y*W+z*X,
                     w*Z+x*Y-y*X+z*W, w*W-x*X-y*Y-z*Z])


def quaternion_rotvec(q):
    q=np.asarray(q,dtype=np.float64);q=q/np.linalg.norm(q)
    if q[3]<0:q=-q
    sine=float(np.linalg.norm(q[:3]))
    return np.zeros(3) if sine<1e-12 else q[:3]*(2*math.atan2(sine,q[3])/sine)


def weights_for(delay, overlap, horizon):
    if not 0 <= delay <= overlap <= horizon:
        raise ValueError('invalid RTC delay/overlap')
    i = np.arange(horizon)
    w = np.clip((delay-1-i)/(overlap-delay+1)+1, 0, 1)
    return np.where(i < overlap, w*np.expm1(w)/(np.e-1), 0).astype(np.float32)


def unscale_targets(state, targets, scale):
    """Invert client multiplier, then pass through original training transforms."""
    state = np.asarray(state, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if state.shape not in ((7,), (8,)) or targets.ndim != 2 or targets.shape[1] != 7:
        raise ValueError('RTC expects state7/state8 and absolute targets Nx7')
    if not np.isfinite(state).all() or not np.isfinite(targets).all() or not 0 < scale <= 2:
        raise ValueError('RTC nonfinite prior or invalid multiplier')
    result = targets.copy()
    base = rotvec_quaternion(state[3:6])
    result[:, :3] = state[:3]+(targets[:, :3]-state[:3])/scale
    inverse = base*np.array([-1,-1,-1,1])
    for i,row in enumerate(targets):
        relative=quaternion_rotvec(quaternion_product(rotvec_quaternion(row[3:6]),inverse))/scale
        result[i,3:6]=quaternion_rotvec(quaternion_product(rotvec_quaternion(relative),base))
    return result.astype(np.float32)
