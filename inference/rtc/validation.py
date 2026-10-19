"""No-motion sampler/codec checks and synchronized RTX latency measurements."""
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation

from codec import VERSION, encode_prefix
from sampling import guided_velocity, prefix_weights


def validate(policy, repeats=12):
    # For a zero field the clean prediction is x. Negative dt must move toward y.
    x, y = jnp.zeros((1, 20, 32)), jnp.ones((1, 20, 32))
    mask = jnp.zeros_like(x).at[:, :7, :7].set(1)
    v = guided_velocity(lambda z: z*0, x, .5, y, mask, 5.)
    next_x = x-.1*v
    assert float(jnp.sum((mask*(y-next_x))**2)) < float(jnp.sum((mask*(y-x))**2))
    assert np.count_nonzero(np.asarray(next_x)[:, :, 7:]) == 0
    w = np.asarray(prefix_weights(7, 13))
    assert np.all(w[:7] == 1) and np.all(w[13:] == 0) and np.all(np.diff(w) <= 0)
    rng = np.random.default_rng(11649)
    for angle in (0, 1e-8, np.pi-1e-8, np.pi):
        start = Rotation.from_rotvec([angle, 0, 0])
        relative = Rotation.from_rotvec([.1, -.2, .05])
        target = relative*start
        state = np.asarray([.1, .2, .3, *start.as_rotvec(), .5])
        physical = encode_prefix([[.12, .19, .31, *target.as_quat(), .7]], state)[0]
        np.testing.assert_allclose(physical[:3], [.02, -.01, .01], atol=1e-7)
        assert (Rotation.from_rotvec(physical[3:6])*relative.inv()).magnitude() < 1e-6
    obs = {"observation/image": np.zeros((224, 224, 3), np.uint8),
           "observation/wrist_image": np.zeros((224, 224, 3), np.uint8),
           "observation/state": np.asarray([0, 0, .3, 0, 0, 0, .5], np.float32),
           "prompt": "Pick up the yellow cube."}
    context = {"version": VERSION, "run_epoch": 1, "request_id": 1, "observation_tick": 7,
               "observation_perf_ns": 350_000_000, "timeline_origin_perf_ns": 0,
               "freeze_until_perf_ns": 700_000_000, "action_row0_offset_s": 0,
               "plan_revision": 1, "freeze_until": 14, "period_s": .05, "phase": "replan",
               "absolute_prefix": [[0, 0, .3, 0, 0, 0, 1, .5]]*13}
    noise = rng.normal(size=(20, 32)).astype(np.float32)
    key = policy.policy._rng
    plain = policy.infer(obs, noise=noise)["actions"]
    policy.policy._rng = key
    zero = policy.infer(dict(obs, rtc_context=context), noise=noise, guidance=False)["actions"]
    np.testing.assert_array_equal(plain, zero)
    assert plain.shape == (20, 7) and np.isfinite(plain).all()
    policy.policy._rng = key
    guided = policy.infer(dict(obs, rtc_context=context), noise=noise)["actions"]
    # Exercise the production fixed D/E request, including a full physical
    # prefix with a shorter guidance overlap (the two lengths must not mix).
    fixed = dict(context, inference_delay=7, execute_horizon=10, prefix_attention_horizon=10)
    fixed['absolute_prefix'] = context['absolute_prefix']+[context['absolute_prefix'][-1]]*7
    assert policy.infer(dict(obs, rtc_context=fixed), noise=noise)['actions'].shape == (20, 7)
    desired = encode_prefix(context["absolute_prefix"], obs["observation/state"])
    expected = policy.normalize({"actions": desired})["actions"]
    plain_normalized = policy.normalize({"actions": plain[:len(desired)]})["actions"]
    guided_normalized = policy.normalize({"actions": guided[:len(desired)]})["actions"]
    weights = np.asarray(prefix_weights(7, 13))[:len(desired), None]
    plain_error = float(np.sum(weights*(plain_normalized-expected)**2))
    guided_error = float(np.sum(weights*(guided_normalized-expected)**2))
    assert guided_error < plain_error, (plain_error, guided_error)
    times = {}
    for mode in ("normal", "rtc"):
        samples = []
        for i in range(repeats):
            context["request_id"] = i+2
            began = time.perf_counter()
            result = policy.infer(obs if mode == "normal" else dict(obs, rtc_context=context))
            samples.append((time.perf_counter()-began)*1000)
            assert np.isfinite(result["actions"]).all()
        times[mode] = {"samples_ms": samples, "p50_ms": float(np.percentile(samples, 50)),
                       "p95_ms": float(np.percentile(samples, 95)), "max_ms": max(samples)}
    return {"toy_guidance_sign": "passed", "prefix_weights_mask": "passed", "so3_roundtrip": "passed",
            "zero_guidance_bitwise_equal": True, "real_model_weighted_prefix_error": {"normal": plain_error, "rtc": guided_error},
            "actions_shape": [20, 7], "latency": times,
            "hardware_commands_sent": 0, "observations": "synthetic RGB/state; not task performance validation"}
