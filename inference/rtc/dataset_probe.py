"""No-motion, real-dataset RTC stage probe.

This isolates the exact sequence used by a live replan: make an old normal
model chunk at t, advance it by E rows, rebase that physical prefix at t+E,
and inspect the guided response against the same prefix.  It never opens OSC.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import einops
import jax
import jax.numpy as jnp
import numpy as np
from openpi.models import model as model_module
from openpi.models.pi0 import make_attn_mask

from codec import HORIZON, PERIOD, encode_prefix, pack_normalized_prefix
from sampling import prefix_weights
from serve import RtcPolicy


def absolute_from_action(actions: np.ndarray, state: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    actions = np.asarray(actions, dtype=np.float64)
    result = np.empty((len(actions), 8), dtype=np.float64)
    result[:, :3] = state[:3] + actions[:, :3]
    base = Rotation.from_rotvec(state[3:6])
    result[:, 3:7] = np.asarray([(Rotation.from_rotvec(row[3:6]) * base).as_quat() for row in actions])
    result[:, 7] = actions[:, 6]
    return result


def load_observation(root: Path, episode: int, index: int):
    import pyarrow.parquet as pq

    table = pq.read_table(root / "data" / "chunk-000" / f"episode_{episode:06d}.parquet")
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    frames = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
    task_index = int(table["task_index"][0].as_py())
    if not 0 <= index < len(state):
        raise ValueError(f"episode {episode} has {len(state)} rows; requested {index}")
    tasks = {int(row["task_index"]): row["task"] for row in
             (json.loads(line) for line in (root / "meta" / "tasks.jsonl").read_text(encoding="utf-8").splitlines())}
    paths = {
        "observation/image": root / "videos" / "chunk-000" / "observation.images.base_0_rgb" / f"episode_{episode:06d}.mp4",
        "observation/wrist_image": root / "videos" / "chunk-000" / "observation.images.left_wrist_0_rgb" / f"episode_{episode:06d}.mp4",
    }
    images = {}
    for key, path in paths.items():
        capture = cv2.VideoCapture(str(path))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frames[index]))
            ok, bgr = capture.read()
            if not ok or bgr is None:
                raise RuntimeError(f"video decode failed for {key}, frame {frames[index]}")
            images[key] = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (224, 224), interpolation=cv2.INTER_AREA).astype(np.uint8)
        finally:
            capture.release()
    return {**images, "observation/state": state[index], "prompt": tasks[task_index]}, int(frames[index])


def context(prefix: np.ndarray, observation_tick: int, delay: int, execute: int):
    origin = 1_000_000_000
    observation_ns = origin + observation_tick * 50_000_000
    return {"version": "nero.rtc.v2", "run_epoch": 1, "request_id": 1,
            "observation_tick": observation_tick, "plan_revision": 1,
            "freeze_until": observation_tick + delay, "phase": "replan",
            "observation_perf_ns": observation_ns, "timeline_origin_perf_ns": origin,
            "freeze_until_perf_ns": observation_ns + delay * 50_000_000,
            "action_row0_offset_s": 0, "period_s": PERIOD,
            "inference_delay": delay, "execute_horizon": execute,
            "prefix_attention_horizon": HORIZON-execute,
            "absolute_prefix": prefix.tolist()}


def position_metrics(actual: np.ndarray, reference: np.ndarray, delay: int):
    error = actual[:, :3] - reference[:, :3]
    norms = np.linalg.norm(error, axis=1)
    def section(values):
        return {"rmse_mm": float(np.sqrt(np.mean(values**2))*1000), "max_mm": float(np.max(values)*1000)}
    return {"hard": section(norms[:delay]), "soft": section(norms[delay:]),
            "all": section(norms)}


def finite_difference_vjp(policy, observation, prefix, weights, noise):
    """Check the real-sample IIGDM VJP with AD and central differences.

    The policy is BF16 internally, so a single tiny perturbation is not a
    reliable numerical derivative.  Report both the JVP/VJP adjoint identity
    and a scale sweep along the guidance and an independent direction.
    """
    model = policy.model
    processed = model_module.preprocess_observation(None, observation, train=False)
    tokens, mask, ar_mask = model.embed_prefix(processed)
    _, cache = model.PaliGemma.llm([tokens, None], mask=make_attn_mask(mask, ar_mask),
                                   positions=jnp.cumsum(mask, axis=1)-1)
    t = jnp.asarray(1.0, dtype=jnp.float32)

    def velocity(z):
        suffix, suffix_mask, suffix_ar, cond = model.embed_suffix(processed, z, jnp.broadcast_to(t, z.shape[0]))
        attention = jnp.concatenate([einops.repeat(mask, "b p -> b s p", s=suffix.shape[1]),
                                     make_attn_mask(suffix_mask, suffix_ar)], axis=-1)
        positions = jnp.sum(mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1)-1
        (_, out), _ = model.PaliGemma.llm([None, suffix], mask=attention, positions=positions,
                                          kv_cache=cache, adarms_cond=[None, cond])
        return model.action_out_proj(out[:, -model.action_horizon:])

    def denoiser(z):
        v = velocity(z)
        return z - t*v, v

    x = jnp.asarray(noise).reshape(1, HORIZON, 32)
    clean, pullback, _ = jax.vjp(denoiser, x, has_aux=True)
    residual = (jnp.asarray(prefix) - clean) * jnp.asarray(weights)
    correction = pullback(residual)[0]
    correction_norm = jnp.linalg.norm(correction)
    guidance_direction = correction / jnp.maximum(correction_norm, 1e-12)
    random_direction = jnp.asarray(
        np.linspace(-1.0, 1.0, x.size, dtype=np.float32).reshape(x.shape)
    )
    random_direction = random_direction / jnp.maximum(jnp.linalg.norm(random_direction), 1e-12)
    scalar = lambda z: jnp.sum(denoiser(z)[0] * residual)
    rows = []
    for name, direction in (("guidance", guidance_direction), ("independent", random_direction)):
        jvp_clean = jax.jvp(lambda z: denoiser(z)[0], (x,), (direction,))[1]
        analytic = jnp.sum(correction * direction)
        jvp_directional = jnp.sum(jvp_clean * residual)
        scales = []
        for epsilon in (1.0, 5e-1, 2e-1, 1e-1, 5e-2, 3e-2, 1e-2):
            plus = scalar(x + epsilon*direction)
            minus = scalar(x - epsilon*direction)
            finite = (plus - minus) / (2*epsilon)
            scales.append({"epsilon": epsilon, "finite_difference": float(finite),
                           "relative_error": float(jnp.abs(finite-analytic) / jnp.maximum(jnp.abs(analytic), 1e-8))})
        rows.append({"direction": name, "vjp_directional": float(analytic),
                     "jvp_directional": float(jvp_directional),
                     "vjp_jvp_relative_error": float(jnp.abs(analytic-jvp_directional) /
                                                       jnp.maximum(jnp.abs(analytic), 1e-8)),
                     "finite_difference": scales})
    return {"t": 1.0, "correction_l2": float(correction_norm),
            "weighted_residual_l2": float(jnp.linalg.norm(residual)), "directions": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--delay", type=int, default=7)
    parser.add_argument("--execute", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.delay <= args.execute < HORIZON:
        parser.error("require 0 <= delay <= execute < H20")

    old_obs, old_frame = load_observation(args.dataset, args.episode, args.index)
    new_obs, new_frame = load_observation(args.dataset, args.episode, args.index + args.execute)
    rng = np.random.default_rng(11649)
    old_noise = rng.normal(size=(20, 32)).astype(np.float32)
    new_noise = rng.normal(size=(20, 32)).astype(np.float32)
    policy = RtcPolicy(args.checkpoint, steps=10, cap=5.)
    original_cap, original_steps, original_rng = policy.cap, policy.steps, policy.policy._rng
    try:
        policy.policy._rng = original_rng
        old_actions = np.asarray(policy.infer(old_obs, noise=old_noise)["actions"])
        old_absolute = absolute_from_action(old_actions, old_obs["observation/state"])
        reference = old_absolute[args.execute:]
        expected = encode_prefix(reference, new_obs["observation/state"])
        roundtrip = absolute_from_action(expected, new_obs["observation/state"])
        codec_error_mm = float(np.max(np.linalg.norm(roundtrip[:, :3] - reference[:, :3], axis=1))*1000)
        normalized = policy.normalize({"actions": expected})["actions"]
        packed = pack_normalized_prefix(normalized)
        transformed = policy.policy._input_transform(new_obs)
        transformed_batch = jax.tree.map(lambda value: jnp.asarray(value)[None, ...], transformed)
        normalization_roundtrip = np.asarray(policy.policy._output_transform(
            {"state": np.asarray(transformed_batch["state"][0]), "actions": np.asarray(packed[0])}
        )["actions"])
        normalization_error_mm = float(np.max(np.linalg.norm(
            normalization_roundtrip[:len(expected), :3] - expected[:, :3], axis=1
        ))*1000)
        weights = np.zeros((1, HORIZON, 32), np.float32)
        weights[0, :, :7] = np.asarray(prefix_weights(args.delay, HORIZON-args.execute))[:, None]
        model_observation = model_module.Observation.from_dict(transformed_batch)
        finite_difference = finite_difference_vjp(policy, model_observation, packed, weights, new_noise)
        report = {"episode": args.episode, "old_index": args.index, "new_index": args.index + args.execute,
                  "old_frame": old_frame, "new_frame": new_frame, "delay": args.delay, "execute": args.execute,
                  "prefix_rows": len(reference), "codec_roundtrip_max_mm": codec_error_mm,
                  "normalization_roundtrip_max_mm": normalization_error_mm,
                  "vjp_finite_difference": finite_difference, "variants": []}
        for cap, steps in ((5., 10), (20., 10), (100., 10), (200., 10), (5., 100)):
            policy.cap, policy.steps, policy.policy._rng = cap, steps, original_rng
            reply = policy.infer(dict(new_obs, rtc_context=context(reference, new_frame, args.delay, args.execute)), noise=new_noise)
            response = np.asarray(reply["actions"])
            policy.policy._rng = original_rng
            _, trace = policy.rtc_trace(policy.policy._rng, model_observation, jnp.asarray(packed), jnp.asarray(weights),
                                        num_steps=steps, cap=cap, noise=jnp.asarray(new_noise).reshape(1, HORIZON, 32))
            trace = jax.device_get(trace)
            report["variants"].append({"cap": cap, "steps": steps,
                                       "position_error": position_metrics(response[:len(expected)], expected, args.delay),
                                       "iigdm_trace": {key: np.asarray(value).reshape(-1).tolist() for key, value in trace.items()},
                                       "service_reported": reply.get("rtc", {})})
    finally:
        policy.cap, policy.steps, policy.policy._rng = original_cap, original_steps, original_rng
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
