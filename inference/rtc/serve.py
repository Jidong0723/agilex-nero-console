"""Standalone 7D/H20 normal + RTC service; leaves the installed OpenPI tree intact."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import threading
import time
import types

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation
from openpi import transforms
from openpi.models import model as model_module
from openpi.models.pi0_config import Pi0Config
from openpi.policies.libero_policy import LiberoInputs
from openpi.policies.policy import Policy
from openpi_client import msgpack_numpy
from websockets.asyncio.server import serve as websocket_serve
from websockets.exceptions import ConnectionClosed
from openpi.shared import nnx_utils
from openpi.training.checkpoints import load_norm_stats
from openpi.training.config import ModelTransformFactory

from codec import VERSION, HORIZON, PERIOD, TIME_CONTRACT, encode_prefix, echo_context, validate_context, pack_normalized_prefix, compact_inputs, guidance_bounds
from sampling import prefix_weights, sample_rtc, sample_rtc_trace
from runtime import PreparedSampler, audit_normalization


class Inputs:
    def __init__(self, model_type):
        self.convert = LiberoInputs(model_type)

    def __call__(self, data):
        state = np.asarray(data["observation/state"], np.float32).copy()
        if state.shape != (7,) or not np.isfinite(state).all() or not 0 <= state[6] <= 1:
            raise ValueError("Expected finite 7D state with gripper in [0,1]")
        for key in ("observation/image", "observation/wrist_image"):
            image = np.asarray(data[key])
            if image.shape != (224, 224, 3) or image.dtype != np.uint8:
                raise ValueError(f"{key} must be RGB uint8 [224,224,3]")
        state[3:6] = Rotation.from_rotvec(state[3:6]).as_rotvec()
        return self.convert(dict(data, **{"observation/state": state}))


class RtcPolicy:
    def __init__(self, checkpoint, steps=10, cap=5.0, *, compute_dtype="bfloat16", parameter_dtype="bfloat16",
                 prefix_attention_schedule="exp", compact_prefix=False, terminal_noise_time=0.0, normal_steps=None,
                 terminal_refinement_steps=1, prefix_mask_mode='timestamp_frozen',
                 warmup_inference_delay=6, warmup_execute_horizon=6):
        if compute_dtype not in {"bfloat16", "float32"} or parameter_dtype not in {"bfloat16", "float32"}:
            raise ValueError("RTC dtype must be bfloat16 or float32")
        if prefix_attention_schedule not in {"exp", "linear", "ones", "zeros"}:
            raise ValueError("Unsupported RTC prefix attention schedule")
        if not 0 <= terminal_noise_time < 1 or (terminal_noise_time and steps < 2):
            raise ValueError('Invalid terminal noise time')
        if type(terminal_refinement_steps) is not int or not 1 <= terminal_refinement_steps <= max(1, steps-1):
            raise ValueError('Invalid terminal refinement steps')
        if terminal_refinement_steps > 1 and not 0 < terminal_noise_time < .02:
            raise ValueError('Multiple terminal steps require 0 < terminal_noise_time < .02')
        checkpoint = Path(checkpoint).resolve(strict=True)
        config = Pi0Config(pi05=True, action_dim=32, action_horizon=HORIZON, discrete_state_input=False,
                           dtype=compute_dtype)
        model = config.load(model_module.restore_params(checkpoint / "params", dtype=jnp.dtype(parameter_dtype)))
        # Retain the model for offline finite-difference diagnostics. Runtime
        # inference continues through ``self.policy`` and ``self.rtc_sample``.
        self.model = model
        asset_id = "nerovla/nero_chunk7_v1"
        stats = load_norm_stats(checkpoint / "assets", asset_id)
        for key in ("state", "actions"):
            if len(stats[key].mean) != 7:
                raise ValueError("Checkpoint assets are not state7/action7")
        self.normalize = transforms.Normalize(stats, use_quantiles=True)
        output = transforms.Unnormalize(stats, use_quantiles=True)
        normalization_audit = audit_normalization(self.normalize, output)
        group = ModelTransformFactory()(config)
        self.policy = Policy(model, transforms=[Inputs(config.model_type), self.normalize, *group.inputs],
                             output_transforms=[*group.outputs, output, lambda d: {"actions": d["actions"][..., :7]}],
                             sample_kwargs={"num_steps": steps if normal_steps is None else normal_steps})
        self.rtc_sample = nnx_utils.module_jit(types.MethodType(sample_rtc, model), static_argnames=("num_steps", "cap", "compact_prefix", "terminal_noise_time", "terminal_refinement_steps"))
        self.rtc_sample = PreparedSampler(self.rtc_sample, jax.tree.flatten, jax.block_until_ready)
        self.rtc_trace = nnx_utils.module_jit(types.MethodType(sample_rtc_trace, model), static_argnames=("num_steps", "cap", "compact_prefix", "terminal_noise_time", "terminal_refinement_steps"))
        self.steps, self.cap = steps, cap
        self.normal_steps = normal_steps if normal_steps is not None else steps
        self.prefix_attention_schedule = prefix_attention_schedule
        self.compact_prefix = bool(compact_prefix)
        self.terminal_noise_time = float(terminal_noise_time)
        self.terminal_refinement_steps = terminal_refinement_steps
        if prefix_mask_mode not in {'legacy_compressed','nominal','timestamp_frozen'}:
            raise ValueError('Invalid prefix mask mode')
        self.prefix_mask_mode = prefix_mask_mode
        if (type(warmup_inference_delay) is not int or type(warmup_execute_horizon) is not int
                or not 0 <= warmup_inference_delay <= warmup_execute_horizon < HORIZON
                or warmup_execute_horizon < 1 or warmup_inference_delay+warmup_execute_horizon > HORIZON):
            raise ValueError('Invalid warmup timing')
        self.warmup_delay,self.warmup_horizon=warmup_inference_delay,warmup_execute_horizon
        # Legacy ones/linear profiles deliberately restrict execution to the
        # old overlap. Exp uses the official D<=E, D+E<=H timing instead.
        self.requires_guided_execution = bool(terminal_noise_time and prefix_attention_schedule in {'ones', 'linear'})
        self.lock = threading.Lock()
        norm_file = checkpoint / "assets" / asset_id / "norm_stats.json"
        source_hash = hashlib.sha256()
        for name in ("codec.py", "sampling.py", "serve.py", "runtime.py", "validation.py", "start.sh"):
            source_hash.update(Path(__file__).with_name(name).read_bytes())
        self.metadata = {"state_dim": 7, "valid_action_dim": 7, "action_dim": 32,
                         "action_horizon": 20, "rtc_version": VERSION, "period_s": PERIOD,
                         "action_time_contract": TIME_CONTRACT, "action_row0_offset_s": 0,
                         "rtc_bootstrap_mode": "hold_prefix_v1",
                         "rtc_bootstrap_attention": "full_hold_v2",
                         "rtc_schedule": "fixed_delay_execute_horizon_v1",
                         "time_semantics_source": "operator_confirmation_11649_2026-10-18",
                         "action_contract": "base_relative_observation_start_rotvec_gripper_absolute",
                         "checkpoint_dir": str(checkpoint), "deployment_label": "NERO pi0.5 H20 step11649 normal + RTC",
                         "norm_stats_sha256": hashlib.sha256(norm_file.read_bytes()).hexdigest(),
                         "normalization_audit": normalization_audit,
                         "rtc_source_sha256": source_hash.hexdigest(), "num_steps": steps, "guidance_cap": cap,
                         "compute_dtype": compute_dtype, "parameter_dtype": parameter_dtype,
                         "prefix_attention_schedule": prefix_attention_schedule,
                         "compact_prefix": self.compact_prefix,
                         "rtc_integration_grid": ("terminal_refined_multistep_v2" if terminal_refinement_steps > 1
                                                  else "terminal_refined_v1") if terminal_noise_time else "uniform",
                         "terminal_noise_time": terminal_noise_time,
                         "terminal_refinement_steps": terminal_refinement_steps,
                         "rtc_prefix_mask_mode": prefix_mask_mode,
                         "normal_num_steps": steps if normal_steps is None else normal_steps,
                         "rtc_requires_execution_within_prefix": self.requires_guided_execution,
                         "rtc_late_response_policy": "guided_same_time_continuity_v1" if terminal_noise_time and prefix_attention_schedule in {'ones','linear','exp'} else None,
                         "rtc_warmup_timing": {"inference_delay":self.warmup_delay,"execute_horizon":self.warmup_horizon},
                         "training_contract_verified": False, "ready": False}
        # Record actual files, including the clone's uncommitted customizations.
        openpi_root = Path(model_module.__file__).resolve().parents[3]
        self.metadata["openpi_source_sha256"] = {
            str(p.relative_to(openpi_root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [openpi_root / "src/openpi/models/pi0.py", openpi_root / "src/openpi/models/pi0_config.py",
                      openpi_root / "src/openpi/transforms.py", openpi_root / "src/openpi/training/config.py",
                      openpi_root / "src/openpi/training/data_loader.py"]}

    def infer(self, obs, *, noise=None, guidance=True):
        with self.lock:
            return self._infer(obs, noise=noise, guidance=guidance)

    def _infer(self, obs, *, noise=None, guidance=True):
        started = time.perf_counter()
        obs = dict(obs)
        context = obs.pop("rtc_context", None)
        if context is None:
            result = self.policy.infer(obs, noise=noise)
            result["policy_timing"]["infer_ms"] = (time.perf_counter()-started)*1000
            return result
        delay, length = validate_context(context)
        mask_start, mask_end, _ = guidance_bounds(context, getattr(self,'prefix_mask_mode','timestamp_frozen'))
        if (getattr(self,'requires_guided_execution',False)
                and context['inference_delay']+2*context['execute_horizon'] > HORIZON):
            raise ValueError('Endpoint-refined guided RTC requires inference_delay + 2*execute_horizon <= H20; use D=6/E=6 for the validated profile')
        # Validate physical prefix even on the explicit zero-guidance path.
        physical = encode_prefix(context["absolute_prefix"], obs["observation/state"])
        if context["phase"] == "bootstrap" or not guidance:
            result = self.policy.infer(obs, noise=noise)
            sampler = "bootstrap_normal" if context["phase"] == "bootstrap" else "zero_guidance_normal"
        else:
            # Relative conversion happened above exactly once. Normalize directly.
            normalized = self.normalize({"actions": physical})["actions"]
            prefix = pack_normalized_prefix(normalized)
            weights = np.zeros((1, 20, 32), np.float32)
            schedule = 'ones' if context['phase']=='bootstrap_hold' else getattr(self, 'prefix_attention_schedule', 'exp')
            row_weights = np.asarray(prefix_weights(mask_start, mask_end, schedule=schedule), dtype=np.float32).copy()
            row_weights[length:] = 0
            weights[0, :, :7] = row_weights[:, None]
            inputs = self.policy._input_transform(obs)
            if getattr(self, 'compact_prefix', False):
                inputs = compact_inputs(inputs)
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
            self.policy._rng, key = jax.random.split(self.policy._rng)
            kwargs = {"num_steps": self.steps, "cap": self.cap}
            if getattr(self, 'compact_prefix', False):
                kwargs['compact_prefix'] = True
            if getattr(self, 'terminal_noise_time', 0.):
                kwargs['terminal_noise_time'] = self.terminal_noise_time
                kwargs['terminal_refinement_steps'] = getattr(self, 'terminal_refinement_steps', 1)
            if noise is not None:
                kwargs["noise"] = jnp.asarray(noise).reshape(1, 20, 32)
            prepared_at = time.perf_counter()
            actions = self.rtc_sample(key, model_module.Observation.from_dict(inputs),
                                      jnp.asarray(prefix), jnp.asarray(weights), **kwargs)
            jax.block_until_ready(actions)
            sampled_at = time.perf_counter()
            result = self.policy._output_transform({"state": np.asarray(inputs["state"][0]),
                                                   "actions": np.asarray(actions[0])})
            sampler = "bootstrap_hold_vjp" if context['phase'] == 'bootstrap_hold' else "rtc_vjp"
        actions = np.asarray(result["actions"])
        if actions.shape != (20, 7) or not np.isfinite(actions).all():
            raise ValueError("Invalid H20 action response")
        result["rtc"] = dict(echo_context(context), sampler=sampler, valid_prefix_steps=length)
        if length:
            relative = encode_prefix(context["absolute_prefix"], obs["observation/state"])[:length]
            # Return physical-unit evidence for each constrained row.  Both
            # action and reference are relative to this request's observation.
            delta = actions[:length] - relative
            effective_schedule = 'ones' if context['phase'] == 'bootstrap_hold' else getattr(self, 'prefix_attention_schedule', 'exp')
            effective_weights = (np.ones(HORIZON, dtype=np.float32) if context['phase'] == 'bootstrap_hold'
                                 else np.asarray(prefix_weights(mask_start, mask_end, schedule=effective_schedule), dtype=np.float32).copy())
            effective_weights[length:] = 0
            result["rtc"]["sampler_config"] = {
                "algorithm": "iigdm_vjp_decreasing_time_euler",
                "num_steps": int(self.steps),
                "normal_num_steps": int(getattr(self, 'normal_steps', self.steps)),
                "guidance_cap": float(self.cap),
                "prefix_attention_schedule": effective_schedule,
                "prefix_mask_mode": getattr(self,'prefix_mask_mode','timestamp_frozen'),
                "mask_start_row": int(mask_start), "mask_end_row": int(mask_end),
                "integration_grid": ("terminal_refined_multistep_v2" if getattr(self, 'terminal_refinement_steps', 1) > 1
                                     else "terminal_refined_v1") if getattr(self, 'terminal_noise_time', 0.) else "uniform",
                "terminal_noise_time": float(getattr(self, 'terminal_noise_time', 0.)),
                "terminal_refinement_steps": int(getattr(self, 'terminal_refinement_steps', 1)),
                "compact_prefix": bool(getattr(self, 'compact_prefix', False)),
                "valid_action_dim": 7,
            }
            result["rtc"]["prefix_residuals"] = [
                {
                    "row": int(row),
                    "guidance_weight": float(effective_weights[row]),
                    "region": "hard" if effective_weights[row] >= 1.0 - 1e-6 else ("soft" if effective_weights[row] > 0 else "unguided"),
                    "position_delta_m": delta[row, :3].astype(float).tolist(),
                    "position_error_m": float(np.linalg.norm(delta[row, :3])),
                    "orientation_delta_rotvec_rad": delta[row, 3:6].astype(float).tolist(),
                    "orientation_error_rad": float(np.linalg.norm(delta[row, 3:6])),
                    "gripper_delta": float(delta[row, 6]),
                }
                for row in range(length)
            ]
            result["rtc"]["prefix_position_rmse_m"] = float(np.sqrt(np.mean((actions[:length, :3]-relative[:, :3])**2)))
            result["rtc"]["prefix_gripper_rmse"] = float(np.sqrt(np.mean((actions[:length, 6]-relative[:, 6])**2)))
            hard = min(mask_start, length)
            if hard:
                result["rtc"]["hard_prefix_position_rmse_m"] = float(np.sqrt(np.mean((actions[:hard, :3]-relative[:hard, :3])**2)))
            if length > hard:
                result["rtc"]["soft_prefix_position_rmse_m"] = float(np.sqrt(np.mean((actions[hard:length, :3]-relative[hard:length, :3])**2)))
        result["policy_timing"] = {"infer_ms": (time.perf_counter()-started)*1000}
        if sampler in {'bootstrap_hold_vjp', 'rtc_vjp'} and getattr(self.rtc_sample, 'last', None):
            result['policy_timing'].update(self.rtc_sample.last)
            result['policy_timing'].update(prepare_ms=(prepared_at-started)*1000,
                postprocess_ms=(time.perf_counter()-sampled_at)*1000)
        return result

    def warmup(self):
        obs = {"observation/image": np.zeros((224, 224, 3), np.uint8),
               "observation/wrist_image": np.zeros((224, 224, 3), np.uint8),
               "observation/state": np.asarray([0, 0, .3, 0, 0, 0, .5], np.float32), "prompt": "Pick up the yellow cube."}
        self.infer(obs)
        self.infer(obs,noise=np.zeros((20,32),np.float32))
        context = {"version": VERSION, "run_epoch": 0, "request_id": 1, "observation_tick": 0,
                   "observation_perf_ns": 0, "timeline_origin_perf_ns": 0,
                   "freeze_until_perf_ns": 350_000_000, "action_row0_offset_s": 0,
                   "plan_revision": 0, "freeze_until": 7, "period_s": PERIOD, "phase": "replan",
                   "inference_delay": 7, "execute_horizon": 10, "prefix_attention_horizon": 10,
                   "absolute_prefix": [[0, 0, .3, 0, 0, 0, 1, .5]]*13}
        if self.requires_guided_execution:
            context.update(inference_delay=6,execute_horizon=6,prefix_attention_horizon=14,
                           freeze_until=6,freeze_until_perf_ns=300_000_000,
                           absolute_prefix=[[0,0,.3,0,0,0,1,.5]]*14)
        elif self.prefix_attention_schedule == 'exp':
            # Warm up the explicitly deployed profile. Request context
            # remains authoritative; no fixed server-side scheduling copy.
            context.update(inference_delay=self.warmup_delay,execute_horizon=self.warmup_horizon,
                           prefix_attention_horizon=HORIZON-self.warmup_horizon,
                           freeze_until=self.warmup_delay,freeze_until_perf_ns=self.warmup_delay*50_000_000,
                           absolute_prefix=[[0,0,.3,0,0,0,1,.5]]*(HORIZON-self.warmup_horizon))
        self.infer(dict(obs, rtc_context=context))
        hold_context = dict(context, phase='bootstrap_hold', observation_tick=0,
                            prefix_attention_horizon=20,
                            absolute_prefix=[[0,0,.3,0,0,0,1,.5]]*20)
        self.infer(dict(obs, rtc_context=hold_context))
        if self.compact_prefix:
            # Precompile every supported prompt bucket before advertising
            # readiness; a new task must not trigger an 8-second RTC compile.
            physical = encode_prefix(context['absolute_prefix'], obs['observation/state'])
            prefix = pack_normalized_prefix(self.normalize({'actions': physical})['actions'])
            weights = np.zeros((1, 20, 32), np.float32)
            weights[0, :, :7] = np.asarray(prefix_weights(context['inference_delay'],context['prefix_attention_horizon'],schedule=self.prefix_attention_schedule))[:, None]
            for size in (64, 128, 200):
                inputs = self.policy._input_transform(obs)
                inputs['tokenized_prompt_mask'] = np.arange(len(inputs['tokenized_prompt_mask'])) < size
                inputs = compact_inputs(inputs)
                inputs = jax.tree.map(lambda value: jnp.asarray(value)[None, ...], inputs)
                actions = self.rtc_sample(self.policy._rng, model_module.Observation.from_dict(inputs),
                    jnp.asarray(prefix), jnp.asarray(weights), num_steps=self.steps, cap=self.cap, compact_prefix=True,
                    terminal_noise_time=self.terminal_noise_time,
                    terminal_refinement_steps=self.terminal_refinement_steps)
                jax.block_until_ready(actions)
                # Optional noise changes the JAX pytree; warm it explicitly as
                # well, so a fixed-seed diagnostic cannot cold-compile online.
                actions = self.rtc_sample(self.policy._rng, model_module.Observation.from_dict(inputs),
                    jnp.asarray(prefix), jnp.asarray(weights), num_steps=self.steps, cap=self.cap, compact_prefix=True,
                    terminal_noise_time=self.terminal_noise_time,
                    terminal_refinement_steps=self.terminal_refinement_steps,
                    noise=jnp.zeros((1,20,32),jnp.float32))
                jax.block_until_ready(actions)
        else:
            # Non-compact deployments also accept explicit diagnostic noise.
            self.infer(dict(obs,rtc_context=context),noise=np.zeros((20,32),np.float32))
        self.rtc_sample.sealed = True
        self.metadata['rtc_signature_guard'] = 'warmed_shapes_and_static_options_v1'
        self.metadata['rtc_prepared_signature_count'] = len(self.rtc_sample.prepared)
        self.metadata["ready"] = True


class Server:
    """One active RTC socket and at most one GPU call; reject, never queue peers."""
    def __init__(self, policy, host, port):
        self.policy, self.host, self.port = policy, host, port
        self.owner = None
        self.busy = False

    async def handler(self, socket):
        packer = msgpack_numpy.Packer()
        await socket.send(packer.pack(self.policy.metadata))
        try:
            async for raw in socket:
                observation = msgpack_numpy.unpackb(raw)
                rtc = isinstance(observation, dict) and "rtc_context" in observation
                if (self.owner is not None and self.owner is not socket) or self.busy:
                    await socket.send("Inference service is busy; another RTC session owns the model")
                    await socket.close(code=1013, reason="Model busy")
                    return
                if rtc:
                    self.owner = socket
                self.busy = True
                try:
                    result = await asyncio.to_thread(self.policy.infer, observation)
                finally:
                    self.busy = False
                await socket.send(packer.pack(result))
        except ConnectionClosed:
            pass
        except Exception as exc:
            logging.exception("Inference request rejected")
            try:
                await socket.send(f"{type(exc).__name__}: {exc}")
                await socket.close(code=1011, reason="Inference rejected")
            except ConnectionClosed:
                pass
        finally:
            if self.owner is socket:
                self.owner = None

    async def run(self):
        async with websocket_serve(self.handler, self.host, self.port, compression=None, max_size=4*1024*1024,
                                   max_queue=1) as server:
            await server.serve_forever()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--normal-steps", type=int, default=10, help="Ordinary inference steps, independent of RTC tuning")
    parser.add_argument("--guidance-cap", type=float, default=5.0,
                        help="Official RTC max_guidance_weight (reference default: 5.0)")
    parser.add_argument("--prefix-attention-schedule", choices=("exp", "linear", "ones", "zeros"), default="exp",
                        help="Official get_prefix_weights schedule; bootstrap always uses ones")
    parser.add_argument("--compute-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--parameter-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--compact-prefix", action="store_true", help="Skip masked third camera and trailing prompt padding in RTC")
    parser.add_argument("--terminal-noise-time", type=float, default=0.,
                        help="0: official uniform integration; >0: explicit endpoint-refined Euler grid")
    parser.add_argument("--terminal-refinement-steps", type=int, default=1,
                        help="Opt-in geometric terminal substeps; 1 preserves terminal_refined_v1")
    parser.add_argument("--prefix-mask-mode", choices=('legacy_compressed','nominal','timestamp_frozen'), default='timestamp_frozen')
    parser.add_argument("--warmup-inference-delay", type=int, default=6)
    parser.add_argument("--warmup-execute-horizon", type=int, default=6)
    parser.add_argument("--validate", action="store_true", help="Run no-motion checks before serving")
    parser.add_argument("--training-contract", type=Path, help="Optional file-backed training provenance JSON; absence leaves verification false")
    args = parser.parse_args()
    if (not 1 <= args.steps <= 100 or not 1 <= args.normal_steps <= 100 or not 1 <= args.port <= 65535
            or not 0 < args.guidance_cap <= 1000):
        parser.error("Invalid steps, port, or guidance cap")
    logging.basicConfig(level=logging.INFO)
    policy = RtcPolicy(args.checkpoint, args.steps, args.guidance_cap,
                       compute_dtype=args.compute_dtype, parameter_dtype=args.parameter_dtype,
                       prefix_attention_schedule=args.prefix_attention_schedule, compact_prefix=args.compact_prefix,
                       terminal_noise_time=args.terminal_noise_time, normal_steps=args.normal_steps,
                       terminal_refinement_steps=args.terminal_refinement_steps, prefix_mask_mode=args.prefix_mask_mode,
                       warmup_inference_delay=args.warmup_inference_delay,warmup_execute_horizon=args.warmup_execute_horizon)
    if args.training_contract:
        contract = json.loads(args.training_contract.read_text())
        required = {"state_dim": 7, "action_dim": 7, "action_horizon": 20, "period_s": .05,
                    "row_zero_alignment": "observation_time", "action_row0_offset_s": 0, "discrete_state_input": False,
                    "normalization": "quantile", "action_contract": policy.metadata["action_contract"]}
        if any(contract.get(k) != v for k, v in required.items()) or Path(contract.get("checkpoint_dir", "")).resolve() != Path(args.checkpoint).resolve():
            raise ValueError("Training provenance does not match this deployment")
        evidence = contract.get("evidence_files")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError("Training provenance needs actual evidence files")
        policy.metadata["training_evidence_sha256"] = {str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in evidence}
        policy.metadata["training_contract_sha256"] = hashlib.sha256(args.training_contract.read_bytes()).hexdigest()
        policy.metadata["training_contract_verified"] = True
    policy.warmup()
    if args.validate:
        from validation import validate
        with policy.rtc_sample.offline_experiment():
            report = validate(policy)
        Path(__file__).with_name("validation_report.json").write_text(json.dumps(report, indent=2)+"\n")
    manifest = dict(policy.metadata, devices=[str(x) for x in jax.devices()])
    manifest["openpi_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest["openpi_dirty"] = subprocess.check_output(["git", "status", "--short"], text=True)
    Path(__file__).with_name("deployment_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    logging.info("READY %s", json.dumps(manifest))
    asyncio.run(Server(policy, args.host, args.port).run())


if __name__ == "__main__":
    main()
