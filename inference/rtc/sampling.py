"""Inference-only RTC for OpenPI's decreasing-time flow. No trained RTC weights.

Reference: Physical-Intelligence/real-time-chunking-kinetix src/model.py.
The correction sign is reversed because OpenPI integrates t=1 -> 0.
"""
import einops
import jax
import jax.numpy as jnp
from openpi.models import model as model_module
from openpi.models.pi0 import make_attn_mask


def prefix_weights(start, end, total=20, schedule="exp"):
    start = jnp.minimum(start, end)
    if schedule == 'ones':
        w = jnp.ones(total)
    elif schedule == 'zeros':
        w = (jnp.arange(total) < start).astype(jnp.float32)
    elif schedule in ('linear', 'exp'):
        w = jnp.clip((start - 1 - jnp.arange(total)) / (end - start + 1) + 1, 0, 1)
        if schedule == 'exp':
            w = w * jnp.expm1(w) / (jnp.e - 1)
    else:
        raise ValueError('Unsupported RTC prefix attention schedule')
    return jnp.where(jnp.arange(total) < end, w, 0)


def _guided_velocity_terms(velocity, x, t, target, weights, cap):
    def denoiser(z):
        v = velocity(z)
        return z - t * v, v
    clean, pullback, v = jax.vjp(denoiser, x, has_aux=True)
    residual = (target - clean) * weights
    correction = pullback(residual)[0]
    # Equivalent to the reference weight after tau=1-t, with finite endpoints.
    strength = jnp.minimum(cap, (t*t + (1-t)**2) / jnp.maximum(t*(1-t), 1e-8))
    return v - strength * correction, clean, correction, strength


def guided_velocity(velocity, x, t, target, weights, cap):
    return _guided_velocity_terms(velocity, x, t, target, weights, cap)[0]


def flow_time_grid(num_steps, terminal_noise_time, terminal_refinement_steps=1):
    """Optional endpoint-refined Euler grid; zero keeps upstream uniform time."""
    if num_steps < 1 or not 0 <= terminal_noise_time < 1:
        raise ValueError('Invalid RTC flow time grid')
    if terminal_noise_time:
        if num_steps < 2:
            raise ValueError('Endpoint refinement needs at least two flow steps')
        if not 1 <= terminal_refinement_steps < num_steps:
            raise ValueError('Invalid terminal refinement steps')
        if terminal_refinement_steps > 1:
            if terminal_noise_time >= .02:
                raise ValueError('Multiple terminal steps require terminal_noise_time < .02')
            # Resolve the stiff final guidance field without changing its VJP,
            # schedule, or cap. This is an opt-in numerical grid, not upstream.
            start = max(.02, terminal_noise_time)
            head = jnp.linspace(1., start, num_steps-terminal_refinement_steps+1)
            tail = jnp.geomspace(start, terminal_noise_time, terminal_refinement_steps)[1:]
            return jnp.concatenate([head, tail, jnp.zeros(1)])
        return jnp.concatenate([jnp.linspace(1.,terminal_noise_time,num_steps),jnp.zeros(1)])
    return jnp.linspace(1.,0.,num_steps+1)


def sample_rtc(self, rng, observation, prefix, weights, *, num_steps=10, noise=None, cap=5.0, compact_prefix=False,
               terminal_noise_time=0.0, terminal_refinement_steps=1):
    image_keys = tuple(k for k in observation.images if not compact_prefix or k != 'right_wrist_0_rgb')
    observation = model_module.preprocess_observation(None, observation, train=False, image_keys=image_keys)
    batch = observation.state.shape[0]
    if noise is None:
        noise = jax.random.normal(rng, (batch, self.action_horizon, self.action_dim))
    tokens, mask, ar_mask = self.embed_prefix(observation)
    _, cache = self.PaliGemma.llm([tokens, None], mask=make_attn_mask(mask, ar_mask),
                                positions=jnp.cumsum(mask, axis=1)-1)

    def velocity(z,t):
        suffix, suffix_mask, suffix_ar, cond = self.embed_suffix(observation, z, jnp.broadcast_to(t, batch))
        attention = jnp.concatenate([einops.repeat(mask, "b p -> b s p", s=suffix.shape[1]),
                                     make_attn_mask(suffix_mask, suffix_ar)], axis=-1)
        positions = jnp.sum(mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1)-1
        (_, out), _ = self.PaliGemma.llm([None, suffix], mask=attention, positions=positions,
                                       kv_cache=cache, adarms_cond=[None, cond])
        return self.action_out_proj(out[:, -self.action_horizon:])

    def step(carry):
        x, t = carry
        v = guided_velocity(lambda z: velocity(z,t), x, t, prefix, weights, cap)
        return x - v / num_steps, t - 1.0 / num_steps

    if terminal_noise_time:
        times=flow_time_grid(num_steps,terminal_noise_time,terminal_refinement_steps)
        def grid_step(x,pair):
            t,next_t=pair
            v=guided_velocity(lambda z: velocity(z,t),x,t,prefix,weights,cap)
            return x-(t-next_t)*v,None
        result,_=jax.lax.scan(grid_step,noise,(times[:-1],times[1:]))
        return result
    result, _ = jax.lax.while_loop(lambda carry: carry[1] >= 0.5 / num_steps, step, (noise, 1.0))
    return result


def sample_rtc_trace(self, rng, observation, prefix, weights, *, num_steps=10, noise=None, cap=5.0, compact_prefix=False,
                     terminal_noise_time=0.0, terminal_refinement_steps=1):
    """RTC sampler plus per-flow-step normalized residual diagnostics.

    This is intentionally separate from ``sample_rtc``: production inference
    remains on the compact while-loop path, while offline validation can locate
    whether convergence fails before or during the VJP correction.
    """
    image_keys = tuple(k for k in observation.images if not compact_prefix or k != 'right_wrist_0_rgb')
    observation = model_module.preprocess_observation(None, observation, train=False, image_keys=image_keys)
    batch = observation.state.shape[0]
    if noise is None:
        noise = jax.random.normal(rng, (batch, self.action_horizon, self.action_dim))
    tokens, mask, ar_mask = self.embed_prefix(observation)
    _, cache = self.PaliGemma.llm([tokens, None], mask=make_attn_mask(mask, ar_mask),
                                positions=jnp.cumsum(mask, axis=1)-1)
    hard_mask = weights >= (1.0 - 1e-6)
    soft_mask = (weights > 0) & ~hard_mask

    def rms(value, selected):
        squared = jnp.square(value) * selected
        return jnp.sqrt(jnp.sum(squared) / jnp.maximum(jnp.sum(selected), 1))

    def step(carry, _):
        x, t = carry
        if terminal_noise_time:
            t,next_t=_
        else:
            next_t=t-1./num_steps

        def velocity(z):
            suffix, suffix_mask, suffix_ar, cond = self.embed_suffix(observation, z, jnp.broadcast_to(t, batch))
            attention = jnp.concatenate([einops.repeat(mask, "b p -> b s p", s=suffix.shape[1]),
                                         make_attn_mask(suffix_mask, suffix_ar)], axis=-1)
            positions = jnp.sum(mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1)-1
            (_, out), _ = self.PaliGemma.llm([None, suffix], mask=attention, positions=positions,
                                           kv_cache=cache, adarms_cond=[None, cond])
            return self.action_out_proj(out[:, -self.action_horizon:])

        guided, clean, correction, strength = _guided_velocity_terms(velocity, x, t, prefix, weights, cap)
        residual = clean - prefix
        next_x = x - (t-next_t)*guided if terminal_noise_time else x-guided/num_steps
        trace = {"t": t, "hard_clean_rmse": rms(residual, hard_mask),
                 "soft_clean_rmse": rms(residual, soft_mask),
                 "correction_rms": jnp.sqrt(jnp.mean(jnp.square(correction))),
                 "guidance_strength": strength}
        return (next_x, next_t), trace

    times=flow_time_grid(num_steps,terminal_noise_time,terminal_refinement_steps)
    xs=(times[:-1],times[1:]) if terminal_noise_time else None
    result, trace = jax.lax.scan(step, (noise, jnp.asarray(1.0)), xs=xs, length=num_steps)
    return result, trace
