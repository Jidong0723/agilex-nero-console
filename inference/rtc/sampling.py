"""Inference-only RTC for OpenPI's decreasing-time flow. No trained RTC weights.

Reference: Physical-Intelligence/real-time-chunking-kinetix src/model.py.
The correction sign is reversed because OpenPI integrates t=1 -> 0.
"""
import einops
import jax
import jax.numpy as jnp
from openpi.models import model as model_module
from openpi.models.pi0 import make_attn_mask


def prefix_weights(start, end, total=20):
    start = jnp.minimum(start, end)
    w = jnp.clip((start - 1 - jnp.arange(total)) / (end - start + 1) + 1, 0, 1)
    w = w * jnp.expm1(w) / (jnp.e - 1)
    return jnp.where(jnp.arange(total) < end, w, 0)


def guided_velocity(velocity, x, t, target, weights, cap):
    def denoiser(z):
        v = velocity(z)
        return z - t * v, v
    clean, pullback, v = jax.vjp(denoiser, x, has_aux=True)
    residual = (target - clean) * weights
    correction = pullback(residual)[0]
    # Equivalent to the reference weight after tau=1-t, with finite endpoints.
    strength = jnp.minimum(cap, (t*t + (1-t)**2) / jnp.maximum(t*(1-t), 1e-8))
    return v - strength * correction


def sample_rtc(self, rng, observation, prefix, weights, *, num_steps=10, noise=None, cap=5.0):
    observation = model_module.preprocess_observation(None, observation, train=False)
    batch = observation.state.shape[0]
    if noise is None:
        noise = jax.random.normal(rng, (batch, self.action_horizon, self.action_dim))
    tokens, mask, ar_mask = self.embed_prefix(observation)
    _, cache = self.PaliGemma.llm([tokens, None], mask=make_attn_mask(mask, ar_mask),
                                positions=jnp.cumsum(mask, axis=1)-1)

    def step(carry):
        x, t = carry
        def velocity(z):
            suffix, suffix_mask, suffix_ar, cond = self.embed_suffix(observation, z, jnp.broadcast_to(t, batch))
            attention = jnp.concatenate([einops.repeat(mask, "b p -> b s p", s=suffix.shape[1]),
                                         make_attn_mask(suffix_mask, suffix_ar)], axis=-1)
            positions = jnp.sum(mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1)-1
            (_, out), _ = self.PaliGemma.llm([None, suffix], mask=attention, positions=positions,
                                           kv_cache=cache, adarms_cond=[None, cond])
            return self.action_out_proj(out[:, -self.action_horizon:])
        v = guided_velocity(velocity, x, t, prefix, weights, cap)
        return x - v / num_steps, t - 1.0 / num_steps

    result, _ = jax.lax.while_loop(lambda carry: carry[1] >= 0.5 / num_steps, step, (noise, 1.0))
    return result
