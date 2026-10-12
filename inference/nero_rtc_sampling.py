"""NERO JAX pi0.5 inference-time RTC. No weight or training-code changes.

RTC VJP guidance adapted from the MIT-licensed Physical Intelligence reference:
https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/main/src/model.py
OpenPI uses noise time 1->0, hence the correction has a negative velocity sign.
The existing model's pi0.5 adaRMS, observation preprocessing and KV cache remain.
"""
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from openpi.models import model as model_api
from openpi.models.pi0 import make_attn_mask


class RTCSampler(nnx.Module):
    def __init__(self, model):
        self.model = model

    def sample(self, rng, observation, prior, weights, max_guidance, noise, num_steps=10):
        # Use the original sampler exactly when no commitment exists. Besides
        # preserving semantics, this avoids bf16 compiler reordering caused by
        # tracing a VJP even when its correction is multiplied by zero.
        return jax.lax.cond(jnp.any(weights > 0),
                            lambda _:self._guided(rng,observation,prior,weights,max_guidance,noise,num_steps),
                            lambda _:self.model.sample_actions(rng,observation,num_steps=num_steps,noise=noise),
                            operand=None)

    def _guided(self, rng, observation, prior, weights, max_guidance, noise, num_steps):
        m = self.model
        observation = model_api.preprocess_observation(None, observation, train=False)
        prefix, prefix_mask, prefix_ar = m.embed_prefix(observation)
        _, cache = m.PaliGemma.llm([prefix, None], mask=make_attn_mask(prefix_mask, prefix_ar),
                                  positions=jnp.cumsum(prefix_mask, axis=1)-1)
        batch = observation.state.shape[0]
        dt = -1./num_steps
        dim_mask = (jnp.arange(m.action_dim) < 7).astype(jnp.float32)[None, None, :]

        def step(carry):
            x, t = carry

            def denoiser(latent):
                suffix, mask, ar, cond = m.embed_suffix(observation, latent, jnp.broadcast_to(t, batch))
                prefix_attention = jnp.broadcast_to(prefix_mask[:, None, :],
                                                     (batch, suffix.shape[1], prefix_mask.shape[1]))
                attention = jnp.concatenate([prefix_attention, make_attn_mask(mask, ar)], axis=-1)
                positions = jnp.sum(prefix_mask, axis=-1)[:, None]+jnp.cumsum(mask, axis=-1)-1
                (_, out), _ = m.PaliGemma.llm([None, suffix], mask=attention, positions=positions,
                                             kv_cache=cache, adarms_cond=[None, cond])
                velocity = m.action_out_proj(out[:, -m.action_horizon:])
                return latent-t*velocity, velocity

            estimate, pullback, velocity = jax.vjp(denoiser, x, has_aux=True)
            error = (prior-estimate)*weights[None, :, None]*dim_mask
            correction = pullback(error)[0]*dim_mask
            tau = 1-t
            numerator = t*t+tau*tau
            # Algebraically the reference c*inv_r2; finite at t=1 and t=0.
            gain = jnp.minimum(numerator/jnp.maximum(t*tau, 1e-8), max_guidance)
            guided = velocity-gain*correction
            return x+dt*guided, t+dt

        result, _ = jax.lax.while_loop(lambda c: c[1] >= -dt/2, step, (noise, 1.))
        return result
