"""Add response diagnostics to the installed server without changing its sampler.

Run beside serve.py. An optional top-level rtc_diagnostic_noise_seed gives
repeatable offline requests; normal console requests keep the existing RNG.
"""
from __future__ import annotations

import functools
import hashlib
import logging
from pathlib import Path
import time

import numpy as np
import serve
from diagnostics import prefix_diagnostics, diagnostic_sampler_override


def install():
    original_infer = serve.RtcPolicy._infer
    original_warmup = serve.RtcPolicy.warmup

    @functools.wraps(original_infer)
    def infer(self, observation, *, noise=None, guidance=True):
        started = time.perf_counter()
        observation = dict(observation)
        seed = observation.pop('rtc_diagnostic_noise_seed', None)
        if seed is not None:
            if noise is not None or isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
                raise ValueError('Diagnostic seed must be uint32 and cannot be combined with noise')
            noise = np.random.default_rng(seed).standard_normal((20, 32)).astype(np.float32)
        reply = original_infer(self, observation, noise=noise, guidance=guidance)
        context = observation.get('rtc_context')
        if context is not None and len(context['absolute_prefix']):
            delay, length = serve.validate_context(context)
            schedule = 'ones' if context['phase'] == 'bootstrap_hold' else getattr(self, 'prefix_attention_schedule', 'exp')
            applied = guidance and context['phase'] != 'bootstrap'
            mask_start,mask_end,_ = serve.guidance_bounds(context,getattr(self,'prefix_mask_mode','timestamp_frozen'))
            weights = np.asarray(serve.prefix_weights(mask_start, mask_end, schedule=schedule)).copy() if applied else np.zeros(20)
            weights[length:] = 0
            terminal = float(getattr(self, 'terminal_noise_time', 0.))
            times = (np.r_[np.linspace(1., terminal, self.steps), 0.] if terminal
                     else np.linspace(1., 0., self.steps+1))
            refinements = int(getattr(self, 'terminal_refinement_steps', 1))
            if terminal and refinements > 1:
                start = max(.02, terminal)
                times = np.r_[np.linspace(1., start, self.steps-refinements+1),
                              np.geomspace(start, terminal, refinements)[1:], 0.]
            config = dict(algorithm='iigdm_vjp_decreasing_time_euler', num_steps=int(self.steps),
                          guidance_cap=float(self.cap), guidance_applied=applied,
                          prefix_attention_schedule=schedule, integration_grid=(('terminal_refined_multistep_v2' if refinements > 1 else 'terminal_refined_v1') if terminal else 'uniform'),
                          integration_times=times.tolist(), terminal_noise_time=terminal,
                          terminal_refinement_steps=refinements,
                          prefix_mask_mode=getattr(self,'prefix_mask_mode','timestamp_frozen'),
                          mask_start_row=int(mask_start),mask_end_row=int(mask_end),
                          compact_prefix=bool(getattr(self, 'compact_prefix', False)), valid_action_dim=7,
                          noise_seed=seed, noise_sha256=hashlib.sha256(np.asarray(noise, dtype='<f4').tobytes()).hexdigest() if noise is not None else None,
                          rtc_source_sha256=self.metadata.get('rtc_source_sha256'),
                          norm_stats_sha256=self.metadata.get('norm_stats_sha256'))
            reply['rtc'].update(prefix_diagnostics(reply['actions'], observation['observation/state'],
                               context['absolute_prefix'][:length], weights[:length], delay=mask_start, config=config))
        reply.setdefault('policy_timing', {})['diagnostic_total_ms'] = (time.perf_counter()-started)*1000
        return reply

    def warmup(self):
        print('RTC_DIAGNOSTICS warmup_started', flush=True)
        original_warmup(self)
        self.metadata['rtc_diagnostics_version'] = 'nero.rtc.diagnostics.v1'
        self.metadata['rtc_diagnostic_sampler_overrides'] = True
        refined = bool(getattr(self, 'terminal_noise_time', 0.))
        self.metadata['rtc_profile'] = (('terminal_refined_multistep_v2' if getattr(self, 'terminal_refinement_steps', 1) > 1
                                         else 'terminal_refined_v1') if refined else 'official_uniform')
        self.metadata['rtc_numerical_integrator_is_upstream_uniform'] = not refined
        self.metadata['rtc_diagnostics_sha256'] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                                 for name in ('diagnostic_server.py', 'diagnostics.py')}
        print('RTC_DIAGNOSTICS warmup_completed', flush=True)

    @functools.wraps(infer)
    def diagnostic_request(self, observation, *, noise=None, guidance=True):
        observation = dict(observation)
        overrides = observation.pop('rtc_diagnostic_sampler', None)
        if overrides is None:
            return infer(self, observation, noise=noise, guidance=guidance)
        if 'rtc_diagnostic_noise_seed' not in observation or 'rtc_context' not in observation:
            raise ValueError('Sampler overrides require an explicit offline RTC noise seed')
        with diagnostic_sampler_override(self, overrides), self.rtc_sample.offline_experiment():
            return infer(self, observation, noise=noise, guidance=guidance)

    serve.RtcPolicy._infer = diagnostic_request
    serve.RtcPolicy.warmup = warmup


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, force=True)
    install()
    serve.main()
