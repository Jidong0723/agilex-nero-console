"""Bounded sampler signatures; no CUDA/JAX dependency in the guard itself."""
from contextlib import contextmanager
import hashlib
import time
import numpy as np


def audit_normalization(normalize, unnormalize):
    """Exercise the real checkpoint transforms, including out-of-quantile data."""
    physical = np.linspace(-2., 2., 35, dtype=np.float32).reshape(5, 7)
    values = {'state': physical.copy(), 'actions': physical.copy()}
    restored = unnormalize(normalize(values))
    errors = {key: float(np.max(np.abs(restored[key]-physical))) for key in values}
    if not all(np.isfinite(value) and value <= 5e-6 for value in errors.values()):
        raise ValueError(f'Checkpoint normalization roundtrip failed: {errors}')
    return dict(mode='quantile', valid_dimensions=7, roundtrip_max_abs=errors)


class PreparedSampler:
    """Reject unseen online pytrees/static options before an expensive compile.

    This observes JIT input signatures, NOT CUDA graph capture or XLA cache
    internals. Warmup must finish successfully before sealing the set.
    """
    def __init__(self, sample, flatten, synchronize):
        self.sample, self.flatten, self.synchronize = sample, flatten, synchronize
        self.prepared = set()
        self.sealed = False
        self.offline = False
        self.last = None

    def signature(self, args, kwargs):
        arrays = dict(kwargs)
        static = tuple((key, arrays.pop(key, default)) for key, default in (
            ('num_steps', 10), ('cap', 5.0), ('compact_prefix', False),
            ('terminal_noise_time', 0.0), ('terminal_refinement_steps', 1)))
        leaves, tree = self.flatten((args, arrays))
        types = tuple((tuple(value.shape), str(value.dtype), bool(getattr(value, 'weak_type', False)))
                      for value in leaves)
        return str(tree), types, static

    def __call__(self, *args, **kwargs):
        signature = self.signature(args, kwargs)
        prepared = signature in self.prepared
        if self.sealed and not prepared and not self.offline:
            raise RuntimeError('RTC sampler signature was not warmed; stop control and restart the model service to prepare it')
        started = time.perf_counter()
        result = self.sample(*args, **kwargs)
        self.synchronize(result)
        self.last = dict(signature_sha256=hashlib.sha256(repr(signature).encode()).hexdigest(),
                         prepared_before_call=prepared, offline_diagnostic=self.offline,
                         synchronized_sample_ms=(time.perf_counter()-started)*1000)
        # Explicit offline experiments never expand the online allowed set.
        if not self.sealed:
            self.prepared.add(signature)
        return result

    @contextmanager
    def offline_experiment(self):
        prior = self.offline
        self.offline = True
        try:
            yield
        finally:
            self.offline = prior
