"""Fixed RTC timing, following upstream eval_flow.py (units: action steps)."""
from dataclasses import dataclass


def preserve_delay_prefix(previous, predicted, inference_delay):
    """Exact old-prefix/new-future splice; guidance is not a hard constraint."""
    if len(previous) != len(predicted) or not 0 <= inference_delay <= len(predicted):
        raise ValueError('RTC splice requires equal chunks and a valid inference_delay')
    return list(previous[:inference_delay])+list(predicted[inference_delay:])


@dataclass(frozen=True)
class RTCTiming:
    inference_delay: int = 6
    execute_horizon: int = 6

    def __post_init__(self):
        for name in ('inference_delay', 'execute_horizon'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f'RTC {name} must be an integer')
        if not 0 <= self.inference_delay <= self.execute_horizon < 20:
            raise ValueError('RTC requires 0 <= inference_delay <= execute_horizon < H20')
        if self.execute_horizon < 1 or self.inference_delay + self.execute_horizon > 20:
            raise ValueError('RTC requires inference_delay + execute_horizon <= H20')

    @property
    def prefix_attention_horizon(self):
        return 20 - self.execute_horizon

    @classmethod
    def from_execution(cls, execution):
        return cls(execution.get('inference_delay', 6), execution.get('execute_horizon', 6))
