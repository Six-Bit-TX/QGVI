"""Native-session transport only for the full QGVI-A controller."""
from speed_provider import ProviderError, SPEEDS, base

MODEL_OUTPUT_ERRORS = base.MODEL_OUTPUT_ERRORS
DEFAULT_TRANSPORTS = {'gpt': 'codex_native_session'}


def make_provider(*args, **kwargs):
    raise ProviderError('native_session_provider_must_be_bound_to_cohort_query')


def validate_effort(name, effort):
    if name != 'gpt' or effort != 'low':
        raise ValueError('This comparison freezes GPT-6.1 Sol / low')
    return effort


def require_speed_config(config):
    speed = config.get('speed')
    if speed not in SPEEDS:
        raise ValueError('Explicit speed arm required')
    tier, fast = SPEEDS[speed]
    if config.get('service_tier', {}).get('gpt') != tier or config.get('fast_mode') is not fast:
        raise ValueError('Inconsistent speed configuration')
    return tier
