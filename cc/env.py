import os
from thds.core.lazy import lazy


def _anthropic_api_key() -> str:
    return open(os.path.expanduser("~/.keys/anthropic-api")).read().strip()


def _openai_api_key() -> str:
    return open(os.path.expanduser("~/.keys/openai-api")).read().strip()


def _set_api_key(env_var: str) -> None:
    os.environ[env_var] = os.environ.get(env_var) or _API_KEYS[env_var]()


_API_KEYS = {
    "OPENAI_API_KEY": _openai_api_key,
    "ANTHROPIC_API_KEY": _anthropic_api_key,
}


@lazy
def activate_api_keys() -> None:
    """So you don't have to have all of them in case you only use one."""
    list(map(_set_api_key, _API_KEYS.keys()))
