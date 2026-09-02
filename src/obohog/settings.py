"""Runtime settings loaded from environment / .env.

Anything that has to be secret or user-specific (API keys, host
overrides, ...) lives here rather than in ``obohog.toml``. The config
file is committable-adjacent (users write it once and check the
example in) while ``.env`` is strictly user-local.

Consumers should call :func:`get_settings` — the returned object is
cached so pydantic's file-load work happens once per process.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _repo_env_path() -> Path:
    """Look for ``.env`` at the project root (cwd) — same convention as
    ``obohog.toml``. No walking up parent directories: the CLI is
    invoked from the project root."""
    return Path.cwd() / ".env"


class ObohogSettings(BaseSettings):
    """Secrets and per-user knobs. Populated from environment variables
    and, if present, a ``.env`` file at the project root.

    Every field is optional so plain ``git-file`` / ``github-release``
    sources still work without any ``.env`` — only providers that
    genuinely need a secret (like ``bioportal``) check the relevant
    field and raise a clear error if it's ``None``.
    """

    model_config = SettingsConfigDict(
        env_file=_repo_env_path(),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    bioportal_api_key: str | None = Field(
        default=None,
        description="BioPortal REST API key (BIOPORTAL_API_KEY in .env).",
    )

    robot_jar: str | None = Field(
        default=None,
        description=(
            "Path to robot.jar, used for format = \"owl\" sources when no "
            "`robot` executable is on PATH (ROBOT_JAR in .env)."
        ),
    )


@lru_cache(maxsize=1)
def get_settings() -> ObohogSettings:
    """Return the process-wide settings singleton."""
    return ObohogSettings()
