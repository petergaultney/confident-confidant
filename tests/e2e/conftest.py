"""Shared fixtures for e2e tests.

These tests hit real APIs (OpenAI, Anthropic) with cheap models.
Run with:  uv run pytest e2e-tests/ -v
"""

import typing as ty
import subprocess
import hashlib
from thds import humenc
from pathlib import Path

import pytest

from cc.config import ConfidentConfidantConfig
from tests.e2e import generate_audio

_E2E_DIR = Path(__file__).parent
_MONOLOGUE_TXT = _E2E_DIR / "fixtures" / "wendell-berry-why-i-am-not-going-to-buy-a-computer.md"
_DIALOGUE_TXT = _E2E_DIR / "fixtures" / "samuel-beckett-waiting-for-godot-act-1.md"

# Cheapest available models for each role
CHEAP_CONFIG = ConfidentConfidantConfig(
    transcription_model="whisper-1",
    reformat_model="gpt-4o-mini",
    note_model="gpt-4o-mini",
    diarization_model="gpt-4o-transcribe-diarize",
)


def _generate_or_get_cached_file(
    source_text: Path, generate_audio_file: ty.Callable[[Path, Path | None], Path]
) -> Path:
    with open(source_text, "rb") as f:
        wordybin = humenc.encode(hashlib.sha256(f.read()).digest())
    audio_file = _E2E_DIR / f"fixtures/generated/{source_text.stem}.{wordybin}.m4a"
    if audio_file.exists():
        return audio_file
    return generate_audio_file(source_text, audio_file)


@pytest.fixture(scope="session")
def long_audio() -> Path:
    return _generate_or_get_cached_file(_MONOLOGUE_TXT, generate_audio.monologue)


@pytest.fixture(scope="session")
def short_audio(long_audio) -> Path:
    """Extract first 60s of the long audio file via ffmpeg — cheap to transcribe."""
    fragment_file = long_audio.with_suffix(".fragment.m4a")
    if fragment_file.exists():
        return fragment_file

    subprocess.run(
        ["ffmpeg", "-y", "-i", str(long_audio), "-t", "60", "-c", "copy", str(fragment_file)],
        check=True,
        capture_output=True,
    )
    assert fragment_file.exists() and fragment_file.stat().st_size > 0
    return fragment_file


@pytest.fixture(scope="session")
def dialogue_audio() -> Path:
    return _generate_or_get_cached_file(_DIALOGUE_TXT, generate_audio.dialogue)
