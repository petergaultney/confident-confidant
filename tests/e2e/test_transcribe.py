"""E2E: transcribe (cc.transcribe.core.transcribe_audio_file)

Tests the split → transcribe chunks → stitch pipeline on a long audio file.
"""

import pytest
from pathlib import Path

from cc.transcribe.core import transcribe_audio_file

from tests.e2e.conftest import CHEAP_CONFIG

pytestmark = pytest.mark.e2e


def test_transcribe_long_audio(long_audio: Path):
    """Transcribe a >25 MB audio file — exercises the splitting / stitching path."""
    transcript_path = transcribe_audio_file(
        long_audio,
        transcription_model=CHEAP_CONFIG.transcription_model,
        transcription_prompt=CHEAP_CONFIG.transcription_prompt,
        reformat_model=CHEAP_CONFIG.reformat_model,
        split_audio_approx_every_s=120,  # split every 2 min so we actually test multi-chunk
    )

    assert transcript_path.exists()
    text = transcript_path.read_text(encoding="utf-8")
    assert len(text.strip()) > 100, "Transcript suspiciously short"


def test_transcribe_short_audio(short_audio: Path):
    """Transcribe a short clip — exercises the single-chunk (no-split) path."""
    transcript_path = transcribe_audio_file(
        short_audio,
        transcription_model=CHEAP_CONFIG.transcription_model,
        transcription_prompt=CHEAP_CONFIG.transcription_prompt,
        reformat_model=CHEAP_CONFIG.reformat_model,
    )

    assert transcript_path.exists()
    text = transcript_path.read_text(encoding="utf-8")
    assert len(text.strip()) > 20, "Transcript suspiciously short"
