"""E2E: transcribe-diarize (cc.transcribe.diarize.core.transcribe_audio_diarized)

Tests the split → diarized-transcribe → format pipeline.
"""

import re
from pathlib import Path

import pytest

from cc.transcribe.diarize.core import transcribe_audio_diarized
from tests.e2e.conftest import CHEAP_CONFIG

pytestmark = pytest.mark.e2e


def test_transcribe_diarize(dialogue_audio: Path):
    """Diarize a short clip and verify speaker labels are present."""

    output = transcribe_audio_diarized(
        dialogue_audio,
        diarization_model=CHEAP_CONFIG.diarization_model,
        split_audio_approx_every_s=2 * 60,
    )

    # transcript file
    assert output.transcript.exists()
    transcript = output.transcript.read_text(encoding="utf-8")
    assert len(transcript.strip()) > 20

    # should contain CHUNK_N_X speaker labels
    assert re.search(
        r"CHUNK_\d+_[A-Z]+:", transcript
    ), "Expected CHUNK_N_X speaker labels in diarized transcript"

    # speakers.toml
    assert output.speakers_toml.exists()
    toml_text = output.speakers_toml.read_text(encoding="utf-8")
    assert "CHUNK_" in toml_text, "speakers.toml should list CHUNK_N_X labels"
