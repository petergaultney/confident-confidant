"""E2E: transcribe-label (cc.transcribe.diarize.label)

Pure text processing — no API calls required.
"""

from pathlib import Path
import textwrap

import pytest

from cc.transcribe.diarize.label import apply_labels, extract_speakers

pytestmark = pytest.mark.e2e

SAMPLE_DIARIZED_TRANSCRIPT = """\
CHUNK_0_A: Hello, welcome to the meeting.

CHUNK_0_B: Thanks for having me.

CHUNK_0_A: Let's get started with the agenda.

CHUNK_0_C: Do you want to get something to drink?

CHUNK_0_B: No.

---

CHUNK_1_A: I wanted to follow up on last week.

CHUNK_1_B: Sure, go ahead.

CHUNK_1_A: The deployment went smoothly.
"""


def test_extract_speakers():
    speakers = extract_speakers(SAMPLE_DIARIZED_TRANSCRIPT)
    assert set(speakers) == {"CHUNK_0_A", "CHUNK_0_B", "CHUNK_0_C", "CHUNK_1_A", "CHUNK_1_B"}


def test_apply_labels(tmp_path: Path):
    """Apply a TOML label mapping and verify merge + replacement."""
    transcript_file = tmp_path / "transcript.txt"
    transcript_file.write_text(SAMPLE_DIARIZED_TRANSCRIPT, encoding="utf-8")

    labels_file = tmp_path / "speakers.toml"
    labels_file.write_text(
        textwrap.dedent(
            """
            Alice = ["CHUNK_0_A", "CHUNK_0_C", "CHUNK_1_A"]
            Bob = ["CHUNK_0_B", "CHUNK_1_B"]
            """
        ),
        encoding="utf-8",
    )

    apply_labels(transcript_file, labels_file)

    output = tmp_path / "transcript.labeled.txt"
    assert output.exists()
    text = output.read_text(encoding="utf-8")

    # Speaker names should be present, CHUNK labels should be gone
    assert "Alice:" in text
    assert "Bob:" in text
    assert "CHUNK_" not in text

    assert (
        text
        == """\
Alice: Hello, welcome to the meeting.

Bob: Thanks for having me.

Alice: Let's get started with the agenda. Do you want to get something to drink?

Bob: No.

---

Alice: I wanted to follow up on last week.

Bob: Sure, go ahead.

Alice: The deployment went smoothly."""
    )
