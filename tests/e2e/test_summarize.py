"""E2E: coco-summarize (cc.__main__.summarize_transcript)

Summarizes a transcript file using a cheap LLM.
"""

from pathlib import Path

import pytest

from cc.llm.summarize import SummaryNote, summarize_transcript

from tests.e2e.conftest import CHEAP_CONFIG

pytestmark = pytest.mark.e2e

SAMPLE_TRANSCRIPT = """\
Today I worked on refactoring the authentication module. The old code had a lot of
duplication between the session-based and token-based flows. I consolidated the shared
logic into a base class and kept the specifics in subclasses.

I also noticed that we weren't properly invalidating refresh tokens on logout. I filed
a ticket for that. I want to make sure we handle that before the next release.

Tomorrow I plan to write integration tests for the new auth flow and start looking at
the rate-limiting middleware that's been on the backlog.
"""


def test_summarize_transcript():
    """Summarize a transcript and verify title + note structure."""
    result = summarize_transcript(
        ll_model=CHEAP_CONFIG.note_model,
        transcript=SAMPLE_TRANSCRIPT,
        prompt=CHEAP_CONFIG.note_prompt,
    )

    assert isinstance(result, SummaryNote)
    assert len(result.title.split()) >= 2, f"Title too short: {result.title!r}"
    assert len(result.note.strip()) > 50, "Note body suspiciously short"


def test_summarize_transcript_file(tmp_path: Path):
    """Exercise the file-level summarize_transcript from __main__."""
    from cc.__main__ import summarize_transcript as summarize_transcript_file

    transcript_file = tmp_path / "transcript.txt"
    transcript_file.write_text(SAMPLE_TRANSCRIPT, encoding="utf-8")

    output_path = tmp_path / "summary.md"
    result = summarize_transcript_file(transcript_file, output_path)

    assert result == output_path
    assert output_path.exists()
    text = output_path.read_text(encoding="utf-8")
    assert text.startswith("# ")
    assert len(text) > 100
