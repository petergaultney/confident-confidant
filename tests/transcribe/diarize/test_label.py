"""Tests for transcribe.diarize.label module."""

from cc.transcribe.diarize.label import (
    _merge_consecutive_speakers,
    _replace_labels,
    extract_speakers,
)


class TestExtractSpeakers:
    def test_extracts_distinct_speakers_sorted(self):
        transcript = """CHUNK_0_A: Hello there.

CHUNK_0_B: Hi!

CHUNK_0_A: How are you?

CHUNK_1_A: Fine, thanks.
"""
        speakers = extract_speakers(transcript)
        assert speakers == ["CHUNK_0_A", "CHUNK_0_B", "CHUNK_1_A"]

    def test_handles_multiple_letter_suffixes(self):
        transcript = """CHUNK_0_AA: Speaker with AA suffix.

CHUNK_0_B: Another speaker.
"""
        speakers = extract_speakers(transcript)
        assert "CHUNK_0_AA" in speakers
        assert "CHUNK_0_B" in speakers

    def test_empty_transcript(self):
        assert extract_speakers("") == []

    def test_no_speakers(self):
        transcript = "Just some text without speaker labels."
        assert extract_speakers(transcript) == []


class TestReplaceLabels:
    def test_replaces_speaker_labels(self):
        transcript = """CHUNK_0_A: Hello.

CHUNK_0_B: Hi!
"""
        label_onto_name = {"CHUNK_0_A": "Caleb", "CHUNK_0_B": "Andrew"}
        result = _replace_labels(transcript, label_onto_name)

        assert "Caleb: Hello." in result
        assert "Andrew: Hi!" in result
        assert "CHUNK_0_A" not in result
        assert "CHUNK_0_B" not in result

    def test_preserves_unmapped_labels(self):
        transcript = """CHUNK_0_A: Hello.

CHUNK_0_C: Unknown speaker.
"""
        label_onto_name = {"CHUNK_0_A": "Caleb"}
        result = _replace_labels(transcript, label_onto_name)

        assert "Caleb: Hello." in result
        assert "CHUNK_0_C: Unknown speaker." in result


class TestMergeConsecutiveSpeakers:
    def test_merges_consecutive_same_speaker(self):
        transcript = """Caleb: Hello there.

Caleb: How are you?

Andrew: I am fine.
"""
        result = _merge_consecutive_speakers(transcript)

        assert "Caleb: Hello there. How are you?" in result
        assert "Andrew: I am fine." in result
        # Should only have 2 speaker lines now
        assert result.count("Caleb:") == 1
        assert result.count("Andrew:") == 1

    def test_preserves_alternating_speakers(self):
        transcript = """Caleb: Hello.

Andrew: Hi!

Caleb: How are you?
"""
        result = _merge_consecutive_speakers(transcript)

        # All three should remain separate
        assert result.count("Caleb:") == 2
        assert result.count("Andrew:") == 1

    def test_handles_chunk_separators(self):
        transcript = """Caleb: End of chunk zero.

---

Caleb: Start of chunk one.
"""
        result = _merge_consecutive_speakers(transcript)

        # Should NOT merge across chunk boundary
        assert result.count("Caleb:") == 2
        assert "---" in result

    def test_merges_multiple_consecutive(self):
        transcript = """Caleb: One.

Caleb: Two.

Caleb: Three.

Andrew: Response.
"""
        result = _merge_consecutive_speakers(transcript)

        assert "Caleb: One. Two. Three." in result
        assert result.count("Caleb:") == 1

    def test_empty_transcript(self):
        assert _merge_consecutive_speakers("") == ""

    def test_preserves_paragraph_breaks_between_speakers(self):
        transcript = """Caleb: Hello.

Andrew: Hi!
"""
        result = _merge_consecutive_speakers(transcript)

        # Should have blank line between speakers
        lines = result.split("\n")
        caleb_idx = next(i for i, line in enumerate(lines) if line.startswith("Caleb:"))
        andrew_idx = next(i for i, line in enumerate(lines) if line.startswith("Andrew:"))
        # There should be a blank line between them
        assert andrew_idx - caleb_idx == 2
        assert lines[caleb_idx + 1] == ""
