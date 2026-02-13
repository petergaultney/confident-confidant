from unittest.mock import patch

import pytest

from cc.transcribe.diarize.identify import (
    _assign_keys,
    _group_by_chunk,
    _parse_utterances,
    _pick_snippets,
    annotate_transcript,
    format_identifications,
    identify_speakers_interactive,
)

_SAMPLE_TRANSCRIPT = """\
CHUNK_0_A: I think we should use DuckDB for the normalized layer and expose it via GizmoSQL.

CHUNK_0_B: Yeah, that sounds right to me, let me check the current setup first.

CHUNK_0_C: We could also integrate it with the MCP server architecture we discussed last week.

--- CHUNK_1 ---

CHUNK_1_A: The visits pipeline outputs are what we really need to get at before anything else.

CHUNK_1_B: Right.

CHUNK_1_C: I agree, let's focus on the pipeline outputs before we tackle normalization.
"""


def test_parse_utterances_extracts_all():
    utterances = _parse_utterances(_SAMPLE_TRANSCRIPT)
    assert len(utterances) == 6
    assert utterances[0][0] == "CHUNK_0_A"
    assert utterances[3][0] == "CHUNK_1_A"
    assert utterances[4][0] == "CHUNK_1_B"


def test_group_by_chunk_basic():
    groups = _group_by_chunk(_parse_utterances(_SAMPLE_TRANSCRIPT))
    assert len(groups) == 2
    assert groups[0][0] == "0"
    assert groups[1][0] == "1"
    # CHUNK_1_B ("Right.") too short — not in label_indices
    assert "CHUNK_1_B" not in groups[1][1]


def test_group_by_chunk_all_short():
    groups = _group_by_chunk(_parse_utterances("CHUNK_0_A: Yeah.\n\nCHUNK_0_B: Right.\n"))
    assert len(groups) == 1
    assert groups[0][1] == {}


def test_group_by_chunk_empty():
    assert _group_by_chunk(_parse_utterances("")) == []


def test_pick_snippets_respects_n():
    transcript = """\
CHUNK_0_A: First substantial utterance about DuckDB normalization layer.

CHUNK_0_A: Second substantial utterance about benchmarking Spark approach.

CHUNK_0_A: Third substantial utterance about MCP server architecture.
"""
    utterances = _parse_utterances(transcript)
    label_indices = _group_by_chunk(utterances)[0][1]
    shown: dict[str, set[int]] = {}
    assert len(_pick_snippets(utterances, label_indices, shown, n=2)) == 2

    # subsequent pick gets the third
    assert len(_pick_snippets(utterances, label_indices, shown, n=2)) == 1


def test_pick_snippets_transcript_order():
    transcript = """\
CHUNK_0_A: First substantial utterance about DuckDB normalization layer.

CHUNK_0_B: First utterance from speaker B about something important here.

CHUNK_0_A: Second substantial utterance about benchmarking the approach.
"""
    utterances = _parse_utterances(transcript)
    label_indices = _group_by_chunk(utterances)[0][1]
    shown: dict[str, set[int]] = {}
    snippets = _pick_snippets(utterances, label_indices, shown, n=2)
    # A gets 2, B gets 1 → sorted by transcript order: A[0], B[1], A[2]
    assert len(snippets) == 3
    assert snippets[0][0] == "CHUNK_0_A"
    assert snippets[1][0] == "CHUNK_0_B"
    assert snippets[2][0] == "CHUNK_0_A"


def test_pick_snippets_only_labels_filter():
    utterances = _parse_utterances(_SAMPLE_TRANSCRIPT)
    label_indices = _group_by_chunk(utterances)[0][1]
    shown: dict[str, set[int]] = {}
    snippets = _pick_snippets(utterances, label_indices, shown, only_labels=["CHUNK_0_A"])
    assert all(label == "CHUNK_0_A" for label, _, _ in snippets)


def test_pick_snippets_skips_short_utterances():
    utterances = _parse_utterances(_SAMPLE_TRANSCRIPT)
    for _, label_indices in _group_by_chunk(utterances):
        shown: dict[str, set[int]] = {}
        labels = [l for l, _, _ in _pick_snippets(utterances, label_indices, shown, n=100)]
        assert "CHUNK_1_B" not in labels


def test_pick_snippets_truncates_long_utterances():
    transcript = f"CHUNK_0_A: {'x' * 300}\n"
    utterances = _parse_utterances(transcript)
    label_indices = _group_by_chunk(utterances)[0][1]
    shown: dict[str, set[int]] = {}
    snippets = _pick_snippets(utterances, label_indices, shown)
    assert snippets[0][1].endswith("...")
    assert len(snippets[0][1]) < 210


def test_pick_snippets_empty_when_all_shown():
    transcript = "CHUNK_0_A: First substantial utterance about DuckDB normalization.\n"
    utterances = _parse_utterances(transcript)
    label_indices = _group_by_chunk(utterances)[0][1]
    shown: dict[str, set[int]] = {}
    _pick_snippets(utterances, label_indices, shown)
    assert _pick_snippets(utterances, label_indices, shown) == []


def test_assign_keys_uses_first_letter():
    assert _assign_keys(["Peter", "Eby", "O'Neill"]) == {
        "p": "Peter", "e": "Eby", "o": "O'Neill",
    }


def test_assign_keys_handles_collision():
    result = _assign_keys(["Peter", "Pat"])
    assert result["p"] == "Peter"
    assert result["a"] == "Pat"


def test_assign_keys_skips_q():
    result = _assign_keys(["Quincy"])
    assert "q" not in result
    assert result["u"] == "Quincy"


def test_format_identifications_consistent():
    transcript = "CHUNK_0_A: Some long utterance about DuckDB.\nCHUNK_0_B: Another thing here.\n"
    result = format_identifications(transcript, {0: "Peter", 1: "Eby"})
    assert "CHUNK_0_A → Peter" in result
    assert "CHUNK_0_B → Eby" in result


def test_format_identifications_conflicting():
    transcript = """\
CHUNK_0_A: First long utterance about DuckDB normalization layer design.

CHUNK_0_A: Second long utterance about benchmarking the Spark approach.
"""
    result = format_identifications(transcript, {0: "Peter", 1: "Eby"})
    assert "CONFLICTING" in result
    assert "Peter" in result
    assert "Eby" in result


def test_format_identifications_with_unknowns():
    transcript = "CHUNK_0_A: Long utterance here.\nCHUNK_0_C: Another long one.\n"
    result = format_identifications(transcript, {0: "Peter"}, unknowns={"CHUNK_0_C"})
    assert "CHUNK_0_A → Peter" in result
    assert "CHUNK_0_C → unknown" in result


def test_format_identifications_empty():
    assert format_identifications("no chunks here", {}) == ""


# --- annotate_transcript ---

def test_annotate_identified():
    transcript = """\
CHUNK_0_A: I think we should use DuckDB.

CHUNK_0_B: That sounds right to me.
"""
    result = annotate_transcript(transcript, {0: "Peter", 1: "Eby"})
    assert "CHUNK_0_A - Peter: I think we should use DuckDB." in result
    assert "CHUNK_0_B - Eby: That sounds right to me." in result


def test_annotate_marks_unknown():
    transcript = "CHUNK_0_A: I think we should use DuckDB for the normalized layer.\n"
    result = annotate_transcript(transcript, {}, unknowns={"CHUNK_0_A"})
    assert "CHUNK_0_A - <uncertain>: I think" in result


def test_annotate_leaves_unlabeled():
    transcript = "CHUNK_0_A: I think we should use DuckDB for the normalized layer.\n"
    result = annotate_transcript(transcript, {})
    assert "CHUNK_0_A: I think" in result
    assert "<uncertain>" not in result


def test_annotate_mixed():
    transcript = """\
CHUNK_0_A: I think we should use DuckDB for the normalized layer.

CHUNK_0_B: That sounds right to me, let me check the current setup.

CHUNK_0_C: We could also integrate it with the MCP server architecture.
"""
    result = annotate_transcript(
        transcript, {0: "Peter"}, unknowns={"CHUNK_0_B"},
    )
    assert "CHUNK_0_A - Peter: I think" in result
    assert "CHUNK_0_B - <uncertain>: That sounds" in result
    assert "CHUNK_0_C: We could" in result  # unlabeled, left as-is


def test_annotate_preserves_non_utterance_lines():
    transcript = """\
Some preamble text.

CHUNK_0_A: I think we should use DuckDB for the normalized layer.

--- CHUNK 1 ---

CHUNK_1_A: The visits pipeline outputs are what we really need.
"""
    result = annotate_transcript(transcript, {0: "Peter", 1: "Eby"})
    assert "Some preamble text." in result
    assert "--- CHUNK 1 ---" in result
    assert "CHUNK_0_A - Peter: I think" in result
    assert "CHUNK_1_A - Eby: The visits" in result


def test_annotate_conflicting_label_per_utterance():
    """When same label is identified as different people, each utterance gets its own name."""
    transcript = """\
CHUNK_0_A: I think we should use DuckDB for the normalized layer.

CHUNK_0_B: That sounds right to me, let me check the current setup.

CHUNK_0_A: Actually let me look at the IR agent logs from Rapid 7.
"""
    result = annotate_transcript(transcript, {0: "Peter", 2: "Eby"})
    assert "CHUNK_0_A - Peter: I think" in result
    assert "CHUNK_0_A - Eby: Actually let me" in result


# --- identify_speakers_interactive ---

def test_identify_speakers_interactive_basic(capsys):
    keys = iter(["p", "e"])
    names = ["Peter", "Eby"]

    transcript = """\
CHUNK_0_A: I think we should use DuckDB for the normalized layer and GizmoSQL.

CHUNK_0_B: That makes sense, let me check the current setup and get back to you.
"""
    with patch("cc.transcribe.diarize.identify._read_key", side_effect=keys):
        utt_ids, unknowns = identify_speakers_interactive(transcript, names)

    assert utt_ids == {0: "Peter", 1: "Eby"}
    assert unknowns == set()


def test_identify_speakers_interactive_two_per_speaker(capsys):
    """With 2 substantial utterances per label, both are shown."""
    keys = iter(["p", "e", "p", "e"])
    names = ["Peter", "Eby"]

    transcript = """\
CHUNK_0_A: I think we should use DuckDB for the normalized layer and GizmoSQL.

CHUNK_0_B: That makes sense, let me check the current setup and get back to you.

CHUNK_0_A: Also we need to benchmark the existing Spark-based approach for comparison.

CHUNK_0_B: Right, I'll set up the performance testing framework for that benchmarking.
"""
    with patch("cc.transcribe.diarize.identify._read_key", side_effect=keys):
        utt_ids, _ = identify_speakers_interactive(transcript, names)

    assert utt_ids[0] == "Peter"
    assert utt_ids[2] == "Peter"
    assert utt_ids[1] == "Eby"
    assert utt_ids[3] == "Eby"


def test_identify_speakers_interactive_shows_all_utterances(capsys):
    """All substantial utterances in the chunk are shown."""
    transcript = """\
CHUNK_0_A: First substantial utterance about DuckDB normalization layer design.

CHUNK_0_B: That makes sense, let me check the current setup and get back to you.

CHUNK_0_A: Second utterance about benchmarking the existing Spark-based approach.

CHUNK_0_A: Third utterance about MCP server architecture and integration points.

CHUNK_0_A: Fourth utterance about Unity Catalog and metadata management strategies.
"""
    keys = iter(["p", "e", "e", "p", "p"])
    names = ["Peter", "Eby"]

    with patch("cc.transcribe.diarize.identify._read_key", side_effect=keys):
        utt_ids, _ = identify_speakers_interactive(transcript, names)

    assert utt_ids[0] == "Peter"  # A first
    assert utt_ids[1] == "Eby"    # B
    assert utt_ids[2] == "Eby"    # A second
    assert utt_ids[3] == "Peter"  # A third
    assert utt_ids[4] == "Peter"  # A fourth


def test_identify_speakers_interactive_detects_conflict(capsys):
    keys = iter(["p", "e", "e"])
    names = ["Peter", "Eby"]

    transcript = """\
CHUNK_0_A: I think we should use DuckDB for the normalized layer and GizmoSQL.

CHUNK_0_B: That makes sense, let me check the current setup and get back to you.

CHUNK_0_A: Actually now I'm looking at the IR agent logs from Rapid 7 instead.
"""
    with patch("cc.transcribe.diarize.identify._read_key", side_effect=keys):
        utt_ids, _ = identify_speakers_interactive(transcript, names)

    assert utt_ids[0] == "Peter"
    assert utt_ids[1] == "Eby"
    assert utt_ids[2] == "Eby"  # conflict on CHUNK_0_A!


def test_identify_speakers_interactive_esc_marks_unknown(capsys):
    """Esc = 'I don't know this speaker' — marks label as unknown."""
    keys = iter(["p", "\x1b"])
    names = ["Peter", "Eby"]

    transcript = """\
CHUNK_0_A: I think we should use DuckDB for the normalized layer and GizmoSQL.

CHUNK_0_B: That makes sense, let me check the current setup and get back to you.
"""
    with patch("cc.transcribe.diarize.identify._read_key", side_effect=keys):
        utt_ids, unknowns = identify_speakers_interactive(transcript, names)

    assert utt_ids == {0: "Peter"}
    assert "CHUNK_0_B" in unknowns


def test_identify_speakers_interactive_q_skips_label(capsys):
    """q = 'done with this label' — auto-skips remaining, stays unlabeled (not unknown)."""
    transcript = """\
CHUNK_0_A: I think we should use DuckDB for the normalized layer and GizmoSQL.

CHUNK_0_B: That makes sense, let me check the current setup and get back to you.

CHUNK_0_C: We could also integrate it with the MCP server architecture we discussed.

CHUNK_0_B: Right, I'll set up the performance testing framework for that benchmarking.
"""
    # identify A, skip B label, identify C — second B utterance auto-skipped
    keys = iter(["p", "q", "e"])
    names = ["Peter", "Eby"]

    with patch("cc.transcribe.diarize.identify._read_key", side_effect=keys):
        utt_ids, unknowns = identify_speakers_interactive(transcript, names)

    assert utt_ids[0] == "Peter"
    assert 1 not in utt_ids  # B was skipped
    assert utt_ids[2] == "Eby"
    assert "CHUNK_0_B" not in unknowns  # q is skip, not unknown

    out = capsys.readouterr().out
    assert "done with" in out


def test_identify_speakers_interactive_backspace_corrects(capsys):
    keys = iter(["p", "\x7f", "e", "e"])
    names = ["Peter", "Eby"]

    transcript = """\
CHUNK_0_A: I think we should use DuckDB for the normalized layer and GizmoSQL.

CHUNK_0_B: That makes sense, let me check the current setup and get back to you.
"""
    with patch("cc.transcribe.diarize.identify._read_key", side_effect=keys):
        utt_ids, _ = identify_speakers_interactive(transcript, names)

    assert utt_ids[0] == "Eby"  # corrected from Peter
    assert utt_ids[1] == "Eby"


def test_identify_speakers_interactive_backspace_undoes_skip_label(capsys):
    """Backspace after q undoes the skip, re-showing that label's utterances."""
    transcript = """\
CHUNK_0_A: I think we should use DuckDB for the normalized layer and GizmoSQL.

CHUNK_0_B: That makes sense, let me check the current setup and get back to you.

CHUNK_0_C: We could also integrate it with the MCP server architecture we discussed.

CHUNK_0_B: Right, I'll set up the performance testing framework for that benchmarking.
"""
    # identify A, skip B, [C shown] backspace (undo skip), identify B, identify C, identify B
    keys = iter(["p", "q", "\x7f", "e", "e", "e"])
    names = ["Peter", "Eby"]

    with patch("cc.transcribe.diarize.identify._read_key", side_effect=keys):
        utt_ids, unknowns = identify_speakers_interactive(transcript, names)

    assert utt_ids[0] == "Peter"
    assert utt_ids[1] == "Eby"
    assert utt_ids[2] == "Eby"
    assert utt_ids[3] == "Eby"
    assert unknowns == set()


def test_identify_speakers_interactive_ctrl_c_aborts():
    keys = iter(["p", "\x03"])
    names = ["Peter", "Eby"]

    transcript = """\
CHUNK_0_A: I think we should use DuckDB for the normalized layer and GizmoSQL.

CHUNK_0_B: That makes sense, let me check the current setup and get back to you.
"""
    with patch("cc.transcribe.diarize.identify._read_key", side_effect=keys):
        with pytest.raises(KeyboardInterrupt):
            identify_speakers_interactive(transcript, names)


def test_identify_speakers_interactive_unknown_key_reprompts(capsys):
    keys = iter(["z", "p", "e"])
    names = ["Peter", "Eby"]

    transcript = """\
CHUNK_0_A: I think we should use DuckDB for the normalized layer and GizmoSQL.

CHUNK_0_B: That makes sense, let me check the current setup and get back to you.
"""
    with patch("cc.transcribe.diarize.identify._read_key", side_effect=keys):
        utt_ids, _ = identify_speakers_interactive(transcript, names)

    assert utt_ids[0] == "Peter"
    assert utt_ids[1] == "Eby"


def test_identify_speakers_interactive_esc_then_identify(capsys):
    """Esc marks unknown, but a later positive ID on the same label still records."""
    transcript = """\
CHUNK_0_A: I think we should use DuckDB for the normalized layer and GizmoSQL.

CHUNK_0_B: That makes sense, let me check the current setup and get back to you.

CHUNK_0_A: Also we need to benchmark the existing Spark-based approach for comparison.

CHUNK_0_B: Right, I'll set up the performance testing framework for that benchmarking.
"""
    # don't-know A[0], identify B[1]=Eby, identify A[2]=Peter, identify B[3]=Eby
    keys = iter(["\x1b", "e", "p", "e"])
    names = ["Peter", "Eby"]

    with patch("cc.transcribe.diarize.identify._read_key", side_effect=keys):
        utt_ids, unknowns = identify_speakers_interactive(transcript, names)

    assert 0 not in utt_ids  # Esc'd, no positive ID
    assert utt_ids[2] == "Peter"
    assert utt_ids[1] == "Eby"
    assert utt_ids[3] == "Eby"
    assert "CHUNK_0_A" in unknowns  # Esc was pressed


def test_identify_speakers_interactive_shows_context(capsys):
    keys = iter(["p", "e", "e"])
    names = ["Peter", "Eby"]

    transcript = """\
CHUNK_0_A: I think we should use DuckDB for the normalized layer and GizmoSQL.

CHUNK_0_B: That makes sense, let me check the current setup and get back to you.

CHUNK_0_C: We could also integrate it with the MCP server architecture we discussed.
"""
    with patch("cc.transcribe.diarize.identify._read_key", side_effect=keys):
        _, _ = identify_speakers_interactive(transcript, names)

    out = capsys.readouterr().out
    # context lines use the 6-space dim prefix
    assert '      CHUNK_0_A: "' in out
    assert '      CHUNK_0_C: "' in out
