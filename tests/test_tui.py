import pytest
from textual.widgets import DataTable

from cc.transcribe.diarize.tui import (
    IdentifyApp,
    _build_rows,
    _detect_conflicts,
    identify_speakers_tui,
)
from cc.transcribe.diarize.identify import (
    _assign_keys,
    _group_by_chunk,
    _parse_utterances,
    _MIN_SNIPPET_LEN,
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


def _make_app(transcript: str = _SAMPLE_TRANSCRIPT, names: list[str] | None = None) -> IdentifyApp:
    names = names or ["Peter", "Eby"]
    utterances = _parse_utterances(transcript)
    chunk_groups = _group_by_chunk(utterances)
    key_map = _assign_keys(names)

    letter_order: dict[str, dict[str, int]] = {}
    for chunk_id, label_indices in chunk_groups:
        seen: list[str] = []
        for label in label_indices:
            letter = label.split("_")[2]
            if letter not in seen:
                seen.append(letter)
        letter_order[chunk_id] = {l: i for i, l in enumerate(seen)}

    all_rows = _build_rows(utterances, chunk_groups, _MIN_SNIPPET_LEN)
    return IdentifyApp(all_rows, names, key_map, letter_order)


def test_build_rows_includes_separators_and_utterances():
    utterances = _parse_utterances(_SAMPLE_TRANSCRIPT)
    chunk_groups = _group_by_chunk(utterances)
    rows = _build_rows(utterances, chunk_groups, _MIN_SNIPPET_LEN)

    separators = [r for r in rows if r.is_separator]
    assert len(separators) == 2  # chunk 0 and chunk 1
    assert separators[0].chunk_id == "0"
    assert separators[1].chunk_id == "1"

    non_sep = [r for r in rows if not r.is_separator]
    assert len(non_sep) == 6  # all 6 utterances (including short ones)


def test_build_rows_marks_short():
    utterances = _parse_utterances(_SAMPLE_TRANSCRIPT)
    chunk_groups = _group_by_chunk(utterances)
    rows = _build_rows(utterances, chunk_groups, _MIN_SNIPPET_LEN)

    short_rows = [r for r in rows if not r.is_separator and r.is_short]
    # "Right." is the only short utterance
    assert len(short_rows) == 1
    assert short_rows[0].text == "Right."


def test_detect_conflicts_none():
    utterances = _parse_utterances(_SAMPLE_TRANSCRIPT)
    chunk_groups = _group_by_chunk(utterances)
    rows = _build_rows(utterances, chunk_groups, _MIN_SNIPPET_LEN)
    # consistent assignment: all A = Peter, all B = Eby
    utt_ids = {r.utt_idx: "Peter" for r in rows if not r.is_separator and r.letter == "A"}
    utt_ids.update({r.utt_idx: "Eby" for r in rows if not r.is_separator and r.letter == "B"})
    assert _detect_conflicts(rows, utt_ids) == set()


def test_detect_conflicts_found():
    transcript = """\
CHUNK_0_A: I think we should use DuckDB for the normalized layer and expose it.

CHUNK_0_B: That sounds right to me, let me check the current setup first.

CHUNK_0_A: Actually let me look at the IR agent logs from Rapid 7 instead.
"""
    utterances = _parse_utterances(transcript)
    chunk_groups = _group_by_chunk(utterances)
    rows = _build_rows(utterances, chunk_groups, _MIN_SNIPPET_LEN)
    # assign same label to different people
    utt_ids = {0: "Peter", 2: "Eby"}  # both CHUNK_0_A, different names
    conflicts = _detect_conflicts(rows, utt_ids)
    assert "CHUNK_0_A" in conflicts


@pytest.mark.asyncio
async def test_tui_assign_speakers():
    app = _make_app()
    async with app.run_test() as pilot:
        await pilot.press("p")  # assign first row to Peter
        await pilot.press("e")  # assign next to Eby

    assert app.utt_ids.get(0) == "Peter"
    assert app.utt_ids.get(1) == "Eby"


@pytest.mark.asyncio
async def test_tui_esc_marks_uncertain():
    app = _make_app()
    async with app.run_test() as pilot:
        await pilot.press("escape")  # mark first row as uncertain

    assert 0 not in app.utt_ids
    # the label for utterance 0 should be in unknowns
    assert "CHUNK_0_A" in app.unknowns


@pytest.mark.asyncio
async def test_tui_undo_with_backspace():
    app = _make_app()
    async with app.run_test() as pilot:
        await pilot.press("p")          # assign Peter
        await pilot.press("backspace")  # undo
        await pilot.press("e")          # reassign Eby

    assert app.utt_ids.get(0) == "Eby"


@pytest.mark.asyncio
async def test_tui_clear_with_u():
    app = _make_app()
    async with app.run_test() as pilot:
        await pilot.press("p")  # assign Peter to row 0
        # navigate back to row 0 (backspace jumps there)
        await pilot.press("backspace")
        # now cursor is on row 0 with Peter still assigned (backspace undid advance, not assignment)
        # actually backspace undoes the assignment too - so reassign then clear
        await pilot.press("p")  # assign Peter
        await pilot.press("up")  # go back
        await pilot.press("u")  # clear

    assert 0 not in app.utt_ids


@pytest.mark.asyncio
async def test_tui_submit_with_enter():
    app = _make_app()
    async with app.run_test() as pilot:
        await pilot.press("p")
        await pilot.press("enter")

    assert app.utt_ids.get(0) == "Peter"
    assert not app.aborted


@pytest.mark.asyncio
async def test_tui_toggle_short_utterances():
    app = _make_app()
    async with app.run_test() as pilot:
        # initially short utterances are hidden
        initial_visible = len(app._visible_rows())
        await pilot.press("tab")  # toggle
        after_toggle = len(app._visible_rows())
        # "Right." is short — should now be visible
        assert after_toggle >= initial_visible

        await pilot.press("tab")  # toggle back
        assert len(app._visible_rows()) == initial_visible


@pytest.mark.asyncio
async def test_tui_up_arrow_reverses_travel():
    """After pressing up, labeling should auto-advance upward."""
    transcript = """\
CHUNK_0_A: I think we should use DuckDB for the normalized layer and expose it.

CHUNK_0_B: That sounds right to me, let me check the current setup first.

CHUNK_0_C: We could also integrate it with the MCP server architecture here.
"""
    app = _make_app(transcript)
    async with app.run_test() as pilot:
        await pilot.press("p")   # A[0] = Peter, auto-advance down to B[1]
        await pilot.press("e")   # B[1] = Eby, auto-advance down to C[2]
        await pilot.press("up")  # travel direction = up, move to B[1]
        await pilot.press("up")  # move to A[0]
        # clear A[0] so it's unlabeled again
        await pilot.press("u")
        # now label B[1] — should NOT jump down, A[0] above is unlabeled
        await pilot.press("up")  # back to B
        # re-label B with a different name to trigger auto-advance
        await pilot.press("u")   # clear B
        await pilot.press("e")   # label B = Eby, travel is up, adjacent up (A) is unlabeled -> advance up

    assert app.utt_ids.get(1) == "Eby"
    # auto-advance should have gone up to A[0]
    assert 0 not in app.utt_ids  # A was cleared and cursor moved there


@pytest.mark.asyncio
async def test_tui_shift_down_jumps_to_unlabeled():
    transcript = """\
CHUNK_0_A: I think we should use DuckDB for the normalized layer and expose it.

CHUNK_0_B: That sounds right to me, let me check the current setup first.

CHUNK_0_C: We could also integrate it with the MCP server architecture here.
"""
    app = _make_app(transcript)
    async with app.run_test() as pilot:
        await pilot.press("p")         # A = Peter, advances to B
        await pilot.press("e")         # B = Eby, advances to C
        await pilot.press("up")        # back to B
        await pilot.press("up")        # back to A
        await pilot.press("u")         # clear A
        # now A is unlabeled, cursor on A. Shift-Down should jump past labeled B to unlabeled C
        await pilot.press("shift+down")

        table = app.query_one("#table", DataTable)
        cursor_row = app._row_map[table.cursor_row]
        assert cursor_row.utt_idx == 2  # landed on C
