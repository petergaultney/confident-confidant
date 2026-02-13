"""Textual TUI for interactive speaker identification."""

import logging
from collections import Counter

from textual.app import App, ComposeResult
from textual.widgets import DataTable, Footer, Static

from cc.transcribe.diarize.identify import (
    _assign_keys,
    _group_by_chunk,
    _parse_utterances,
)

logger = logging.getLogger(__name__)

_SPEAKER_EMOJI = ["🔵", "🟢", "🟡", "🟠", "🔴", "🟣", "⚪", "⚫"]
_CHUNK_SEPARATOR = "__chunk_sep__"


def _emoji_for(letter_index: int) -> str:
    if letter_index < len(_SPEAKER_EMOJI):
        return _SPEAKER_EMOJI[letter_index]
    return "💜"  # overflow


def _word_count(text: str) -> int:
    return len(text.split())


def _truncate(text: str, max_len: int) -> str:
    return text[:max_len] + ".." if len(text) > max_len else text


def _stretch_text_column(table: DataTable, text_col_idx: int, total_width: int) -> None:
    """Give all remaining width to the text column."""
    if not table.columns or total_width <= 0:
        return

    columns = list(table.columns.values())
    padding_total = 2 * table.cell_padding * len(columns) + 1
    fixed_total = sum(c.width for i, c in enumerate(columns) if i != text_col_idx)
    remaining = total_width - fixed_total - padding_total
    if remaining > 10:
        columns[text_col_idx].auto_width = False
        columns[text_col_idx].width = remaining


class _Row:
    """One displayable row — either an utterance or a chunk separator."""

    __slots__ = ("utt_idx", "label", "text", "chunk_id", "letter", "is_short", "is_separator")

    def __init__(
        self,
        utt_idx: int = -1,
        label: str = "",
        text: str = "",
        chunk_id: str = "",
        letter: str = "",
        is_short: bool = False,
        is_separator: bool = False,
    ):
        self.utt_idx = utt_idx
        self.label = label
        self.text = text
        self.chunk_id = chunk_id
        self.letter = letter
        self.is_short = is_short
        self.is_separator = is_separator


def _build_rows(
    utterances: list[tuple[str, str]],
    chunk_groups: list[tuple[str, dict[str, list[int]]]],
    min_snippet_len: int,
) -> list[_Row]:
    """Build the full row list including chunk separators and all utterances."""
    # build a map from chunk_id to all utterance indices in that chunk
    chunk_utts: dict[str, list[int]] = {}
    for i, (label, _) in enumerate(utterances):
        chunk_utts.setdefault(label.split("_")[1], []).append(i)

    rows: list[_Row] = []
    for chunk_id, _ in chunk_groups:
        rows.append(_Row(chunk_id=chunk_id, is_separator=True))
        for i in chunk_utts.get(chunk_id, []):
            label, text = utterances[i]
            rows.append(_Row(
                utt_idx=i, label=label, text=text,
                chunk_id=chunk_id, letter=label.split("_")[2],
                is_short=len(text) < min_snippet_len,
            ))

    return rows


def _detect_conflicts(
    rows: list[_Row],
    utt_ids: dict[int, str],
) -> set[str]:
    """Return labels that have been assigned to more than one distinct name."""
    label_names: dict[str, set[str]] = {}
    for row in rows:
        if row.is_separator:
            continue
        name = utt_ids.get(row.utt_idx)
        if name:
            label_names.setdefault(row.label, set()).add(name)

    return {label for label, names in label_names.items() if len(names) > 1}


class IdentifyApp(App):
    """Speaker identification TUI."""

    BINDINGS = [("ctrl+q", "quit", "Quit")]

    CSS = """
    #legend { height: 1; padding: 0 1; }
    #table { height: 1fr; }
    #expanded { height: auto; max-height: 6; padding: 0 1; }
    #status { height: 1; padding: 0 1; }
    """

    def __init__(
        self,
        all_rows: list[_Row],
        names: list[str],
        key_map: dict[str, str],
        letter_order: dict[str, dict[str, int]],
    ):
        super().__init__()
        self._all_rows = all_rows
        self._names = names
        self._key_map = key_map
        self._letter_order = letter_order  # chunk_id -> {letter: index}
        self._show_short = False

        self.utt_ids: dict[int, str] = {}
        self.unknowns: set[str] = set()
        self.aborted = False
        self._travel_down = True

        # undo stack: (utt_idx, previous_value_or_None)
        self._undo_stack: list[tuple[int, str | None]] = []

    def _visible_rows(self) -> list[_Row]:
        if self._show_short:
            return self._all_rows
        return [r for r in self._all_rows if not r.is_short]

    def compose(self) -> ComposeResult:
        legend_parts = [f"{key} = {name}" for key, name in self._key_map.items()]
        yield Static(
            "  ".join(legend_parts)
            + "    Esc = uncertain  u = clear  Tab = short  Shift-↑↓ = jump  Enter = submit",
            id="legend",
        )
        yield DataTable(id="table", cursor_type="row")
        yield Static("", id="expanded")
        yield Static("", id="status")

    def on_mount(self) -> None:
        table = self.query_one("#table", DataTable)
        who_width = max(len(n) for n in self._names) + 1
        table.add_column("Lbl", width=5, key="lbl")
        table.add_column("Wc", width=4, key="wc")
        table.add_column("!", width=1, key="conflict")
        table.add_column("Who", width=who_width, key="who")
        table.add_column("Text", width=40, key="text")
        self._rebuild_table()
        self._advance_to_first_utterance()
        self.call_later(self._stretch)

    def _stretch(self) -> None:
        _stretch_text_column(self.query_one("#table", DataTable), 4, self.size.width)

    def on_resize(self) -> None:
        self._stretch()

    # --- table build / incremental update ---

    def _rebuild_table(self) -> None:
        """Full clear + rebuild. Only used on mount and toggle-short."""
        table = self.query_one("#table", DataTable)
        table.clear()

        conflicts = _detect_conflicts(self._all_rows, self.utt_ids)
        visible = self._visible_rows()

        self._row_map: list[_Row] = []
        for row in visible:
            if row.is_separator:
                table.add_row(f"── CHUNK {row.chunk_id} ──", "", "", "", "", key=f"sep_{row.chunk_id}")
                self._row_map.append(row)
                continue

            order = self._letter_order.get(row.chunk_id, {})
            emoji = _emoji_for(order.get(row.letter, 0))
            table.add_row(
                f"{emoji} {row.letter}",
                str(_word_count(row.text)),
                "×" if row.label in conflicts else "",
                self.utt_ids.get(row.utt_idx, ""),
                _truncate(row.text, 120),
                key=str(row.utt_idx),
            )
            self._row_map.append(row)

        self._update_status()
        self._update_expanded()

    def _refresh_cells(self, changed_label: str) -> None:
        """Update only the Who and conflict columns for rows sharing changed_label."""
        table = self.query_one("#table", DataTable)
        conflicts = _detect_conflicts(self._all_rows, self.utt_ids)
        for row in self._row_map:
            if row.is_separator or row.label != changed_label:
                continue
            key = str(row.utt_idx)
            table.update_cell(key, "who", self.utt_ids.get(row.utt_idx, ""))
            table.update_cell(key, "conflict", "×" if row.label in conflicts else "")
        self._update_status()

    # --- status / expanded ---

    def _update_status(self) -> None:
        total = sum(1 for r in self._all_rows if not r.is_separator and not r.is_short)
        labeled = sum(1 for r in self._all_rows if not r.is_separator and r.utt_idx in self.utt_ids)
        conflicts = _detect_conflicts(self._all_rows, self.utt_ids)
        parts = [
            f"{labeled}/{total} labeled",
            f"{len(self.unknowns)} uncertain",
            f"{len(conflicts)} conflicts",
        ]
        self.query_one("#status", Static).update("  ·  ".join(parts) + "  ·  Enter = submit")

    def _update_expanded(self) -> None:
        table = self.query_one("#table", DataTable)
        cursor = table.cursor_row
        if 0 <= cursor < len(self._row_map):
            row = self._row_map[cursor]
            if not row.is_separator:
                self.query_one("#expanded", Static).update(f"► {row.text}")
                return

        self.query_one("#expanded", Static).update("")

    # --- cursor helpers ---

    def _current_row(self) -> _Row | None:
        table = self.query_one("#table", DataTable)
        cursor = table.cursor_row
        if 0 <= cursor < len(self._row_map):
            row = self._row_map[cursor]
            if not row.is_separator:
                return row

        return None

    def _scroll_with_margin(self) -> None:
        """Adjust scroll only when cursor is near the edge of the viewport."""
        table = self.query_one("#table", DataTable)
        cursor = table.cursor_row
        if cursor < 0 or not self._row_map:
            return

        # DataTable rows are 1 terminal line tall; cell_padding is horizontal only
        content_h = table.scrollable_content_region.height
        if content_h <= 4:
            return

        scroll_top = table.scroll_y
        margin = max(2, content_h // 4)

        # cursor above viewport
        if cursor < scroll_top:
            table.scroll_to(y=float(cursor), animate=False)
            return

        # cursor in bottom 25% — scroll to show look-ahead rows below
        cursor_from_top = cursor - scroll_top
        if cursor_from_top > content_h - margin and cursor + 1 < len(self._row_map):
            look_ahead = min(cursor + margin, len(self._row_map) - 1)
            table.scroll_to(y=float(look_ahead - content_h + 1), animate=False)

    def _advance_to_first_utterance(self) -> None:
        table = self.query_one("#table", DataTable)
        for i, row in enumerate(self._row_map):
            if not row.is_separator:
                table.move_cursor(row=i)
                return

    def _is_unlabeled(self, row: _Row) -> bool:
        return not row.is_separator and row.utt_idx not in self.utt_ids and row.label not in self.unknowns

    def _next_non_sep(self, from_row: int, down: bool) -> int:
        """Index of next non-separator row in direction, or -1."""
        step = 1 if down else -1
        i = from_row + step
        while 0 <= i < len(self._row_map):
            if not self._row_map[i].is_separator:
                return i
            i += step

        return -1

    def _auto_advance(self, from_row: int) -> None:
        """Move cursor to adjacent unlabeled row in travel direction, if one exists."""
        adj = self._next_non_sep(from_row, self._travel_down)
        if adj >= 0 and self._is_unlabeled(self._row_map[adj]):
            self.query_one("#table", DataTable).move_cursor(row=adj)
            self._scroll_with_margin()

    def _jump_to_unlabeled(self, down: bool) -> None:
        """Jump to the next unlabeled row in the given direction (Shift-Up/Down)."""
        table = self.query_one("#table", DataTable)
        self._travel_down = down
        step = 1 if down else -1
        i = table.cursor_row + step
        while 0 <= i < len(self._row_map):
            if self._is_unlabeled(self._row_map[i]):
                table.move_cursor(row=i)
                self._scroll_with_margin()
                return
            i += step

    def _row_map_index(self, utt_idx: int) -> int:
        for i, row in enumerate(self._row_map):
            if not row.is_separator and row.utt_idx == utt_idx:
                return i

        return -1

    # --- actions ---

    def _assign(self, name: str) -> None:
        row = self._current_row()
        if not row:
            return

        self._undo_stack.append((row.utt_idx, self.utt_ids.get(row.utt_idx)))
        self.utt_ids[row.utt_idx] = name
        logger.info("utt[%d] %s -> %s", row.utt_idx, row.label, name)
        self._refresh_cells(row.label)
        self._auto_advance(self._row_map_index(row.utt_idx))

    def _mark_uncertain(self) -> None:
        row = self._current_row()
        if not row:
            return

        self._undo_stack.append((row.utt_idx, self.utt_ids.get(row.utt_idx)))
        self.utt_ids.pop(row.utt_idx, None)
        self.unknowns.add(row.label)
        logger.info("utt[%d] %s -> <uncertain>", row.utt_idx, row.label)
        acted_idx = self._row_map_index(row.utt_idx)
        self._refresh_cells(row.label)
        self._auto_advance(acted_idx)

    def _toggle_short(self) -> None:
        anchor = self._current_row()
        anchor_utt_idx = anchor.utt_idx if anchor else -1

        self._show_short = not self._show_short
        self._rebuild_table()

        table = self.query_one("#table", DataTable)
        if anchor_utt_idx >= 0:
            exact = self._row_map_index(anchor_utt_idx)
            if exact >= 0:
                table.move_cursor(row=exact)
                self._scroll_with_margin()
                return

        # row disappeared — find nearest unlabeled
        anchor_global = anchor.utt_idx if anchor else 0
        best_i, best_dist = -1, float("inf")
        for i, row in enumerate(self._row_map):
            if row.is_separator:
                continue
            dist = abs(row.utt_idx - anchor_global)
            if not self._is_unlabeled(row):
                dist += 10000
            if dist < best_dist:
                best_dist = dist
                best_i = i

        if best_i >= 0:
            table.move_cursor(row=best_i)
        self._scroll_with_margin()

    def _clear_current(self) -> None:
        row = self._current_row()
        if not row:
            return

        if row.utt_idx in self.utt_ids:
            del self.utt_ids[row.utt_idx]
            logger.info("utt[%d] %s -> cleared", row.utt_idx, row.label)
            self._refresh_cells(row.label)

    def _undo_last(self) -> None:
        if not self._undo_stack:
            return

        utt_idx, old_value = self._undo_stack.pop()

        # find the label before changing state
        label = ""
        for row in self._row_map:
            if not row.is_separator and row.utt_idx == utt_idx:
                label = row.label
                break

        if old_value is not None:
            self.utt_ids[utt_idx] = old_value
        else:
            self.utt_ids.pop(utt_idx, None)

        logger.info("backspace: undo utt[%d] (restored %s)", utt_idx, old_value)
        if label:
            self._refresh_cells(label)

        idx = self._row_map_index(utt_idx)
        if idx >= 0:
            self.query_one("#table", DataTable).move_cursor(row=idx)
            self._scroll_with_margin()

    def action_quit(self) -> None:
        self.aborted = True
        self.exit()

    def on_key(self, event) -> None:
        key = event.key

        if key in ("up", "down"):
            self._travel_down = key == "down"
            return

        if key == "shift+down":
            self._jump_to_unlabeled(down=True)
            event.prevent_default()
            return

        if key == "shift+up":
            self._jump_to_unlabeled(down=False)
            event.prevent_default()
            return

        if key == "enter":
            self.exit()
            return

        if key == "escape":
            self._mark_uncertain()
            event.prevent_default()
            return

        if key == "u":
            self._clear_current()
            event.prevent_default()
            return

        if key == "tab":
            self._toggle_short()
            event.prevent_default()
            return

        if key == "backspace":
            self._undo_last()
            event.prevent_default()
            return

        # speaker assignment keys
        if key in self._key_map:
            self._assign(self._key_map[key])
            event.prevent_default()
            return

    def on_data_table_row_highlighted(self, event) -> None:
        self._update_expanded()


def identify_speakers_tui(
    transcript: str,
    names: list[str],
) -> tuple[dict[int, str], set[str]]:
    """Run the TUI for speaker identification.

    Returns (utt_ids, unknowns) — same interface as identify_speakers_interactive.
    """
    utterances = _parse_utterances(transcript)
    chunk_groups = _group_by_chunk(utterances)
    if not any(labels for _, labels in chunk_groups):
        print("No CHUNK_N_X speaker labels found in transcript — nothing to identify.")
        return {}, set()

    key_map = _assign_keys(names)
    logger.info(
        "Starting TUI identification: %d utterances, %d chunks, names=%s",
        len(utterances), len(chunk_groups), names,
    )

    # build letter order per chunk (for emoji assignment)
    letter_order: dict[str, dict[str, int]] = {}
    for chunk_id, label_indices in chunk_groups:
        seen: list[str] = []
        for label in label_indices:
            letter = label.split("_")[2]
            if letter not in seen:
                seen.append(letter)
        letter_order[chunk_id] = {l: i for i, l in enumerate(seen)}

    from cc.transcribe.diarize.identify import _MIN_SNIPPET_LEN

    all_rows = _build_rows(utterances, chunk_groups, _MIN_SNIPPET_LEN)
    app = IdentifyApp(all_rows, names, key_map, letter_order)
    app.run()

    if app.aborted:
        raise KeyboardInterrupt

    return app.utt_ids, app.unknowns
