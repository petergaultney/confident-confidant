"""Interactive speaker identification for diarized transcripts."""

import logging
import re
import sys
import termios
import tty
from collections import Counter

logger = logging.getLogger(__name__)

_SPEAKER_RE = re.compile(r"^(CHUNK_\d+_[A-Z]+):\s*(.+)$", re.MULTILINE)

_MIN_SNIPPET_LEN = 40
_MAX_SNIPPET_DISPLAY = 200
_MAX_CONTEXT_DISPLAY = 120

_DIM = "\033[2m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _truncate(text: str, max_len: int = _MAX_SNIPPET_DISPLAY) -> str:
    return text[:max_len] + "..." if len(text) > max_len else text


def _parse_utterances(transcript: str) -> list[tuple[str, str]]:
    """All (label, text) pairs in transcript order."""
    return [(m.group(1), m.group(2).strip()) for m in _SPEAKER_RE.finditer(transcript)]


def _chunk_of(label: str) -> str:
    """'CHUNK_0_A' -> '0'"""
    return label.split("_")[1]


def _group_by_chunk(
    utterances: list[tuple[str, str]],
) -> list[tuple[str, dict[str, list[int]]]]:
    """Group indices of substantial utterances by chunk.

    Returns [(chunk_id, {label: [utterance_indices]})] in chunk order.
    """
    chunk_order: list[str] = []
    chunks: dict[str, dict[str, list[int]]] = {}
    for i, (label, text) in enumerate(utterances):
        chunk = _chunk_of(label)
        if chunk not in chunks:
            chunk_order.append(chunk)
            chunks[chunk] = {}
        if len(text) >= _MIN_SNIPPET_LEN:
            chunks[chunk].setdefault(label, []).append(i)
    return [(c, chunks[c]) for c in chunk_order]


def _pick_snippets(
    utterances: list[tuple[str, str]],
    label_indices: dict[str, list[int]],
    already_shown: dict[str, set[int]],
    n: int = 2,
    only_labels: list[str] | None = None,
) -> list[tuple[str, str, int]]:
    """Pick up to n not-yet-shown utterances per label, in transcript order."""
    candidates: list[tuple[str, str, int]] = []
    for label, indices in label_indices.items():
        if only_labels and label not in only_labels:
            continue
        shown = already_shown.get(label, set())
        added = 0
        for idx in indices:
            if idx not in shown:
                candidates.append((label, _truncate(utterances[idx][1]), idx))
                shown.add(idx)
                added += 1
                if added >= n:
                    break
        already_shown[label] = shown
    candidates.sort(key=lambda x: x[2])
    return candidates


def _find_context(
    utterances: list[tuple[str, str]], target_idx: int,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Find 1 substantial utterance before and after target_idx."""
    before: list[tuple[str, str]] = []
    for i in range(target_idx - 1, -1, -1):
        label, text = utterances[i]
        if len(text) >= _MIN_SNIPPET_LEN:
            before.append((label, _truncate(text, _MAX_CONTEXT_DISPLAY)))
            break

    after: list[tuple[str, str]] = []
    for i in range(target_idx + 1, len(utterances)):
        label, text = utterances[i]
        if len(text) >= _MIN_SNIPPET_LEN:
            after.append((label, _truncate(text, _MAX_CONTEXT_DISPLAY)))
            break

    return before, after


def _read_key() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def prompt_for_names(context_hint: str) -> list[str]:
    if context_hint:
        print(f"\nMeeting context:\n  {context_hint}\n")
    raw = input("Speaker names (comma-separated): ")
    return [n.strip() for n in raw.split(",") if n.strip()]


def _assign_keys(names: list[str]) -> dict[str, str]:
    """Assign a unique lowercase key to each name, preferring letters from the name itself."""
    used: set[str] = set()
    key_map: dict[str, str] = {}
    for name in names:
        assigned = False
        for ch in name:
            k = ch.lower()
            if k.isalpha() and k not in used and k != "q":
                key_map[k] = name
                used.add(k)
                assigned = True
                break

        if not assigned:
            for code in range(ord("a"), ord("z") + 1):
                k = chr(code)
                if k not in used and k != "q":
                    key_map[k] = name
                    used.add(k)
                    break

    return key_map


def _label_summary(
    utterances: list[tuple[str, str]],
    utt_ids: dict[int, str],
    unknowns: set[str],
) -> list[str]:
    """Per-label summary lines, grouping per-utterance IDs by label."""
    label_ids: dict[str, list[str]] = {}
    for idx in sorted(utt_ids):
        label_ids.setdefault(utterances[idx][0], []).append(utt_ids[idx])

    lines: list[str] = []
    for label, ids in label_ids.items():
        if len(set(ids)) == 1:
            lines.append(f"  {label} → {ids[0]}")
        else:
            counts = Counter(ids)
            detail = ", ".join(f"{n} ({c})" for n, c in counts.most_common())
            lines.append(
                f"  {label} → CONFLICTING: {detail}"
                " — label likely assigned to multiple speakers"
            )

    for label in sorted(unknowns):
        if label not in label_ids:
            lines.append(f"  {label} → unknown")

    return lines


def format_identifications(
    transcript: str,
    utt_ids: dict[int, str],
    unknowns: set[str] | None = None,
) -> str:
    """User-facing summary: identified labels + explicit unknowns."""
    return "\n".join(
        _label_summary(_parse_utterances(transcript), utt_ids, unknowns or set())
    )


def annotate_transcript(
    transcript: str,
    utt_ids: dict[int, str],
    unknowns: set[str] | None = None,
) -> str:
    """Annotate CHUNK_N_X labels with identification info, preserving originals.

    Identified: ``CHUNK_0_A - Peter:``
    Unknown:    ``CHUNK_0_A - <uncertain>:``
    Unlabeled:  ``CHUNK_0_A:`` (unchanged)
    """
    _unknowns = unknowns or set()
    parts: list[str] = []
    last_end = 0
    for i, m in enumerate(_SPEAKER_RE.finditer(transcript)):
        label = m.group(1)
        if i in utt_ids:
            parts.append(transcript[last_end:m.end(1)])
            parts.append(f" - {utt_ids[i]}")
            last_end = m.end(1)
        elif label in _unknowns:
            parts.append(transcript[last_end:m.end(1)])
            parts.append(" - <uncertain>")
            last_end = m.end(1)

    parts.append(transcript[last_end:])
    return "".join(parts)


def identify_speakers_interactive(
    transcript: str,
    names: list[str],
) -> tuple[dict[int, str], set[str]]:
    """Walk through every substantial utterance chunk by chunk.

    Returns (utt_ids, unknowns) where utt_ids maps utterance indices (into
    _parse_utterances order) to identified names, and unknowns contains labels
    the user explicitly marked as unknown (Esc).

    q = skip this label (LLM infers), Esc = don't know, Ctrl-C aborts.
    """
    utterances = _parse_utterances(transcript)
    chunk_groups = _group_by_chunk(utterances)
    logger.info(
        "Starting interactive identification: %d utterances, %d chunks, names=%s",
        len(utterances), len(chunk_groups), names,
    )
    if not any(labels for _, labels in chunk_groups):
        print("No CHUNK_N_X speaker labels found in transcript — nothing to identify.")
        print("(Was this transcript already labeled with real names?)")
        return {}, set()

    key_map = _assign_keys(names)
    valid_keys = set(key_map.keys())

    legend = "    ".join(f"{k} = {name}" for k, name in key_map.items())
    help_line = f"  {legend}    Esc = uncertain    q = done w/ label    Bksp = back"

    utt_ids: dict[int, str] = {}
    unknowns: set[str] = set()

    for chunk_id, label_indices in chunk_groups:
        print(f"\n{help_line}\n\n--- CHUNK {chunk_id} ---\n")

        # all substantial utterances in transcript order
        batch: list[tuple[str, str, int]] = [
            (label, _truncate(utterances[idx][1]), idx)
            for label, indices in label_indices.items()
            for idx in indices
        ]
        batch.sort(key=lambda x: x[2])
        if not batch:
            continue

        give_up: set[str] = set()
        # parallel tracking for backspace: batch position and action taken
        shown_at: list[int] = []
        actions: list[str | None] = []  # name string, None (esc), or "give_up"

        # per-label utterance counts for display (e.g. "[2/5]")
        label_totals = {label: len(indices) for label, indices in label_indices.items()}

        idx = 0
        while idx < len(batch):
            label, snippet, utt_idx = batch[idx]

            if label in give_up:
                idx += 1
                continue

            # compute on the fly so backspace never leaves stale state
            seq = sum(1 for j in range(idx) if batch[j][0] == label) + 1
            before, after = _find_context(utterances, utt_idx)
            for ctx_label, ctx_text in before:
                print(f'{_DIM}      {ctx_label}: "{ctx_text}"{_RESET}')
            print(f"{_BOLD}{label} [{seq}/{label_totals[label]}]:{_RESET}")
            print(f'  "{snippet}"')
            for ctx_label, ctx_text in after:
                print(f'{_DIM}      {ctx_label}: "{ctx_text}"{_RESET}')

            sys.stdout.write("  → ")
            sys.stdout.flush()
            key = _read_key()

            if key == "\x03":
                print("abort")
                raise KeyboardInterrupt

            if key in ("\x7f", "\x08"):
                if shown_at:
                    print("back\n")
                    prev_pos = shown_at.pop()
                    prev_action = actions.pop()
                    prev_label = batch[prev_pos][0]
                    prev_utt_idx = batch[prev_pos][2]
                    logger.info("backspace: undo utt[%d] %s (was %s)", prev_utt_idx, prev_label, prev_action)

                    if prev_action == "give_up":
                        give_up.discard(prev_label)
                    elif prev_action is not None:
                        del utt_ids[prev_utt_idx]

                    idx = prev_pos
                    continue

                print("(first)\n")
                continue

            shown_at.append(idx)

            if key == "\x1b":  # Esc — "I don't know this speaker"
                unknowns.add(label)
                actions.append(None)
                logger.info("utt[%d] %s -> <uncertain>", utt_idx, label)
                print("don't know")
            elif key == "q":  # done with this label, infer the rest
                give_up.add(label)
                actions.append("give_up")
                logger.info("utt[%d] %s -> done (skip remaining)", utt_idx, label)
                print(f"done with {label}")
            elif key in valid_keys:
                utt_ids[utt_idx] = key_map[key]
                actions.append(key_map[key])
                logger.info("utt[%d] %s -> %s", utt_idx, label, key_map[key])
                print(key_map[key])
            else:
                shown_at.pop()  # wasn't actually acted on
                print("?")
                continue

            print()
            idx += 1

    print(format_identifications(transcript, utt_ids, unknowns))
    print()
    return utt_ids, unknowns
