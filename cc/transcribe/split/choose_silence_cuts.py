import logging
import os
import re
import typing as ty
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Silence:
    start: float
    end: float

    @property
    def mid(self) -> float:
        return (self.start + self.end) / 2.0


class Cut(ty.NamedTuple):
    target: float
    chosen: float
    delta: float


class _NoSilenceException(ValueError):
    pass


class _NoCutsException(ValueError):
    pass


_SILENCE_START_RE = re.compile(r"silence_start:\s*([0-9]+(?:\.[0-9]+)?)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*([0-9]+(?:\.[0-9]+)?)")


def _parse_silences(lines: ty.Iterable[str]) -> list[Silence]:
    silences: list[Silence] = []
    current_start: float | None = None

    for line in lines:
        if m1 := _SILENCE_START_RE.search(line):
            current_start = float(m1.group(1))
            continue

        if m2 := _SILENCE_END_RE.search(line):
            end = float(m2.group(1))
            if current_start is None:
                # Occasionally logs can be truncated; ignore unmatched end.
                continue
            if end >= current_start:
                silences.append(Silence(start=current_start, end=end))
            current_start = None

    return silences


def _choose_cuts(
    mids: list[float],
    every: float,
    duration: float | None,
    window: float | None,
    start_at: float,
    stop_before_end: float,
) -> list[Cut]:
    """
    Returns list of Cut(target, chosen, delta) where delta = chosen - target.
    """
    if not mids:
        return []

    mids_sorted = sorted(mids)
    used = set()

    # If duration is not provided, infer something reasonable from the last midpoint.
    inferred_duration = mids_sorted[-1] + every
    dur = duration if duration is not None else inferred_duration

    # Targets: start_at + k*every, but don't schedule a cut too close to end
    last_target = dur - stop_before_end
    t = start_at

    results: list[Cut] = []
    while t <= last_target:
        # candidate mids
        if window is None:
            candidates = [(abs(m - t), m) for m in mids_sorted]
        else:
            candidates = [(abs(m - t), m) for m in mids_sorted if abs(m - t) <= window]

        candidates.sort(key=lambda x: x[0])

        chosen: float | None = None
        for _, m in candidates:
            if m not in used:
                chosen = m
                break

        if chosen is None:
            # No unused candidate within the window. If window was set, fall back to global nearest unused.
            if window is not None:
                candidates2 = [(abs(m - t), m) for m in mids_sorted if m not in used]
                candidates2.sort(key=lambda x: x[0])
                if candidates2:
                    chosen = candidates2[0][1]

        if chosen is None:
            break

        used.add(chosen)
        results.append(Cut(target=t, chosen=chosen, delta=chosen - t))
        t += every

    return results


def choose_cuts(
    silence_log_path: os.PathLike[str] | Path,
    *,
    every: float = 1200.0,
    duration: float | None = None,
    window: float | None = 90.0,
    start_at: float | None = None,
    stop_before_end: float = 30.0,
) -> list[Cut]:
    effective_window = window
    effective_start_at = start_at if start_at is not None else every

    # open() works with both Path and Source (Source implements __fspath__)
    with open(silence_log_path, "r", encoding="utf-8", errors="replace") as f:
        silences = _parse_silences(f)

    mids = [s.mid for s in silences]
    cuts = _choose_cuts(
        mids=mids,
        every=every,
        duration=duration,
        window=effective_window,
        start_at=effective_start_at,
        stop_before_end=stop_before_end,
    )

    if not silences:
        raise _NoSilenceException("No silence intervals found in log.")
    if not cuts:
        raise _NoCutsException(
            "No cut points could be selected (try increasing --window or check your log)."
        )

    logger.info(f"Found {len(silences)} silence intervals.")
    logger.info(f"Selected {len(cuts)} cut points near every {every:.0f}s.")
    if effective_window is not None:
        logger.info(f"Window: +/- {effective_window:.0f}s (fallback to global nearest if needed).")

    return cuts
