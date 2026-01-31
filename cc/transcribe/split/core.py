"""Audio splitting functionality using ffmpeg.

We are splitting long files into chunks.  We want to split _on silent sections
of the audio file_ so that all spoken words in the audio remain intact."""

import logging
import re
import subprocess
import typing as ty
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from thds.core.source import Source
from thds.mops import pure

from cc.transcribe.split.choose_silence_cuts import Cut, choose_cuts
from cc.transcribe.split.env import which_ffmpeg_or_raise
from cc.transcribe.workdir import workdir

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Chunk:
    index: int
    audio_src: Source
    start_time: float = 0.0
    end_time: float | None = None


def extract_audio(input_file: Source) -> Source:
    """Extract audio track from input file to m4a format."""
    which_ffmpeg_or_raise()

    output_audio_file = workdir() / "audio.m4a"
    workdir().mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            *"ffmpeg -hide_banner -loglevel error -i".split(),
            str(input_file.path()),  # paths can have spaces in them
            *"-vn -map 0:a:0 -c:a aac".split(),
            str(output_audio_file),
        ],
        check=True,
    )
    return Source.from_file(output_audio_file)


def _detect_silence(audio_file: Source) -> Source:
    """Run ffmpeg silencedetect and write log file."""
    which_ffmpeg_or_raise()

    log_file = workdir() / "silence.log"
    workdir().mkdir(parents=True, exist_ok=True)

    # ffmpeg writes silencedetect output to stderr
    result = subprocess.run(
        [
            *"ffmpeg -hide_banner -i".split(),
            str(audio_file.path()),  # paths can have spaces in them
            *"-vn -af silencedetect=noise=-35dB:d=0.4 -f null -".split(),
        ],
        capture_output=True,
        text=True,
    )

    log_file.write_text(result.stderr, encoding="utf-8")
    logger.info(f"Wrote: {log_file}")
    return Source.from_file(log_file)


def _extract_index_from_filename(filename: str) -> int:
    """Extract chunk index from filename like 'chunk_001.m4a' -> 1."""
    match = re.search(r"chunk_(\d+)", filename)
    return int(match.group(1)) if match else 0


@lru_cache()
def _get_audio_duration(audio_file: Source) -> float:
    """Get duration of audio file in seconds using ffprobe."""
    which_ffmpeg_or_raise()

    result = subprocess.run(
        [
            *"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1".split(),
            str(audio_file.path()),  # paths can have spaces in them
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def _fmt_float(x: float, digits: int) -> str:
    return f"{x:.{digits}f}".rstrip("0").rstrip(".")


def _get_max_volume(audio_file: Path) -> float | None:
    """Get max volume of audio file in dB using ffmpeg volumedetect."""
    which_ffmpeg_or_raise()

    result = subprocess.run(
        [
            *"ffmpeg -hide_banner -i".split(),
            str(audio_file),  # paths can have spaces in them
            *"-af volumedetect -f null -".split(),
        ],
        capture_output=True,
        text=True,
    )
    match = re.search(r"max_volume:\s*(-?\d+\.?\d*)\s*dB", result.stderr)
    return float(match.group(1)) if match else None


_DEFAULT_SILENCE_THRESHOLD: ty.Final = -35.0


def _is_silent(audio_file: Path, threshold_db: float = _DEFAULT_SILENCE_THRESHOLD) -> bool:
    """Check if audio file is silent (max volume below threshold)."""
    max_vol = _get_max_volume(audio_file)
    return max_vol is not None and max_vol < threshold_db


def _fmt_cuts_for_ffmpeg(cuts: ty.Iterable[Cut]) -> str:
    return ",".join(_fmt_float(cut.chosen, digits=6) for cut in cuts)


def _split_on_silence(audio_file: Source, cuts: list[Cut]) -> list[Chunk]:
    """Split audio file at the specified cut points."""
    which_ffmpeg_or_raise()

    chunks_dir = workdir() / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    if cuts:
        cuts_str = _fmt_cuts_for_ffmpeg(cuts)
        subprocess.run(
            [
                *"ffmpeg -hide_banner -loglevel error -i".split(),
                str(audio_file.path()),  # paths can have spaces in them
                *f"-f segment -segment_times {cuts_str} -reset_timestamps 1 -c copy".split(),
                f"{chunks_dir}/chunk_%03d.m4a",
            ],
            check=True,
        )
    else:
        # No cuts - just copy the whole file as chunk_000
        subprocess.run(
            [
                *"ffmpeg -hide_banner -loglevel error -i".split(),
                str(audio_file.path()),
                *"-c copy {chunks_dir}/chunk_000.m4a".split(),
            ],
            check=True,
        )

    # Compute chunk boundaries from cuts
    duration = _get_audio_duration(audio_file)
    cut_times = [c.chosen for c in cuts]
    boundaries = [0.0] + cut_times + [duration]

    logger.info(f"Chunks in: {chunks_dir}")
    chunk_files = sorted(chunks_dir.glob("*.m4a"), key=lambda f: f.name)
    chunks = [
        Chunk(
            index=_extract_index_from_filename(f.name),
            audio_src=Source.from_file(f),
            start_time=boundaries[i],
            end_time=boundaries[i + 1],
        )
        for i, f in enumerate(chunk_files)
    ]

    # Filter out silent chunks
    non_silent = [c for c in chunks if not _is_silent(c.audio_src.path())]
    if skipped := len(chunks) - len(non_silent):
        logger.info(f"Skipped {skipped} silent chunk(s)")

    assert non_silent, (
        f"Apparently the volume never exceeded {_DEFAULT_SILENCE_THRESHOLD}dB for this audio file"
    )
    return non_silent


def _is_audio_file_chunk_sized(
    audio_file_duration: float, split_every_s: float, window_s: float
) -> bool:
    max_chunk_size = split_every_s + window_s
    return audio_file_duration <= max_chunk_size


@pure.magic()
def split_audio_on_silences(
    input_file: Source, every: float = 1200.0, window: float = 90.0
) -> list[Chunk]:
    """Run the full split pipeline: extract audio, detect silence, choose cuts, split."""
    audio_file = extract_audio(input_file)

    audio_duration = _get_audio_duration(audio_file)
    if _is_audio_file_chunk_sized(audio_duration, every, window):
        # don't split
        if _is_silent(audio_file.path()):
            raise ValueError(
                f"Audio file has no speech in it (at least not >={_DEFAULT_SILENCE_THRESHOLD:.1f}dB); "
                " we will not transcribe"
            )

        return [
            Chunk(
                index=0,
                audio_src=Source.from_file(audio_file),
                start_time=0,
                end_time=audio_duration,
            )
        ]

    log_file = _detect_silence(audio_file)
    cuts = choose_cuts(silence_log_path=log_file, every=every, window=window)
    chunks = _split_on_silence(audio_file, cuts)
    return chunks
