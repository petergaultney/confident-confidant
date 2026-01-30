#!/usr/bin/env -S uv run python
"""
Generate multi-speaker audio from a dialogue transcript.
Uses macOS `say` for TTS and `ffmpeg` for concatenation.

Usage:
    python generate_audio.py <transcript.md> [output.m4a] [--preview]
"""

import typing as ty
import re
import subprocess
import tempfile
from pathlib import Path

Mode = ty.Literal["monologue", "dialogue"]


class _Utterance(ty.NamedTuple):
    voice: str
    text: str
    label: str


def _generate_audio(
    utterances: list[_Utterance], output_file: Path, silence_sec: float, mode: Mode
) -> Path:
    """Say each utterance to AIFF, interleave with silence, concat with ffmpeg."""
    tmpdir = Path(tempfile.mkdtemp(prefix=f"{mode}-audio-"))
    print(f"tmp dir: {tmpdir}")

    silence_ms = int(silence_sec * 1000)
    silence_file = tmpdir / "silence.aiff"
    subprocess.run(["say", "-o", silence_file, f"[[slnc {silence_ms}]]"], check=True)

    seg_paths: list[Path] = []
    total = len(utterances)
    for i, utt in enumerate(utterances):
        seg_file = tmpdir / f"seg_{i:04d}.aiff"
        print(f"[{i + 1}/{total}] {utt.label}")
        subprocess.run(["say", "-v", utt.voice, "-o", seg_file, f"'{utt.text}'"], check=True)
        seg_paths.extend([seg_file, silence_file])

    file_list_path = tmpdir / "files.txt"
    file_list_path.write_text("".join(f"file '{p}'\n" for p in seg_paths))

    concat_aiff = tmpdir / "concat.aiff"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            file_list_path,
            "-c",
            "copy",
            concat_aiff,
        ],
        check=True,
    )

    if output_file.suffix == ".aiff":
        concat_aiff.rename(output_file)
    else:
        subprocess.run(["ffmpeg", "-y", "-i", concat_aiff, output_file], check=True)

    print(f"\nOutput: {output_file}")
    print(f"Temp files in: {tmpdir}")
    return output_file


def _clean_text(text: str) -> str:
    """Remove markdown formatting and stage directions for TTS."""
    # remove inline stage directions: _(...)_  and optional trailing dot/space
    text = re.sub(r"_\([^)]*\)_\.?\s*", " ", text)
    # remove headings
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    # remove blockquote markers
    text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)
    # remove numbered list prefixes (e.g. "1. ")
    text = re.sub(r"^\d+\.\s+", "", text, flags=re.MULTILINE)
    # remove italic/bold markers
    text = re.sub(r"[_*]+", "", text)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # strip leading dot left over from stage-direction removal
    text = re.sub(r"^\.\s*", "", text)
    return text


def _read_paragraphs(filepath: Path) -> list[str]:
    """Read a file and split into paragraphs (blank-line delimited), stripping markdown."""
    with open(filepath) as f:
        content = f.read()

    raw_paragraphs = re.split(r"\n\s*\n", content)
    paragraphs: list[str] = []
    for raw in raw_paragraphs:
        cleaned = _clean_text(raw.strip())
        if cleaned and cleaned != "---":
            paragraphs.append(cleaned)
    return paragraphs


#
# DIALOGUE
#


class Segment(ty.NamedTuple):
    speaker: str
    raw_text: str


NARRATOR = "NARRATOR"

# Voices to assign round-robin to speakers as they appear in the transcript.
# The first voice is reserved for the narrator.
VOICE_POOL: list[str] = [
    "Samantha",  # narrator (index 0)
    "Daniel",
    "Ralph",
    "Fred",
    "Albert",
    "Junior",
    "Kathy",
    "Reed",
]

_SPEAKER_PATTERN = re.compile(r"^([A-Z][A-Z ]+):\s+(.*)", re.DOTALL)


def _parse_script(filepath: Path) -> list[Segment]:
    """Parse the markdown transcript into (speaker, text) segments."""
    with open(filepath) as f:
        lines = f.readlines()

    segments: list[Segment] = []
    in_content = False
    current_speaker: str | None = None
    current_text = ""

    def flush() -> None:
        nonlocal current_speaker, current_text
        if current_speaker and current_text.strip():
            segments.append(Segment(current_speaker, current_text.strip()))
        current_speaker = None
        current_text = ""

    for line in lines:
        stripped = line.strip()

        # content boundaries: <!--start--> and <!--end-->
        if stripped == "<!--start-->":
            in_content = True
            continue
        if stripped == "<!--end-->":
            flush()
            break

        if not in_content:
            continue

        # skip --- section breaks within content
        if stripped == "---":
            continue

        # blank line ends current block
        if stripped == "":
            flush()
            continue

        # skip metadata lines
        if stripped.startswith(">"):
            continue

        # headings -> narrator
        if stripped.startswith("#"):
            flush()
            heading_text = stripped.lstrip("#").strip()
            if heading_text:
                segments.append(Segment(NARRATOR, heading_text))
            continue

        # check for speaker line
        m = _SPEAKER_PATTERN.match(stripped)
        if m:
            flush()
            current_speaker = m.group(1)
            current_text = m.group(2)
            continue

        # full italic line -> narrator (but not inline stage dirs like "_(giving up)_")
        if stripped.startswith("_") and not stripped.startswith("_("):
            flush()
            text = re.sub(r"^_|_$", "", stripped)
            current_speaker = NARRATOR
            current_text = text
            continue

        # continuation: starts with _( or lowercase -> append to current block
        if stripped.startswith("_(") or (stripped and stripped[0].islower()):
            if current_speaker:
                current_text += " " + stripped
            else:
                current_speaker = NARRATOR
                current_text = stripped
            continue

        # anything else (capital start, not a speaker) -> narrator / stage direction
        flush()
        current_speaker = NARRATOR
        current_text = stripped

    flush()
    return segments


def _assign_voices(segments: list[Segment]) -> dict[str, str]:
    """Assign voices round-robin to speakers in order of first appearance."""
    voice_map: dict[str, str] = {NARRATOR: VOICE_POOL[0]}
    for speaker, _ in segments:
        if speaker not in voice_map:
            voice_map[speaker] = VOICE_POOL[len(voice_map) % len(VOICE_POOL)]
    return voice_map


#
# API
#


def monologue(
    script: Path, output_file: Path | None = None, voice: str = "Daniel", silence_sec: float = 0.6
) -> Path:
    output_file = output_file or script.with_suffix(".m4a")
    paragraphs = _read_paragraphs(script)
    utterances = [
        _Utterance(
            voice=voice,
            text=para,
            label=para[:70] + ("..." if len(para) > 70 else ""),
        )
        for para in paragraphs
    ]
    return _generate_audio(utterances, output_file, silence_sec, mode="monologue")


def preview_monologue(script: Path) -> None:
    paragraphs = _read_paragraphs(script)
    for i, para in enumerate(paragraphs):
        print(f"  [{i + 1}] {para[:90]}{'...' if len(para) > 90 else ''}")
    print(f"\n  {len(paragraphs)} paragraphs total")


def dialogue(script: Path, output_file: Path | None = None, silence_sec: float = 0.4) -> Path:
    output_file = output_file or script.with_suffix(".m4a")
    segments = _parse_script(script)
    voice_map = _assign_voices(segments)
    utterances = [
        _Utterance(
            voice=voice_map[speaker],
            text=text,
            label=f"{speaker} ({voice_map[speaker]}): {text[:70]}{'...' if len(text) > 70 else ''}",
        )
        for speaker, raw_text in segments
        if (text := _clean_text(raw_text))
    ]
    return _generate_audio(utterances, output_file, silence_sec, mode="dialogue")


def preview_dialogue(script: Path) -> None:
    segments = _parse_script(script)
    voice_map = _assign_voices(segments)

    for speaker, raw in segments:
        text = _clean_text(raw)
        if text:
            voice = voice_map[speaker]
            print(f"  {speaker:>25s} ({voice:>10s}):  {text[:80]}")
    print(f"\n  {len(segments)} segments total")
    print(f"  Voice assignments: {voice_map}")


def cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate TTS audio from a markdown script.")
    parser.add_argument("script", type=Path, help="Path to the markdown script")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=None,
        help="Output audio file (default: <script>.m4a)",
    )
    parser.add_argument(
        "--mode",
        choices=ty.get_args(Mode),
        default="monologue",
        help="Generation mode (default: monologue)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Print parsed segments without generating audio",
    )
    parser.add_argument(
        "--silence",
        type=float,
        default=None,
        help="Seconds of silence between segments",
    )

    args = parser.parse_args()

    if args.mode == "monologue":
        if args.preview:
            preview_monologue(args.script)
        else:
            kwargs: dict[str, ty.Any] = {"silence": args.silence} if args.silence is not None else {}
            monologue(args.script, args.output, **kwargs)
    else:
        if args.preview:
            preview_dialogue(args.script)
        else:
            kwargs = {"silence": args.silence} if args.silence is not None else {}
            dialogue(args.script, args.output, **kwargs)


if __name__ == "__main__":
    cli()
