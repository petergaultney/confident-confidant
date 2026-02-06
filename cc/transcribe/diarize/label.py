"""Replace CHUNK_N_X speaker labels with human-readable names."""

import argparse
import logging
import re
import tomllib
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_label_mappings(labels_path: Path) -> dict[str, str]:
    """Load label mappings from TOML file.

    TOML format:
        Caleb = ["CHUNK_0_A", "CHUNK_1_B"]
        Austin = ["CHUNK_0_B", "CHUNK_1_A"]

    Returns dict mapping CHUNK_N_X -> name.
    """
    with open(labels_path, "rb") as f:
        data = tomllib.load(f)

    mappings: dict[str, str] = {}
    for name, labels in data.items():
        if isinstance(labels, list):
            for label in labels:
                mappings[label] = name
        else:
            # Single label as string
            mappings[labels] = name

    return mappings


def extract_speakers(transcript: str) -> list[str]:
    """Extract distinct CHUNK_N_X speaker labels from transcript, in order of appearance."""
    pattern = re.compile(r"^(CHUNK_\d+_[A-Z]+):", re.MULTILINE)
    speakers: set[str] = set()
    for match in pattern.finditer(transcript):
        label = match.group(1)
        speakers.add(label)
    return sorted(speakers)


def _replace_labels(transcript: str, label_onto_name: dict[str, str]) -> str:
    """Replace CHUNK_N_X labels with mapped names."""

    # Pattern matches CHUNK_N_X at the start of lines (speaker labels)
    # e.g., "CHUNK_0_A:" or "CHUNK_1_B:"
    def replace_match(match: re.Match[str]) -> str:
        label = match.group(1)
        name = label_onto_name.get(label, label)  # Keep original if not mapped
        return f"{name}:"

    # Replace labels at start of lines
    result = re.sub(r"^(CHUNK_\d+_[A-Z]+):", replace_match, transcript, flags=re.MULTILINE)

    return result


def _merge_consecutive_speakers(transcript: str) -> str:
    """Merge consecutive blocks from the same speaker.

    Before:
        Caleb: Hello there.

        Caleb: How are you?

        Austin: Good!

    After:
        Caleb: Hello there. How are you?

        Austin: Good!
    """
    # Pattern to match speaker blocks: "Speaker: text" followed by blank lines
    # We'll parse line by line and merge as we go
    lines = transcript.split("\n")
    result_blocks: list[tuple[str, list[str]]] = []  # (speaker, [text lines])

    current_speaker: str | None = None
    current_texts: list[str] = []
    speaker_pattern = re.compile(r"^([^:]+):\s*(.*)$")

    for line in lines:
        stripped = line.strip()

        # Check if this line starts a speaker block
        match = speaker_pattern.match(stripped)
        if match and not stripped.startswith("---"):
            speaker = match.group(1)
            text = match.group(2)

            if speaker == current_speaker:
                # Same speaker - accumulate text
                current_texts.append(text)
            else:
                # Different speaker - flush previous and start new
                if current_speaker is not None:
                    result_blocks.append((current_speaker, current_texts))
                current_speaker = speaker
                current_texts = [text]
        elif stripped == "---":
            # Chunk separator - flush current and add separator
            if current_speaker is not None:
                result_blocks.append((current_speaker, current_texts))
                current_speaker = None
                current_texts = []
            result_blocks.append(("---", []))
        # Skip blank lines (they'll be re-added between blocks)

    # Flush final block
    if current_speaker is not None:
        result_blocks.append((current_speaker, current_texts))

    # Reconstruct transcript
    output_lines: list[str] = []
    for i, (speaker, texts) in enumerate(result_blocks):
        if speaker == "---":
            output_lines.append("---")
        else:
            merged_text = " ".join(texts)
            output_lines.append(f"{speaker}: {merged_text}")

        # Add blank line between blocks (except after last)
        if i < len(result_blocks) - 1:
            output_lines.append("")

    return "\n".join(output_lines)


def extract_speakers_(transcript_path: Path) -> None:
    """Extract and print distinct speaker labels from transcript."""
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript not found: {transcript_path}")

    transcript = transcript_path.read_text(encoding="utf-8")
    speakers = extract_speakers(transcript)

    logger.info(
        f"Found {len(speakers)} distinct speakers:\n  " + "\n  ".join(speaker for speaker in speakers)
    )


def apply_labels(transcript_path: Path, labels_path: Path) -> Path:
    """Apply label mappings and merge consecutive same-speaker blocks."""
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript not found: {transcript_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    # Load inputs
    transcript = transcript_path.read_text(encoding="utf-8")
    label_onto_name = _load_label_mappings(labels_path)

    logger.info(
        f"Loaded {len(label_onto_name)} label mappings:\n  "
        + "\n  ".join(f"{label} -> {name}" for label, name in sorted(label_onto_name.items()))
    )

    # Replace labels
    result = _replace_labels(transcript, label_onto_name)

    # Merge consecutive same-speaker blocks
    result = _merge_consecutive_speakers(result)

    # Write to new file
    output_path = transcript_path.with_suffix(".labeled.txt")
    output_path.write_text(result, encoding="utf-8")
    logger.info(f"Wrote: {output_path}")
    return output_path


def cli() -> None:
    parser = argparse.ArgumentParser(
        prog="transcribe-label",
        description="Replace CHUNK_N_X speaker labels with human-readable names.",
    )
    parser.add_argument("transcript", help="Path to transcript.txt", type=Path)
    parser.add_argument("labels", nargs="?", help="Path to labels.toml", type=Path)
    parser.add_argument(
        "--speakers",
        action="store_true",
        help="Extract and print distinct speaker labels (no labels file needed)",
    )
    args = parser.parse_args()

    if args.speakers:
        extract_speakers_(args.transcript)
    else:
        if not args.labels:
            parser.error("labels file is required unless --extract is specified")
        apply_labels(transcript_path=args.transcript, labels_path=args.labels)


if __name__ == "__main__":
    cli()
