import json
from dataclasses import asdict
from pathlib import Path

from .types import ChunkTranscript


def chunk_name(index: int) -> str:
    """Matches the split step's chunk filenames; under mops the chunk's own path is just '_bytes'."""
    return f"chunk_{index:03d}"


def write_chunk_transcript(out_dir: Path, transcript: ChunkTranscript) -> None:
    """For troubleshooting: one JSON file per chunk, named like the chunk's audio file."""
    out_json = out_dir / f"{chunk_name(transcript.index)}.json"
    out_json.write_text(
        json.dumps(asdict(transcript), ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
