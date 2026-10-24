import json
from dataclasses import asdict
from pathlib import Path

from .types import ChunkTranscript


def write_chunk_transcript(out_dir: Path, transcript: ChunkTranscript) -> None:
    """For troubleshooting: one JSON file per chunk, named after the chunk's audio file."""
    out_json = out_dir / f"{transcript.audio_src.path().stem}.json"
    out_json.write_text(
        json.dumps(asdict(transcript), ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
