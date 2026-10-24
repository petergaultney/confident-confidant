from dataclasses import dataclass

from thds.core.source import Source


@dataclass(frozen=True)
class ChunkTranscript:
    index: int
    text: str
    audio_src: Source
