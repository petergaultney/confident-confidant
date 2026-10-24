from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkTranscript:
    # No audio_src: under mops every chunk's Source resolves to the same '_bytes' basename, so
    # returning several of them raises DuplicateSourceBasenameError.
    index: int
    text: str
