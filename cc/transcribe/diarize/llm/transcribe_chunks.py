"""Transcription with GPT-4o diarization model."""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from openai import OpenAI
from thds.core.source import Source
from thds.mops import pure

from cc.env import activate_api_keys
from cc.transcribe.split import Chunk
from cc.transcribe.workdir import workdir

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DiarizedSegment:
    """A segment of speech attributed to a speaker within a chunk."""

    speaker: str  # e.g., "CHUNK_0_A", "CHUNK_1_B"
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class DiarizedChunkTranscript:
    """Transcription result for a single chunk with diarization."""

    index: int
    audio_src: Source  # Use Source instead of Path/str for memoization
    segments: list[DiarizedSegment]


def _rename_speaker(original: str, chunk_index: int) -> str:
    """Rename speaker from A/B/C to CHUNK_{index}_A/B/C."""
    # OpenAI returns speakers as "A", "B", "C", etc.
    return f"CHUNK_{chunk_index}_{original}"


def _transcribe_chunk_diarized(chunk: Chunk, model: str, out_dir: Path) -> DiarizedChunkTranscript:
    """Transcribe a single chunk with GPT-4o diarization."""
    activate_api_keys()

    client = OpenAI()

    with open(chunk.audio_src.path(), "rb") as f:
        # Using OpenAI client directly instead of litellm because of
        # https://github.com/BerriAI/litellm/issues/18125
        # litellm doesn't properly pass chunking_strategy which is required for diarization
        response = client.audio.transcriptions.create(
            model=model,
            file=f,
            response_format="diarized_json",
            chunking_strategy="auto",  # Required for audio > 30s
        )
        # gpt-4o-transcribe-diarize does not support prompts
        # https://developers.openai.com/api/docs/guides/speech-to-text/#prompting

    # Parse response - diarized_json returns segments with speaker labels
    segments: list[DiarizedSegment] = []

    # The diarized_json response has 'segments' with speaker info
    for seg in response.segments:
        segments.append(
            DiarizedSegment(
                speaker=_rename_speaker(seg.speaker, chunk.index),
                text=seg.text.strip(),
                start=seg.start,
                end=seg.end,
            )
        )

    transcript = DiarizedChunkTranscript(
        index=chunk.index,
        audio_src=chunk.audio_src,
        segments=segments,
    )

    # for troubleshooting
    out_json = out_dir / f"{chunk.audio_src.path().stem}.diarized.json"
    out_json.write_text(
        json.dumps(asdict(transcript), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    return transcript


class _TranscriptionError(Exception):
    def __init__(self, file: str, exception: Exception):
        self.file = file
        self.exception = exception


@pure.magic()
def transcribe_chunks_diarized(chunks: list[Chunk], model: str) -> list[DiarizedChunkTranscript]:
    """Transcribe chunks with GPT-4o diarization model."""
    out_dir = workdir() / "diarized-transcripts"
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = sorted(chunks, key=lambda c: c.index)

    logger.info(f"Transcribing {len(chunks)} chunks with diarization using {model}...")
    successes: list[DiarizedChunkTranscript] = []
    failures: list[_TranscriptionError] = []

    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = {
            ex.submit(_transcribe_chunk_diarized, chunk, model, out_dir): chunk for chunk in chunks
        }
        for future in as_completed(futures):
            chunk = futures[future]
            chunk_name = chunk.audio_src.path().name
            try:
                successes.append(future.result())
                logger.info(f"ok  {chunk_name}")
            except Exception as exc:
                failures.append(_TranscriptionError(chunk_name, exc))
                logger.error(f"Error caught for {chunk_name}: {exc}", exc_info=exc)

    if failures:
        raise ExceptionGroup("Transcribing some chunks failed", [f.exception for f in failures])

    return sorted(successes, key=lambda t: t.index)
