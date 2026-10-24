import logging
import typing as ty
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI, omit
from thds.mops import pure

from cc.env import activate_api_keys
from cc.transcribe.split import Chunk
from cc.transcribe.workdir import workdir

from . import debug
from .types import ChunkTranscript

logger = logging.getLogger(__name__)


class _TranscriptionError(Exception):
    def __init__(self, file: Path, exception: Exception):
        self.file = file
        # Store as string to ensure picklability (raw exceptions may contain unpicklable state)
        self.exception = f"{type(exception).__name__}: {exception}"


def _transcribe_one(chunk: Chunk, model: str, prompt: str, out_dir: Path) -> ChunkTranscript:
    activate_api_keys()

    client = OpenAI()
    with chunk.audio_src.path().open("rb") as f:
        # The SDK infers the audio format from the filename, and under mops the path is just '_bytes'.
        resp = client.audio.transcriptions.create(
            model=model,
            prompt=prompt.strip() or omit,
            file=(f"{debug.chunk_name(chunk.index)}.m4a", f),
        )

    transcript = ChunkTranscript(index=chunk.index, text=resp.text)
    debug.write_chunk_transcript(out_dir, transcript)
    return transcript


@pure.magic()
def transcribe_chunks(chunks: ty.Sequence[Chunk], model: str, prompt: str) -> list[ChunkTranscript]:
    out_dir = workdir() / "chunk-transcripts"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Transcribing {len(chunks)} chunks, using model {model}, with prompt '{prompt}'...")
    successes: list[ChunkTranscript] = []
    failures: list[_TranscriptionError] = []
    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = {ex.submit(_transcribe_one, chunk, model, prompt, out_dir): chunk for chunk in chunks}
        for future in as_completed(futures):
            chunk = futures[future]
            try:
                successes.append(future.result())
                logger.info(f"ok  {debug.chunk_name(chunk.index)}")
            except Exception as exc:
                logger.error(f"Error raised for job {chunk.index}", exc_info=exc)
                failures.append(_TranscriptionError(chunk.audio_src.path(), exc))

    if failures:
        raise ExceptionGroup("Transcribing some chunks failed", failures)

    return successes
