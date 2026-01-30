import json
import logging
import typing as ty
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from openai import OpenAI, omit
from thds.core.source import Source
from thds.mops import pure

from cc.env import activate_api_keys
from cc.transcribe.split import Chunk
from cc.transcribe.workdir import workdir

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkTranscript:
    index: int
    text: str
    audio_src: Source


class _TranscriptionError(Exception):
    def __init__(self, file: Path, exception: Exception):
        self.file = file
        # Store as string to ensure picklability (raw exceptions may contain unpicklable state)
        self.exception = f"{type(exception).__name__}: {exception}"


def _transcribe_one(chunk: Chunk, model: str, prompt: str, out_dir: Path) -> ChunkTranscript:
    activate_api_keys()

    client = OpenAI()
    with chunk.audio_src.path().open("rb") as f:
        resp = client.audio.transcriptions.create(model=model, prompt=prompt.strip() or omit, file=f)

    transcript = ChunkTranscript(index=chunk.index, text=resp.text, audio_src=chunk.audio_src)

    # for troubleshooting
    out_json = out_dir / f"{chunk.audio_src.path().stem}.json"
    out_json.write_text(
        json.dumps(asdict(transcript), ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

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
                logger.info(f"ok  {chunk.audio_src.path().name}")
            except Exception as exc:
                logger.error(f"Error raised for job {chunk.index}", exc_info=exc)
                failures.append(_TranscriptionError(chunk.audio_src.path(), exc))

    if failures:
        raise ExceptionGroup("Transcribing some chunks failed", failures)

    return successes
