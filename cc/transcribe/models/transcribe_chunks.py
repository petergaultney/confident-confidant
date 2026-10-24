import logging
import typing as ty

from cc.transcribe.split import Chunk

from . import openai, parakeet, types

logger = logging.getLogger(__name__)


def transcribe_chunks(
    chunks: ty.Sequence[Chunk], model: str, prompt: str
) -> list[types.ChunkTranscript]:
    if (hf_repo := parakeet.hf_repo_from_model_name(model)) is not None:
        if prompt.strip():
            logger.info("Parakeet takes no prompt; we do not use transcription_context to transcribe.")
        return parakeet.transcribe_chunks(chunks, hf_repo=hf_repo)
    return openai.transcribe_chunks(chunks, model, prompt)
