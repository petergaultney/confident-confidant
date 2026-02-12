import logging
from pathlib import Path

from thds.core import source

from cc.config import DEFAULT_CONFIG
from cc.transcribe import llm
from cc.transcribe.split import split_audio_on_silences
from cc.transcribe.stitch import stitch_transcripts
from cc.transcribe.workdir import derive_workdir, workdir

logger = logging.getLogger(__name__)


def transcribe_audio_file(
    input_file: Path,
    *,
    transcription_model: str = DEFAULT_CONFIG.transcription_model,
    transcription_context: str = DEFAULT_CONFIG.transcription_context,
    reformat_model: str = DEFAULT_CONFIG.reformat_model,
    split_audio_approx_every_s: float = DEFAULT_CONFIG.split_audio_approx_every_s,
) -> Path:
    """Transcribe an audio file, returning the path to the final transcript.

    Args:
        input_file: Path to the audio file to transcribe

    Returns:
        Path to the generated transcript file
    """
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    workdir.set_global(derive_workdir(input_file, "transcribe"))
    workdir().mkdir(parents=True, exist_ok=True)

    # Run pipeline steps (decorator handles caching)
    chunks = split_audio_on_silences(source.from_file(input_file), every=split_audio_approx_every_s)
    chunk_transcripts = llm.transcribe_chunks(
        chunks, model=transcription_model, prompt=transcription_context
    )
    final = stitch_transcripts(chunk_transcripts, model=reformat_model)

    logger.info(f"Done. Output in: {workdir()}")
    logger.info(f"  transcript.txt: {final.path()}")

    return final.path()
