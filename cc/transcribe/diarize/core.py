import logging
import typing as ty
from pathlib import Path

from thds.core import source

from cc.config import DEFAULT_CONFIG
from cc.transcribe.diarize.format import format_diarized_transcripts
from cc.transcribe.diarize.label import extract_speakers
from cc.transcribe.diarize.llm.transcribe_chunks import transcribe_chunks_diarized
from cc.transcribe.split import split_audio_on_silences
from cc.transcribe.workdir import derive_workdir, workdir

logger = logging.getLogger(__name__)


class Output(ty.NamedTuple):
    transcript: Path
    speakers_toml: Path


def transcribe_audio_diarized(
    input_file: Path,
    *,
    diarization_model: str = DEFAULT_CONFIG.diarization_model,
    split_audio_approx_every_s: float = DEFAULT_CONFIG.split_audio_approx_every_s,
) -> Output:
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    workdir.set_global(derive_workdir(input_file, kind="transcribe-gpt-diarize"))
    workdir().mkdir(parents=True, exist_ok=True)

    logger.info("Splitting audio on silences...")
    chunks = split_audio_on_silences(source.from_file(input_file), every=split_audio_approx_every_s)
    logger.info(f"Split into {len(chunks)} chunks")

    logger.info("Transcribing with diarization...")
    transcripts = transcribe_chunks_diarized(chunks, model=diarization_model)

    logger.info("Formatting transcript...")  # (merge same-speaker segments, add paragraph breaks)
    transcript = format_diarized_transcripts(transcripts)

    # Write speakers list for labeling (don't clobber existing edits)
    speakers_toml = workdir() / "speakers.toml"
    if not speakers_toml.exists():
        speakers = extract_speakers(transcript=transcript.path().read_text(encoding="utf-8"))
        speakers_toml.write_text("\n".join(f"# {s}" for s in speakers) + "\n", encoding="utf-8")
        logger.info(f"Wrote: {speakers_toml} ({len(speakers)} speakers)")

    logger.info(f"Done. Output: {transcript}")
    return Output(transcript.path(), speakers_toml)
