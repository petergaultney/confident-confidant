"""Two-phase vault-aware diarized meeting transcription."""

import logging
import tomllib
from pathlib import Path

from cc import llm
from cc.config import (
    ConfidentConfidantConfig,
    collect_configs_root_to_file,
    interpret_dir_config,
    read_config_from_directory_hierarchy,
    resolve_prompt,
)
from cc.files import copy_file, create_unique_file_path, generate_new_filename, hash_file
from cc.llm.summarize import SummaryNote
from cc.output_note import create_transcript_note
from cc.transcribe.diarize.core import transcribe_audio_diarized
from cc.transcribe.diarize.identify import (
    annotate_transcript,
    identify_speakers_interactive,
    prompt_for_names,
)
from cc.transcribe.diarize.label import apply_labels
from cc.vault import (
    VaultIndex,
    build_vault_index,
    extract_prompt_tags,
    find_linking_notes,
    find_section_context,
    find_vault_root,
    replace_links_in_notes,
)

logger = logging.getLogger(__name__)


def _speakers_toml_has_mappings(speakers_toml: Path) -> bool:
    if not speakers_toml.exists():
        return False

    try:
        with open(speakers_toml, "rb") as f:
            return len(tomllib.load(f)) > 0
    except tomllib.TOMLDecodeError:
        return False


def _extract_meeting_context(
    index: VaultIndex, vault_root: Path, audio_path: Path
) -> str:
    contexts: list[str] = []
    for note_path in find_linking_notes(index, vault_root, audio_path):
        try:
            contexts.extend(find_section_context(index, in_md_file=note_path, target_file=audio_path))
        except Exception as e:
            logger.warning(f"Could not read {note_path}: {e}")

    return "\n\n".join(contexts)


def _enrich_prompt(base_prompt: str, meeting_context: str) -> str:
    if not meeting_context:
        return base_prompt

    return (
        f"Context about this meeting: {meeting_context}\n\n"
        f"This is a multi-speaker meeting transcript with speaker labels. "
        f"Use the context above to inform your understanding of who is speaking "
        f"and what topics are being discussed.\n\n"
        f"{base_prompt}"
    )


def _phase1_transcribe(
    audio_path: Path, config: ConfidentConfidantConfig
) -> tuple[Path, Path]:
    """Returns (transcript_path, speakers_toml_path)."""
    output = transcribe_audio_diarized(
        audio_path,
        diarization_model=config.diarization_model,
        split_audio_approx_every_s=config.split_audio_approx_every_s,
    )
    return output.transcript, output.speakers_toml


def _finalize_note(
    summary: SummaryNote,
    audio_path: Path,
    reference_dir: Path,
    index: VaultIndex,
    vault_root: Path,
    config: ConfidentConfidantConfig,
    dry_run: bool,
) -> Path:
    """Move audio, create note, replace links — shared by both phase 2 flows.

    reference_dir is the base for resolving ./ config paths (audio_dir, notes_dir).
    Usually the linking note's directory — NOT the audio file's current directory,
    which may already be inside cc/audio/ from a previous run.
    """
    title, note = summary
    original_audio_hash = hash_file(audio_path)

    # use a synthetic path so interpret_dir_config resolves ./ relative to reference_dir
    ref_path = reference_dir / audio_path.name

    filename_base = generate_new_filename(config.datetime_fmt, audio_path, title)
    new_audio_path = create_unique_file_path(
        audio_path,
        interpret_dir_config(vault_root, ref_path, config.audio_dir),
        filename_base,
    )
    copy_file(audio_path, new_audio_path, dry_run=dry_run)

    transcript_note_path = (
        interpret_dir_config(vault_root, ref_path, config.notes_dir) / f"{filename_base}.md"
    )
    create_transcript_note(
        vault_root,
        audio_path if dry_run else new_audio_path,
        transcript_note_path,
        title,
        note,
        original_audio_hash,
    )

    if hash_file(audio_path) != original_audio_hash:
        raise ValueError(f"File hash changed during processing for {audio_path.name}, aborting")

    replace_links_in_notes(
        index, vault_root,
        find_linking_notes(index, vault_root, audio_path),
        audio_path, transcript_note_path, title, dry_run=dry_run,
    )

    if not dry_run:
        logger.info(f"Removing original audio file: {audio_path}")
        audio_path.unlink()

    logger.info(
        f"Successfully processed {audio_path} -> {transcript_note_path.relative_to(vault_root)}"
    )
    return transcript_note_path


def _phase2_label_and_summarize(
    transcript: Path,
    speakers_toml: Path,
    config: ConfidentConfidantConfig,
    prompt: str,
    meeting_context: str,
) -> SummaryNote:
    labeled_transcript = apply_labels(transcript, speakers_toml)
    return llm.summarize.summarize_transcript(
        config.note_model,
        transcript=labeled_transcript.read_text(encoding="utf-8"),
        prompt=_enrich_prompt(prompt, meeting_context),
        context=config.transcription_context,
    )


def _phase2_interactive(
    transcript: Path,
    config: ConfidentConfidantConfig,
    prompt: str,
    meeting_context: str,
    speaker_names: list[str] | None,
) -> SummaryNote:
    logger.info("Phase 2 interactive: transcript=%s", transcript)
    raw = transcript.read_text(encoding="utf-8")

    # get speaker names from CLI arg or interactive prompt
    names = speaker_names or prompt_for_names(
        meeting_context or config.transcription_context
    )
    if not names:
        raise SystemExit("No speaker names provided.")

    utt_ids, unknowns = identify_speakers_interactive(raw, names)
    if not utt_ids and not unknowns:
        raise SystemExit("No speaker labels to identify. Is the transcript already labeled?")

    annotated = annotate_transcript(raw, utt_ids, unknowns)
    logger.info("Annotated transcript preview (first 500 chars):\n%s", annotated[:500])

    print("Sending to LLM for fixup + summarization...")
    return llm.summarize.summarize_diarized_meeting(
        config.note_model,
        annotated_transcript=annotated,
        prompt=_enrich_prompt(prompt, meeting_context),
        context=config.transcription_context,
    )


def _resolve_prompt_tags(index: VaultIndex, vault_root: Path, audio_path: Path) -> list[str]:
    tags: list[str] = []
    seen: set[str] = set()
    for note_path in find_linking_notes(index, vault_root, audio_path):
        for tag in extract_prompt_tags(index, in_md_file=note_path, target_file=audio_path):
            if tag not in seen:
                tags.append(tag)
                seen.add(tag)
    return tags


def process_meeting(
    audio_path: Path,
    dry_run: bool,
    interactive: bool = False,
    speaker_names: list[str] | None = None,
) -> Path | None:
    """Two-phase diarized meeting processing.

    Phase 1: Transcribe + diarize (mops-cached, free on re-run).
    Phase 2 (TOML): If speakers.toml has mappings, label + summarize.
    Phase 2 (interactive): Walk through snippets, identify speakers, single LLM pass.
    """
    audio_path = audio_path.resolve()
    vault_root = find_vault_root(audio_path)
    index = build_vault_index(vault_root)
    config = read_config_from_directory_hierarchy(audio_path)

    prompt_tags = _resolve_prompt_tags(index, vault_root, audio_path)
    prompt = resolve_prompt(collect_configs_root_to_file(audio_path), prompt_tags)

    meeting_context = _extract_meeting_context(index, vault_root, audio_path)
    if meeting_context:
        logger.info(f"Extracted meeting context: {meeting_context}")

    # resolve ./ config paths relative to the linking note's directory, not the
    # audio file's current directory (which may already be inside cc/audio/)
    linking_notes = find_linking_notes(index, vault_root, audio_path)
    reference_dir = linking_notes[0].parent if linking_notes else audio_path.parent

    # phase 1 always runs (mops-cached, so free on re-run)
    transcript, speakers_toml = _phase1_transcribe(audio_path, config)

    if interactive:
        summary = _phase2_interactive(
            transcript, config, prompt, meeting_context, speaker_names,
        )
        return _finalize_note(summary, audio_path, reference_dir, index, vault_root, config, dry_run)

    if not _speakers_toml_has_mappings(speakers_toml):
        print(f"\n{'=' * 60}")
        print("PHASE 1 COMPLETE - Diarized transcription done.")
        print()
        print(f"  Transcript: {transcript}")
        print(f"  Speakers:   {speakers_toml}")
        print()
        print("Edit speakers.toml to map CHUNK_N_X labels to real names, e.g.:")
        print('  Peter = ["CHUNK_0_B", "CHUNK_1_A"]')
        print('  Grant = ["CHUNK_0_A", "CHUNK_1_B"]')
        print()
        print("Then re-run the same command to complete phase 2.")
        print(f"{'=' * 60}\n")
        return None

    logger.info("speakers.toml has mappings - proceeding to phase 2")
    summary = _phase2_label_and_summarize(
        transcript, speakers_toml, config, prompt, meeting_context,
    )
    return _finalize_note(summary, audio_path, reference_dir, index, vault_root, config, dry_run)


def _setup_logging() -> Path:
    """Configure logging to both stderr and a session log file in /tmp."""
    import time as _time

    log_dir = Path("/tmp/coco-meeting-logs")
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"{_time.strftime('%Y%m%d_%H%M%S')}.log"

    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_path),
    ])
    logging.getLogger(__name__).info(f"Session log: {log_path}")
    return log_path


def main() -> None:
    import argparse

    log_path = _setup_logging()

    parser = argparse.ArgumentParser(
        description="Two-phase vault-aware diarized meeting transcription.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "audio_file",
        type=Path,
        help="Path to the audio file to transcribe (must be within a vault).",
    )
    parser.add_argument(
        "--no-mutate",
        action="store_true",
        help="Don't move files or modify vault notes.",
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Use interactive speaker identification instead of speakers.toml.",
    )
    parser.add_argument(
        "--speakers",
        type=str,
        default=None,
        help="Comma-separated speaker names for interactive mode (e.g. 'Peter,Eby,Grant').",
    )

    args = parser.parse_args()
    speaker_names = [n.strip() for n in args.speakers.split(",") if n.strip()] if args.speakers else None
    note_path = process_meeting(
        args.audio_file.resolve(),
        dry_run=args.no_mutate,
        interactive=args.interactive,
        speaker_names=speaker_names,
    )
    if note_path:
        print(f"\n✅ {note_path}")
