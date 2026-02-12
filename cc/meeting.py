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
from cc.output_note import create_transcript_note
from cc.transcribe.diarize.core import transcribe_audio_diarized
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


def _phase2_label_and_summarize(
    transcript: Path,
    speakers_toml: Path,
    audio_path: Path,
    index: VaultIndex,
    vault_root: Path,
    config: ConfidentConfidantConfig,
    prompt: str,
    meeting_context: str,
    dry_run: bool,
) -> Path:
    labeled_transcript = apply_labels(transcript, speakers_toml)

    title, note = llm.summarize.summarize_transcript(
        config.note_model,
        transcript=labeled_transcript.read_text(encoding="utf-8"),
        prompt=_enrich_prompt(prompt, meeting_context),
        context=config.transcription_context,
    )

    original_audio_hash = hash_file(audio_path)

    filename_base = generate_new_filename(config.datetime_fmt, audio_path, title)
    new_audio_path = create_unique_file_path(
        audio_path,
        interpret_dir_config(vault_root, audio_path, config.audio_dir),
        filename_base,
    )
    copy_file(audio_path, new_audio_path, dry_run=dry_run)

    transcript_note_path = (
        interpret_dir_config(vault_root, audio_path, config.notes_dir) / f"{filename_base}.md"
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

    linking_notes = find_linking_notes(index, vault_root, audio_path)
    replace_links_in_notes(
        index, vault_root, linking_notes, audio_path, transcript_note_path, title, dry_run=dry_run
    )

    if not dry_run:
        logger.info(f"Removing original audio file: {audio_path}")
        audio_path.unlink()

    logger.info(
        f"Successfully processed {audio_path} -> {transcript_note_path.relative_to(vault_root)}"
    )
    return transcript_note_path


def process_meeting(audio_path: Path, dry_run: bool) -> Path | None:
    """Two-phase diarized meeting processing.

    Phase 1: Transcribe + diarize, write speakers.toml, print instructions, return None.
    Phase 2: If speakers.toml has mappings, label + summarize + create note, return note path.

    Mops caching means re-running transcription on the same file is free.
    """
    audio_path = audio_path.resolve()
    vault_root = find_vault_root(audio_path)
    index = build_vault_index(vault_root)
    config = read_config_from_directory_hierarchy(audio_path)

    # extract prompt tags from linking notes
    prompt_tags: list[str] = []
    seen: set[str] = set()
    for note_path in find_linking_notes(index, vault_root, audio_path):
        for tag in extract_prompt_tags(index, in_md_file=note_path, target_file=audio_path):
            if tag not in seen:
                prompt_tags.append(tag)
                seen.add(tag)
    prompt = resolve_prompt(collect_configs_root_to_file(audio_path), prompt_tags)

    meeting_context = _extract_meeting_context(index, vault_root, audio_path)
    if meeting_context:
        logger.info(f"Extracted meeting context: {meeting_context}")

    # phase 1 always runs (mops-cached, so free on re-run)
    transcript, speakers_toml = _phase1_transcribe(audio_path, config)

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
    return _phase2_label_and_summarize(
        transcript, speakers_toml, audio_path,
        index, vault_root, config, prompt, meeting_context, dry_run,
    )


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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

    args = parser.parse_args()
    process_meeting(args.audio_file.resolve(), dry_run=args.no_mutate)
