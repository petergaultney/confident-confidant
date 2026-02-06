import itertools
import logging
import time
from functools import partial
from pathlib import Path

from cc import llm
from cc.config import interpret_dir_config, read_config_from_directory_hierarchy
from cc.files import copy_file, create_unique_file_path, generate_new_filename, hash_file
from cc.output_note import create_transcript_note
from cc.vault import (
    VaultIndex,
    link_line_has_tag,
    build_vault_index,
    find_linking_notes,
    find_vault_root,
    replace_links_in_notes,
)
from cc.transcribe import transcribe_audio_file

logger = logging.getLogger(__name__)


def summarize_transcript(transcript_path: Path, output_path: Path | None = None) -> Path:
    """Summarize an existing transcript file and write the result to a markdown file.

    Args:
        transcript_path: Path to the transcript .txt file to summarize
        output_path: Optional output file path. If not provided, generates a filename
                    using the same pattern as audio processing: {timestamp}_{title}.md

    Returns:
        Path to the created markdown file
    """
    transcript_path = transcript_path.resolve()

    # Validate input file
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript file not found: {transcript_path}")

    if not transcript_path.is_file():
        raise ValueError(f"Path is not a file: {transcript_path}")

    # Read transcript content
    logger.info(f"Reading transcript from: {transcript_path}")
    transcript_content = transcript_path.read_text(encoding="utf-8")

    if not transcript_content.strip():
        raise ValueError("Transcript file is empty")

    # Load config from directory hierarchy
    tconfig = read_config_from_directory_hierarchy(transcript_path)

    # Perform the summarization
    logger.info(f"Generating summary using model: {tconfig.note_model}")
    title, note_content = llm.summarize.summarize_transcript(
        ll_model=tconfig.note_model,
        transcript=transcript_content,
        prompt=tconfig.note_prompt,
    )

    # Determine output path
    if output_path is None:
        filename_base = generate_new_filename(tconfig.datetime_fmt, transcript_path, title)
        output_path = transcript_path.parent / f"{filename_base}.md"
    else:
        output_path = output_path.resolve()

    # Write the note
    full_content = f"# {title}\n\n{note_content}"
    output_path.write_text(full_content, encoding="utf-8")
    logger.info(f"Summary written to: {output_path}")

    return output_path


def process_audio_file(
    index: VaultIndex, vault_root: Path, dry_run: bool, audio_path: Path
) -> None | Path:
    """Process a single audio file through the complete workflow.

    All configuration for this is read from the directory hierarchy of the audio file.
    """
    if vault_root / ".trash" in audio_path.parents:
        return None

    audio_filename = audio_path.name
    tconfig = read_config_from_directory_hierarchy(audio_path)

    if dry_run:
        print(tconfig)

    if tconfig.skip_dir:
        return None

    linking_notes = find_linking_notes(index, vault_root, audio_path)
    if not linking_notes:
        logger.info(f"No notes link to {audio_path}; we will leave this one untranscribed.")
        return None

    if any(
        link_line_has_tag(index, in_md_file=note, target_file=audio_path, tag="#diarize")
        for note in linking_notes
    ):
        logger.info(f"Skipping {audio_path} - tagged #diarize (use coco-meeting instead)")
        return None

    logger.info(f"Processing audio file: {audio_path}")
    original_audio_hash = hash_file(audio_path)

    transcript_file = transcribe_audio_file(
        audio_path,
        transcription_model=tconfig.transcription_model,
        transcription_prompt=tconfig.transcription_prompt,
        reformat_model=tconfig.reformat_model,
        split_audio_approx_every_s=tconfig.split_audio_approx_every_s,
    )
    title, note = llm.summarize.summarize_transcript(
        tconfig.note_model,
        transcript=transcript_file.read_text(),
        prompt=tconfig.note_prompt,
    )

    filename_base = generate_new_filename(tconfig.datetime_fmt, audio_path, title)
    new_audio_path = create_unique_file_path(
        audio_path, interpret_dir_config(vault_root, audio_path, tconfig.audio_dir), filename_base
    )
    copy_file(audio_path, new_audio_path, dry_run=dry_run)

    # Create transcript note
    transcript_note_path = (
        interpret_dir_config(vault_root, audio_path, tconfig.notes_dir) / f"{filename_base}.md"
    )
    create_transcript_note(
        vault_root,
        audio_path if dry_run else new_audio_path,
        transcript_note_path,
        title,
        note,
        original_audio_hash,
    )

    # Verify file integrity before final operations
    if hash_file(audio_path) != original_audio_hash:
        raise ValueError(f"File hash changed during processing for {audio_filename}, aborting")

    replace_links_in_notes(
        index, vault_root, linking_notes, audio_path, transcript_note_path, title, dry_run=dry_run
    )
    if not dry_run:
        logger.info(f"Removing original audio file because everything else succeeded: {audio_path}")
        audio_path.unlink()

    logger.info(f"Successfully processed {audio_path} -> {transcript_note_path.relative_to(vault_root)}")
    return transcript_note_path


def process_vault_recordings(process_vault_path: Path, dry_run: bool) -> None:
    """Main function to process all audio files in the vault."""
    vault_root = find_vault_root(process_vault_path)
    index = build_vault_index(vault_root)
    logger.info(f"Built vault index with {len(index)} unique file stems")

    process_recording = partial(process_audio_file, index, vault_root, dry_run)

    if process_vault_path.is_file():
        logger.info(f"Processing a single file: {process_vault_path}")
        all_audio_files = [process_vault_path]
    else:
        # Find all unrenamed audio recordings
        audio_patterns = ["Recording*.webm", "Recording*.m4a"]
        all_audio_files = itertools.chain.from_iterable(  # type: ignore[assignment]
            process_vault_path.rglob(p) for p in audio_patterns
        )
        logger.info(f"Looking for linked audio recordings in directory: {process_vault_path}")

    processed_count = 0
    error_count = 0
    for audio_file in all_audio_files:
        try:
            if process_recording(audio_file):
                processed_count += 1
        except Exception as e:
            logger.exception(f"Error processing {audio_file}: {e}")
            error_count += 1

    msg = f"Files successfully processed: {processed_count}"
    if error_count > 0:
        msg += f"; errors encountered: {error_count}"
    if processed_count + error_count > 0:
        logger.info(msg)


def main() -> None:
    """Entry point for `cc` command - process audio recordings."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(
        description=(
            "The (overly?) Confident Confidant. "
            "Turn audio recordings into helpful Obsidian notes with transcripts."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "process_vault_dir",
        type=Path,
        help="The directory that you want to process, which should be within an Obsidian vault.",
    )
    parser.add_argument(
        "--no-mutate",
        action="store_true",
        help="Don't do the actual file move and link mutation - this is a quasi dry-run.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Run the script in a loop, processing new files as they appear.",
    )

    args = parser.parse_args()
    process_vault_dir = args.process_vault_dir.resolve()
    run = partial(process_vault_recordings, process_vault_dir, args.no_mutate)
    run()
    if args.loop:
        while True:
            time.sleep(10)
            run()


def main_summarize() -> None:
    """Entry point for `cc-summarize` command - summarize existing transcripts."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(
        description="Summarize an existing transcript file using an LLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "transcript_file",
        type=Path,
        help="Path to the transcript .txt file to summarize",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output file path (default: generates {timestamp}_{title}.md in same directory)",
    )

    args = parser.parse_args()
    summarize_transcript(args.transcript_file, args.output)


if __name__ == "__main__":
    main()
