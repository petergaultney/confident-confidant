#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "litellm",
#   "openai",
# ]
# ///

import hashlib
import itertools
import logging
import os
import re
import shutil
import textwrap
import typing as ty
from datetime import datetime
from functools import partial
from pathlib import Path

import openai
from litellm import completion

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _anthropic_api_key() -> str:
    return open(os.path.expanduser("~/.keys/claude-api")).read().strip()


def _openai_api_key() -> str:
    return open(os.path.expanduser("~/.keys/openai-api")).read().strip()


def _set_api_key(env_var: str):
    os.environ[env_var] = os.environ.get(env_var) or _API_KEYS[env_var]()


_API_KEYS = {
    "OPENAI_API_KEY": _openai_api_key,
    "ANTHROPIC_API_KEY": _anthropic_api_key,
}
list(map(_set_api_key, _API_KEYS.keys()))


def hash_file(file_path: Path) -> str:
    """Generate SHA-256 hash of file contents."""
    logger.info(f"Hashing file: {file_path}")
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def find_linking_notes(vault_root: Path, audio_filename: str) -> list[Path]:
    """Find all text notes that link to the given audio file."""
    logger.info(f"Looking for notes linking to: {audio_filename}")
    linking_notes = []
    link_pattern = rf"!\[\[{re.escape(audio_filename)}\]\]"

    for note_path in vault_root.rglob("*.md"):
        try:
            content = note_path.read_text(encoding="utf-8")
            if re.search(link_pattern, content):
                linking_notes.append(note_path)
                logger.info(f"Found linking note: {note_path}")
        except Exception as e:
            logger.warning(f"Could not read {note_path}: {e}")

    return linking_notes


def transcribe_audio(audio_path: Path) -> str:
    """Transcribe audio file using OpenAI Whisper."""
    logger.info(f"Transcribing audio: {audio_path}")
    client = openai.OpenAI()

    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file)

    return transcript.text


def _test_whitespace_equivalence(original: str, modified: str) -> bool:
    """Check if the second string contains the first, ignoring whitespace."""
    original_cleaned = re.sub(r"\s+", " ", original).strip()
    modified_cleaned = re.sub(r"\s+", " ", modified).strip()
    return original_cleaned in modified_cleaned


_DEFAULT_PROMPT = textwrap.dedent(
    f"""
    2. A concise summary (2-3 sentences) - I am the speaker, so use first-person perspective
    3. A complete and organized markdown-native outline,
        with the highest level of heading being `##`,
        because it will be embedded inside an existing markdown document.
        If it makes sense for the content, try to orient around high level categories like
        "intuitions", "constraints", "assumptions",
        "alternatives or rejected ideas", "tradeoffs", and "next steps",
        though these don't necessarily need to be present or even the headings.
       If the audio is more like a retelling of my day, then present the summary without headings,
        but in three distinct sections as:
          A. bullet points of things I worked on
          B. specific 'tasks' for the future (format as Obsidian markdown tasks, e.g. ` - [ ] <task text`
          C. Remaining insights, through-lines, or points to ponder.
        If you think it fits neither of these categories, use your best judgment on the outline.
    4. A readable transcript of the audio, broken up into paragraphs.
        Never leave the most key thoughts buried in long paragraphs.
        Change ONLY whitespace!

    Format your response as:
    # Summary

    {{summary}}

    # Outline

    {{outline}}

    # Full transcript

    {{full_readable_transcript}}
    """
)


def transform_transcript_into_note(
    claude_model: str,
    transcript: str,
    prompt: str = _DEFAULT_PROMPT,
) -> tuple[str, str]:
    """Get summary and title from Claude."""
    logger.info("Getting summary and title from Claude")

    prompt = (
        textwrap.dedent(
            f"""
        Please analyze the transcript at the end and provide:

        1. A short title (3-7 words, suitable for a filename) - put this as the very last
        line, preceded by a newline.
        """
        )
        + (prompt or _DEFAULT_PROMPT)
        + f"\nRaw Transcript:\n{transcript}"
    )

    response = completion(model=claude_model, messages=[{"role": "user", "content": prompt}])
    content = response["choices"][0]["message"]["content"]

    note, last_line_is_title = content.rsplit("\n", 1)
    if not _test_whitespace_equivalence(transcript, note):
        note += f"# Raw transcript\n{transcript}"
    return note, last_line_is_title


def _sanitize_title(title: str) -> str:
    """Convert title to filesystem-safe filename."""
    # Replace spaces with dashes, remove unsafe characters
    sanitized = re.sub(r"[^\w\s-]", "", title)
    sanitized = re.sub(r"\s+", "-", sanitized)
    sanitized = sanitized.strip("-").lower()
    return sanitized


def generate_new_filename(audio_path: Path, title: str) -> str:
    """Generate new filename with audio file creation timestamp and title."""
    # Get the creation time of the audio file
    creation_time = datetime.fromtimestamp(audio_path.stat().st_ctime)
    timestamp = creation_time.strftime("%y-%m-%d_%H%M")
    return f"{timestamp}_{_sanitize_title(title)}"


def create_transcript_note(
    vault_root: Path,
    new_audio_path: Path,
    transcript_note_path: Path,
    title: str,
    prompt_response: str,
    file_hash: str,
) -> Path:
    """Create the transcript note with metadata."""
    logger.info(f"Creating transcript note: {transcript_note_path}")

    # Get file stats for metadata
    stat = new_audio_path.stat()
    file_size_mb = stat.st_size / (1024 * 1024)

    content = f"""
# {title}

![[{new_audio_path.relative_to(vault_root)}]]
size: {file_size_mb:.2f} MB | processed: {datetime.now().strftime("%Y-%m-%d %H:%M")} | sha256: `{file_hash}`

{prompt_response}
"""

    transcript_note_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_note_path.write_text(content, encoding="utf-8")
    return transcript_note_path


def create_new_audio_path(original_path: Path, target_dir: Path, new_filename: str) -> Path:
    new_path = target_dir / f"{new_filename}.{original_path.suffix}"

    # Handle filename conflicts
    counter = 1
    while new_path.exists():
        new_path = target_dir / f"{new_filename}-{counter}.{original_path.suffix}"
        counter += 1

    return new_path


def _copy_file(original_path: Path, new_path: Path, dry_run: bool = True) -> None:
    if not dry_run:
        logger.info(f"Copying audio file: {original_path} -> {new_path}")
        new_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original_path, new_path)
    else:
        logger.info(f"DRY RUN: Would copy {original_path} -> {new_path}")


def replace_links_in_notes(
    linking_notes: list[Path],
    old_filename: str,
    new_link: str,
    dry_run: bool = False,
) -> None:
    """Replace audio links with transcript links in linking notes."""
    old_pattern = rf"!\[\[{re.escape(old_filename)}\]\]"

    for note_path in linking_notes:
        try:
            content = note_path.read_text(encoding="utf-8")
            new_content = re.sub(old_pattern, new_link, content)

            if not dry_run:
                note_path.write_text(new_content, encoding="utf-8")
                logger.info(f"Updated links in: {note_path}")
            else:
                logger.info(
                    f"DRY RUN: Would update links in: {note_path} - the new text would look like"
                )
                print(new_content)
        except Exception as e:
            logger.error(f"Failed to update links in {note_path}: {e}")


def process_audio_file(
    model: str,
    dry_run: bool,
    prompt_file_path: Path,
    vault_root: Path,
    audio_dir: Path,
    transcript_dir: Path,
    audio_path: Path,
) -> None:
    """Process a single audio file through the complete workflow."""
    audio_filename = audio_path.name
    logger.info(f"Processing audio file: {audio_filename}")

    # Find linking notes
    linking_notes = find_linking_notes(vault_root, audio_filename)
    if not linking_notes:
        logger.info(f"No linking notes found for {audio_filename}, skipping")
        return

    # Hash the original file
    original_hash = hash_file(audio_path)

    # Transcribe audio
    transcript = transcribe_audio(audio_path)

    # Get summary and title
    note, title = transform_transcript_into_note(
        model,
        transcript,
        prompt=prompt_file_path.read_text() if prompt_file_path and prompt_file_path.is_file() else "",
    )
    filename_base = generate_new_filename(audio_path, title)
    new_audio_path = create_new_audio_path(audio_path, audio_dir, filename_base)
    _copy_file(audio_path, new_audio_path, dry_run=False)
    # always copy the file, even in dry run mode - this lets us make sure that the transcript note is immediately valid.

    # Create transcript note
    transcript_note_path = transcript_dir / f"{filename_base}.md"
    transcript_note = create_transcript_note(
        vault_root,
        new_audio_path,
        transcript_note_path,
        title,
        note,
        original_hash,
    )

    # Verify file integrity before final operations
    current_hash = hash_file(audio_path)
    if current_hash != original_hash:
        raise ValueError(f"File hash changed during processing for {audio_filename}, aborting")

    # Copy and rename audio file
    # Replace links in notes
    new_transcript_note_link = f"[[{transcript_note_path.relative_to(vault_root)}]]"
    replace_links_in_notes(linking_notes, audio_filename, new_transcript_note_link, dry_run=dry_run)
    # Remove original audio file
    logger.info(f"Removing original audio file because everything else succeeded: {audio_path}")
    if not dry_run:
        audio_path.unlink()

    logger.info(f"Successfully processed {audio_path} -> {new_transcript_note_link}")


def process_vault_recordings(
    vault_output_dir: Path,
    process_recording: ty.Callable[[Path, Path, Path, Path], None],
) -> None:
    """Main function to process all audio files in the vault."""
    audio_dir = vault_output_dir.resolve() / "audio"
    transcript_dir = vault_output_dir.resolve() / "transcripts"

    vault_path = Path(vault_output_dir).resolve()
    while not (vault_path / ".obsidian").is_dir():
        vault_path = vault_path.parent
        assert vault_path.exists(), f"Vault path {vault_output_dir} does not contain .obsidian directory"
        assert vault_path != vault_path.parent, f"Vault path {vault_output_dir} is not a valid vault"

    logger.info(f"Starting audio processing in vault: {vault_path}")
    logger.info(f"Audio directory: {audio_dir}")
    logger.info(f"Transcript directory: {transcript_dir}")

    # Find all audio recordings not in the target audio directory
    audio_patterns = ["Recording *.webm", "Recording *.m4a"]
    all_audio_files = itertools.chain.from_iterable(vault_path.rglob(p) for p in audio_patterns)

    processed_count = 0
    for audio_file in all_audio_files:
        # Skip files already in the target directory
        if audio_dir in audio_file.parents:
            continue

        try:
            process_recording(vault_path, audio_dir, transcript_dir, audio_file)
            processed_count += 1
        except Exception as e:
            logger.exception(f"Error processing {audio_file}: {e}")

    logger.info(f"Processing complete. Files successfully processed: {processed_count}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process audio files in a vault.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "vault_output_dir",
        type=Path,
        help="Directory within Vault to store processed audio and transcripts",
    )
    parser.add_argument(
        "--prompt-file",
        "-p",
        help="A file containing a custom prompt for handling the transcript",
        default=None,
    )
    parser.add_argument(
        "--model",
        default="anthropic/claude-sonnet-4-20250514",
    )
    parser.add_argument(
        "--execute",
        "-x",
        action="store_true",
        help="Run the actual file move and note edits (default is dry run)",
    )
    args = parser.parse_args()

    process_vault_recordings(
        args.vault_output_dir,
        partial(
            process_audio_file,
            args.model,
            not args.execute,
            args.prompt_file,
        ),
    )
