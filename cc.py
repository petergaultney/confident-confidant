#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "litellm",
#   "openai",
#   "hjson",
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
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache, partial
from pathlib import Path

import hjson
import openai
from litellm import completion

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _anthropic_api_key() -> str:
    return open(os.path.expanduser("~/.keys/anthropic-api")).read().strip()


def _openai_api_key() -> str:
    return open(os.path.expanduser("~/.keys/openai-api")).read().strip()


def _set_api_key(env_var: str):
    os.environ[env_var] = os.environ.get(env_var) or _API_KEYS[env_var]()


_API_KEYS = {
    "OPENAI_API_KEY": _openai_api_key,
    "ANTHROPIC_API_KEY": _anthropic_api_key,
}


def activate_api_keys():
    """So you don't have to have all of them in case you only use one."""
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


def transcribe_audio(audio_path: Path, transcription_model: str, transcription_prompt: str) -> str:
    """Transcribe audio file using OpenAI Whisper or newer models."""
    transcription_model = transcription_model or "whisper-1"
    logger.info(
        f"Transcribing audio: {audio_path} with {transcription_model}, using prompt: {transcription_prompt}"
    )
    activate_api_keys()
    client = openai.OpenAI()

    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model=transcription_model,
            file=audio_file,
            prompt=transcription_prompt,
        )

    return transcript.text


def _test_transcript_equivalence(original: str, modified: str) -> bool:
    """Check if the second string contains the first, ignoring whitespace and capitalization."""
    original_cleaned = re.sub(r"\s+", " ", original).strip().lower()
    modified_cleaned = re.sub(r"\s+", " ", modified).strip().lower()
    return original_cleaned in modified_cleaned


_DEFAULT_NOTE_PROMPT = textwrap.dedent(
    """
    2. A concise summary (2-3 sentences) - I am the speaker, so use first-person perspective
    3. A complete and organized markdown-native outline,
        with the highest level of heading being `##`,
        because it will be embedded inside an existing markdown document.
        If it makes sense for the content, try to orient around high level categories like
        "intuitions", "constraints", "assumptions",
        "alternatives or rejected ideas", "tradeoffs", and "next steps",
        though these don't necessarily need to be present or even the headings.
       If the audio is more like a retelling of my day, then write the outline without headings,
        but in three distinct sections as:
          A. bullet points of things I worked on
          B. specific 'tasks' for the future (format as Markdown tasks, e.g. ` - [ ] <task text>`
          C. Remaining insights, through-lines, or points to ponder.
        If you think it fits neither of these categories, use your best judgment on the outline.
    4. A readable transcript of the audio, broken up into paragraphs.
        Never leave the most key thoughts buried in long paragraphs.
        Change ONLY whitespace!

    Format your response as:
    # Summary

    {summary}

    # Outline

    {outline}

    # Full transcript

    {full_readable_transcript}
    """
)


@dataclass
class ConfidentConfidantConfig:
    transcription_model: str = "whisper-1"  # or "gpt-4o-transcribe"
    transcription_prompt: str = ""
    note_model: str = "anthropic/claude-sonnet-4-20250514"
    note_prompt: str = _DEFAULT_NOTE_PROMPT
    audio_dir: str = "./audio"
    # dirs can be relative to the original audio file (starts with ./) or to the vault root (starts with :)
    transcripts_dir: str = "./transcripts"
    skip_dir: bool = False
    # if True, skip any files in this directory, and in subdirectories that do not have more specific config.


_DEFAULT_CONFIG = ConfidentConfidantConfig()


def extract_heading_content(markdown_text: str, header_string: str) -> None | str:
    """
    Extract content between a heading and the next same-level heading,
    ignoring any headers that appear within code blocks.

    Args:
        markdown_text: The full markdown string
        header_string: The full header string (e.g., "# Introduction" or "## Main Content")

    Returns:
        The content between the heading and next same-level heading, or None if not found
    """
    # Calculate heading level by counting # at the start
    heading_level = 0
    for char in header_string:
        if char != "#":
            break
        heading_level += 1

    if heading_level == 0:
        raise ValueError("Header string must start with # characters")

    # Extract the heading text
    heading_text = header_string.lstrip("#").strip()
    hash_marks = "#" * heading_level

    lines = markdown_text.split("\n")
    in_code_block = False
    found_target_heading = False
    content_lines = []

    for line in lines:
        # Check for fenced code block boundaries
        if line.startswith("```") and not line.startswith("````"):
            in_code_block = not in_code_block
            if found_target_heading:
                content_lines.append(line)
            continue

        # If we're in a code block, don't process headers
        if in_code_block:
            if found_target_heading:
                content_lines.append(line)
            continue

        # Check if this is our target heading
        if line.startswith(hash_marks + " ") and line[len(hash_marks) :].strip() == heading_text:
            found_target_heading = True
            continue

        # Check if this is another heading at the same level (end condition)
        if found_target_heading and line.startswith(hash_marks + " "):
            break

        # Collect content if we're in the target section
        if found_target_heading:
            content_lines.append(line)

    return "\n".join(content_lines).rstrip() if found_target_heading else None


def extract_code_block(text: str) -> None | str:
    """
    Extract the first code block from the given text.

    Args:
        text: The text containing a potential code block

    Returns:
        The code block content (without the backticks), or None if not found
    """
    if not text:
        return None

    # Pattern explanation:
    # ^\s*```[^`].*?$ - Opening line: exactly 3 backticks, non-backtick chars, end of line
    # (.*?) - Capture the code content (non-greedy)
    # ^\s*```\s*$ - Closing line: optional whitespace, exactly 3 backticks, optional whitespace, end of line
    pattern = r"^\s*```[^`].*?$\n(.*?)^\s*```\s*$"

    match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
    return match.group(1).rstrip("\n") if match else None


def _parse_hjson(some_text: str) -> dict[str, ty.Any]:
    """Parse HJSON text into a dictionary."""
    try:
        return hjson.loads(some_text)
    except hjson.HjsonDecodeError:
        return hjson.loads("{" + some_text + "}")


@lru_cache  # cache this so we don't have to re-read the directory hierarchy for every file.
def _read_config_from_directory_hierarchy(any_path: Path) -> ConfidentConfidantConfig:
    """Return the first config found in the first file where the config is not identical to the default config.

    'First' means we look 'upward' in the directory hierarchy for a config file, starting
    from the given path.
    """
    current_dir = any_path if any_path.is_dir() else any_path.parent

    if any_path.parent == any_path:
        logger.info(f"Using default config for {current_dir}")
        return _DEFAULT_CONFIG  # reached the root of the filesystem, no config found

    def parse_config(config_md: str) -> ConfidentConfidantConfig:
        config_section = extract_heading_content(config_md, "# Confident Confidant Config")
        if not config_section:
            return _DEFAULT_CONFIG

        config = ConfidentConfidantConfig()

        base_config_text = extract_heading_content(config_section, "## Base Config") or ""
        if escaped_config := extract_code_block(base_config_text):
            base_config_text = escaped_config
        base_config_text = base_config_text.strip()
        if base_config_text:
            base_config = _parse_hjson(base_config_text)
            config.audio_dir = base_config.get("audio_dir") or config.audio_dir
            config.transcripts_dir = base_config.get("transcripts_dir") or config.transcripts_dir
            config.skip_dir = base_config.get("skip_dir", False)
            config.transcription_model = (
                base_config.get("transcription_model") or config.transcription_model
            )
            config.note_model = base_config.get("note_model") or config.note_model

        if transcription_prompt := extract_heading_content(config_section, "## Transcription Prompt"):
            config.transcription_prompt = transcription_prompt

        note_prompt = extract_heading_content(config_section, "## Note Prompt")
        if note_prompt:
            if escaped_prompt := extract_code_block(note_prompt):
                note_prompt = escaped_prompt
            config.note_prompt = note_prompt.strip()

        return config

    config_files = (
        current_dir / ".cc-config.md",
        current_dir / "cc-config.md",
        current_dir / ".cc-config.txt",
        current_dir / "cc-config.txt",
        current_dir / (current_dir.name + ".md"),
        # look inside a 'folder note', if any, under the heading `# Transcript Prompt`
    )

    for config_file in config_files:
        if config_file.is_file():
            config = parse_config(config_file.read_text())
            if config and config != _DEFAULT_CONFIG:
                logger.info(f"Using discovered config for {current_dir}, from {config_file}")
                return config

    return _read_config_from_directory_hierarchy(current_dir.parent)  # recurse up the directory tree


def transform_transcript_into_note(
    ll_model: str,
    transcript: str,
    prompt: str,
) -> tuple[str, str]:
    """Get title and summary note from LLM; the summary note will be formatted as returned by the LLM.

    Your prompt MUST NOT redefine the first line of output from the LLM, which is
    specified to be a short title that can also be used as a filename.

    The function also tacks on the raw transcript if your resulting note does not include
    some whitespace-compressed version of the transcript.

    """
    logger.info(f"Getting note and title from {ll_model}")

    prompt = (
        textwrap.dedent(
            f"""
        Please analyze the raw transcript at the end and provide:

        1. A short title (3-7 words, suitable for a filename) - put this as the very first
           line, followed by a newline, regardless of any formatting instructions that follow.
           This must never be missing, and it must always be at least 3 words and more than 7,
           and they must be as unique as possible using the content of the transcript, since this will
           be part of a filename.
        """
        )
        + (prompt or _DEFAULT_NOTE_PROMPT)
        + (
            "\n\n" + "Remember - regardless of the rest of the format of your response,"
            " the very first line must be a 3-7 word title on a line by itself."
        )
        + f"\n\nRaw transcript to be analyzed:\n{transcript}"
    )

    activate_api_keys()
    print(prompt)
    response = completion(model=ll_model, messages=[{"role": "user", "content": prompt}])
    content = response["choices"][0]["message"]["content"]
    print(content)

    first_line_is_title, rest_of_note = content.split("\n", 1)
    if not _test_transcript_equivalence(transcript, rest_of_note):
        rest_of_note += f"\n# Raw transcript\n{transcript}"
    return rest_of_note, first_line_is_title.strip()


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

    content = (
        f"""
# {title}

![[{new_audio_path.relative_to(vault_root)}]]
size: {file_size_mb:.2f} MB | processed: {datetime.now().strftime("%Y-%m-%d %H:%M")} | sha256: `{file_hash}`

{prompt_response}
""".strip()
        + "\n"
    )

    transcript_note_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_note_path.write_text(content, encoding="utf-8")
    return transcript_note_path


def create_new_audio_path(original_path: Path, target_dir: Path, new_filename: str) -> Path:
    new_path = target_dir / f"{new_filename}{original_path.suffix}"

    # Handle filename conflicts
    counter = 1
    while new_path.exists():
        new_path = target_dir / f"{new_filename}-{counter}{original_path.suffix}"
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
    root: Path,
    linking_notes: list[Path],
    old_filename: str,
    transcript_note_path: Path,
    dry_run: bool = False,
) -> None:
    """Replace audio links with transcript links in linking notes."""
    old_pattern = rf"!\[\[{re.escape(old_filename)}\]\]"
    transcript_note_link = f"[[{transcript_note_path.relative_to(root)}]]"

    for note_path in linking_notes:
        try:
            content = note_path.read_text(encoding="utf-8")
            new_content = re.sub(old_pattern, transcript_note_link, content)

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


@lru_cache
def _find_vault_root(any_path: Path) -> Path:
    # recursive
    """Find the root of the Vault by looking for the .obsidian or .git directory."""
    if (any_path / ".obsidian").is_dir() or (any_path / ".git").is_dir():
        return any_path

    if any_path.parent == any_path:
        return any_path

    return _find_vault_root(any_path.parent)


def _interpret_dir_config(vault_root: Path, audio_path: Path, config_str: str) -> Path:
    """
    ./ means relative to the audio file's directory.
    : means relative to the vault root.
    """
    if config_str.startswith("./"):
        return audio_path.parent / config_str[2:]
    elif config_str.startswith(":"):
        return vault_root / config_str[1:]
    raise ValueError(f"Invalid directory config: {config_str}. Must start with './' or ':'.")


def process_audio_file(
    dry_run: bool,
    audio_path: Path,
) -> None | Path:
    """Process a single audio file through the complete workflow.

    All configuration for this is read from the directory hierarchy of the audio file.
    """

    vault_root = _find_vault_root(audio_path)
    audio_filename = audio_path.name
    tconfig = _read_config_from_directory_hierarchy(audio_path)

    if dry_run:
        print(tconfig)

    if tconfig.skip_dir:
        return None

    linking_notes = find_linking_notes(vault_root, audio_filename)
    if not linking_notes:
        logger.info(f"No notes link to {audio_filename}; we will leave this one untranscribed.")
        return None

    logger.info(f"Processing audio file: {audio_path}")
    original_audio_hash = hash_file(audio_path)

    note, title = transform_transcript_into_note(
        tconfig.note_model,
        transcribe_audio(  # TRANSCRIPTION HAPPENS HERE
            audio_path, tconfig.transcription_model, tconfig.transcription_prompt
        ),
        prompt=tconfig.note_prompt,
    )

    filename_base = generate_new_filename(audio_path, title)
    new_audio_path = create_new_audio_path(
        audio_path, _interpret_dir_config(vault_root, audio_path, tconfig.audio_dir), filename_base
    )
    _copy_file(audio_path, new_audio_path, dry_run=False)
    # always copy the file, even in dry run mode - this is non-mutating to existing notes
    # and data, lets us make sure that the transcript note is immediately valid.

    # Create transcript note
    transcript_note_path = (
        _interpret_dir_config(vault_root, audio_path, tconfig.transcripts_dir) / f"{filename_base}.md"
    )
    transcript_note = create_transcript_note(
        vault_root,
        new_audio_path,
        transcript_note_path,
        title,
        note,
        original_audio_hash,
    )

    # Verify file integrity before final operations
    if hash_file(audio_path) != original_audio_hash:
        raise ValueError(f"File hash changed during processing for {audio_filename}, aborting")

    replace_links_in_notes(
        vault_root, linking_notes, audio_filename, transcript_note_path, dry_run=dry_run
    )
    if not dry_run:
        logger.info(f"Removing original audio file because everything else succeeded: {audio_path}")
        audio_path.unlink()

    logger.info(f"Successfully processed {audio_path} -> {transcript_note_path.relative_to(vault_root)}")
    return transcript_note_path


def process_vault_recordings(
    process_vault_path: Path,
    process_recording: ty.Callable[[Path], None | Path],
) -> None:
    """Main function to process all audio files in the vault."""
    if process_vault_path.is_file():
        logger.info(f"Processing a single file: {process_vault_path}")
        all_audio_files = [process_vault_path]
    else:
        # Find all unrenamed audio recordings
        audio_patterns = ["Recording *.webm", "Recording *.m4a"]
        all_audio_files = itertools.chain.from_iterable(
            process_vault_path.rglob(p) for p in audio_patterns
        )
        logger.info(f"Will look for audio recordings in directory: {process_vault_path}")

    processed_count = 0
    for audio_file in all_audio_files:
        try:
            if process_recording(audio_file):
                processed_count += 1
        except Exception as e:
            logger.exception(f"Error processing {audio_file}: {e}")

    logger.info(f"Processing complete. Files successfully processed: {processed_count}")


if __name__ == "__main__":
    import argparse

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
    run = partial(
        process_vault_recordings,
        args.process_vault_dir,
        partial(
            process_audio_file,
            args.no_mutate,
        ),
    )

    run()
    if args.loop:
        while True:
            time.sleep(10)
            run()
