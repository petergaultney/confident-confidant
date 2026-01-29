import logging
import textwrap
import typing as ty
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import hjson

from cc.md import extract_code_block, extract_heading_content

logger = logging.getLogger(__name__)

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
    # for transcribing:
    transcription_model: str = "gpt-4o-transcribe"  # or "whisper-1", or "gpt-4o-mini-transcribe"
    transcription_prompt: str = ""
    reformat_model: str = "gpt-4o"
    # ^ for stitching together chunk-transcripts if the audio file is long
    split_audio_approx_every_s: int = 20 * 60  # 20 minutes
    diarization_model: str = "gpt-4o-transcribe-diarize"

    # for summarizing:
    note_model: str = "anthropic/claude-sonnet-4-20250514"
    note_prompt: str = _DEFAULT_NOTE_PROMPT

    # for outputting:
    audio_dir: str = "./cc/audio"
    # dirs can be relative to the original audio file (starts with ./) or to the vault root (starts with :)
    notes_dir: str = "./cc/notes"
    datetime_fmt: str = (
        "%y-%m-%d_%H%M"  # very opinionated, sorry - I don't expect to live until 2100
    )
    skip_dir: bool = False
    # if True, skip any files in this directory, and in subdirectories that do not have more specific config.


DEFAULT_CONFIG = ConfidentConfidantConfig()


def _parse_hjson(some_text: str) -> dict[str, ty.Any]:
    """Parse HJSON text into a dictionary."""
    try:
        return hjson.loads(some_text)
    except hjson.HjsonDecodeError:
        return hjson.loads("{" + some_text + "}")


@lru_cache  # cache this so we don't have to re-read the directory hierarchy for every file.
def read_config_from_directory_hierarchy(any_path: Path) -> ConfidentConfidantConfig:
    """Return the first config found in the first file where the config is not identical to the default config.

    'First' means we look 'upward' in the directory hierarchy for a config file, starting
    from the given path.
    """
    current_dir = any_path if any_path.is_dir() else any_path.parent

    if any_path.parent == any_path:
        logger.info(f"Using default config for {current_dir}")
        return DEFAULT_CONFIG  # reached the root of the filesystem, no config found

    def parse_config(config_md: str) -> ConfidentConfidantConfig:
        cc_config_md = extract_heading_content(config_md, "Confident Confidant Config")
        if not cc_config_md:
            return DEFAULT_CONFIG

        config = ConfidentConfidantConfig()

        base_config_text = extract_heading_content(cc_config_md, "Base Config") or ""
        if escaped_config := extract_code_block(base_config_text):
            base_config_text = escaped_config
        base_config_text = base_config_text.strip()
        if base_config_text:
            base_config = _parse_hjson(base_config_text)
            for key, value in base_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        if transcription_prompt := extract_heading_content(cc_config_md, "Transcription Prompt"):
            config.transcription_prompt = transcription_prompt

        if note_prompt := extract_heading_content(cc_config_md, "Note Prompt"):
            if escaped_prompt := extract_code_block(note_prompt):
                note_prompt = escaped_prompt
            config.note_prompt = note_prompt.strip()

        return config

    config_files = (
        current_dir / ".cc-config.md",
        current_dir / "cc-config.md",
        current_dir / (current_dir.name + ".md"),
        # look inside a 'folder note', if any, under the heading `# Transcript Prompt`
    )

    for config_file in config_files:
        if config_file.is_file():
            config = parse_config(config_file.read_text())
            if config and config != DEFAULT_CONFIG:
                logger.info(f"Using discovered config for {current_dir}, from {config_file}")
                return config

    return read_config_from_directory_hierarchy(current_dir.parent)  # recurse up the directory tree


def interpret_dir_config(vault_root: Path, audio_path: Path, config_str: str) -> Path:
    """
    ./ means relative to the audio file's directory.
    : means relative to the vault root.
    """
    if config_str.startswith("./"):
        return audio_path.parent / config_str[2:]
    elif config_str.startswith(":"):
        return vault_root / config_str[1:]
    raise ValueError(f"Invalid directory config: {config_str}. Must start with './' or ':'.")
