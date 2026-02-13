import logging
import textwrap
import typing as ty
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import hjson

from cc.md import extract_code_block, extract_heading_content, extract_headings_by_prefix

logger = logging.getLogger(__name__)

DEFAULT_NOTE_PROMPT = textwrap.dedent(
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
    transcription_context: str = ""
    # names, terms, pronunciations — passed to both transcription API and summarizer
    reformat_model: str = "gpt-4o"
    # ^ for stitching together chunk-transcripts if the audio file is long
    split_audio_approx_every_s: int = 20 * 60  # 20 minutes
    diarization_model: str = "gpt-4o-transcribe-diarize"

    # for summarizing:
    note_model: str = "gpt-5.2"
    note_prompts: dict[str, str] = field(default_factory=dict)
    # keyed by prompt name ("default", "meeting", etc.)

    # for outputting:
    audio_dir: str = "./cc/audio"
    # dirs can be relative to the original audio file (starts with ./) or to the vault root (starts with :)
    notes_dir: str = "./cc/notes"
    datetime_fmt: str = "%y-%m-%d_%H%M"  # very opinionated, sorry - I don't expect to live until 2100
    skip_dir: bool = False
    # if True, skip any files in this directory, and in subdirectories that do not have more specific config.


DEFAULT_CONFIG = ConfidentConfidantConfig(note_prompts={"default": DEFAULT_NOTE_PROMPT})


def _parse_hjson(some_text: str) -> dict[str, ty.Any]:
    """Parse HJSON text into a dictionary."""
    try:
        return hjson.loads(some_text)
    except hjson.HjsonDecodeError:
        return hjson.loads("{" + some_text + "}")


def _parse_config_md(config_md: str) -> ConfidentConfidantConfig:
    cc_config_md = extract_heading_content(config_md, "Confident Confidant Config")
    if not cc_config_md:
        return ConfidentConfidantConfig()

    config = ConfidentConfidantConfig()

    base_config_text = extract_heading_content(cc_config_md, "Base Config") or ""
    if escaped_config := extract_code_block(base_config_text):
        base_config_text = escaped_config
    base_config_text = base_config_text.strip()
    if base_config_text:
        base_config = _parse_hjson(base_config_text)
        # backwards compat
        if "note_prompt" in base_config:
            config.note_prompts["default"] = base_config.pop("note_prompt")
        if "transcription_prompt" in base_config:
            base_config["transcription_context"] = base_config.pop("transcription_prompt")
        for key, value in base_config.items():
            if hasattr(config, key):
                setattr(config, key, value)

    # support both old and new heading names
    for heading in ("Transcription Context", "Transcription Prompt"):
        if text := extract_heading_content(cc_config_md, heading):
            config.transcription_context = text
            break

    for name, content in extract_headings_by_prefix(cc_config_md, "Note Prompt").items():
        if escaped := extract_code_block(content):
            content = escaped
        config.note_prompts[name] = content.strip()

    return config


_CONFIG_FILENAMES = (".cc-config.md", "cc-config.md")


def _find_config_in_dir(current_dir: Path) -> ConfidentConfidantConfig | None:
    """Return the first non-empty config found in current_dir, or None."""
    config_files = (
        *(current_dir / name for name in _CONFIG_FILENAMES),
        current_dir / (current_dir.name + ".md"),
    )
    for config_file in config_files:
        if config_file.is_file():
            config = _parse_config_md(config_file.read_text())
            if config != ConfidentConfidantConfig():
                return config

    return None


@lru_cache
def collect_configs_root_to_file(any_path: Path) -> tuple[ConfidentConfidantConfig, ...]:
    """Collect all non-empty configs from filesystem root down to any_path's directory."""
    configs: list[ConfidentConfidantConfig] = []
    current = any_path if any_path.is_dir() else any_path.parent
    while current != current.parent:
        config = _find_config_in_dir(current)
        if config is not None:
            configs.append(config)
        current = current.parent
    configs.reverse()  # root-to-file order
    return tuple(configs)


@lru_cache
def read_config_from_directory_hierarchy(any_path: Path) -> ConfidentConfidantConfig:
    """Return the nearest non-default config for scalar fields.

    Walks upward from any_path; returns the config closest to the file.
    For prompt resolution, use resolve_prompt with collect_configs_root_to_file instead.
    """
    configs = collect_configs_root_to_file(any_path)
    return configs[-1] if configs else DEFAULT_CONFIG


def resolve_prompt(
    configs: tuple[ConfidentConfidantConfig, ...] | list[ConfidentConfidantConfig],
    prompt_names: list[str],
) -> str:
    """Resolve the final prompt from a hierarchy of configs and selected prompt names.

    For each name in prompt_names (in tag order), collect that name's prompt from
    every config (root-to-file), concatenating with '\\n\\n'.
    If prompt_names is empty, use ["default"].
    """
    if not prompt_names:
        prompt_names = ["default"]

    parts: list[str] = []
    for name in prompt_names:
        for config in configs:
            if name in config.note_prompts:
                parts.append(config.note_prompts[name])

    return "\n\n".join(parts) if parts else DEFAULT_NOTE_PROMPT


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
