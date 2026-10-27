from pathlib import Path

import pytest

from cc.config import (
    ConfidentConfidantConfig,
    DEFAULT_NOTE_PROMPT,
    _parse_config_md,
    read_config_from_directory_hierarchy,
    resolve_prompt,
)


def test_parse_bare_note_prompt():
    md = """\
# Confident Confidant Config

## Note Prompt
my custom prompt"""
    config = _parse_config_md(md)
    assert config.note_prompts == {"default": "my custom prompt"}


def test_parse_named_note_prompt():
    md = """\
# Confident Confidant Config

## Note Prompt: meeting
meeting instructions"""
    config = _parse_config_md(md)
    assert config.note_prompts == {"meeting": "meeting instructions"}


def test_parse_multiple_named_prompts():
    md = """\
# Confident Confidant Config

## Note Prompt
default prompt

## Note Prompt: meeting
meeting prompt"""
    config = _parse_config_md(md)
    assert "default" in config.note_prompts
    assert "meeting" in config.note_prompts


def test_parse_hjson_note_prompt_backcompat():
    md = """\
# Confident Confidant Config

## Base Config
```hjson
note_prompt: legacy prompt text
```"""
    config = _parse_config_md(md)
    assert config.note_prompts == {"default": "legacy prompt text"}


def test_parse_note_prompt_in_code_block():
    md = """\
# Confident Confidant Config

## Note Prompt: meeting

```
I am Peter. This is a meeting.
```"""
    config = _parse_config_md(md)
    assert config.note_prompts["meeting"] == "I am Peter. This is a meeting."


def test_parse_no_cc_config_heading():
    config = _parse_config_md("# Something Else\ncontent")
    assert config == ConfidentConfidantConfig()


def test_parse_scalar_fields_still_work():
    md = """\
# Confident Confidant Config

## Base Config
```hjson
note_model: anthropic/claude-haiku-3-20240307
```"""
    config = _parse_config_md(md)
    assert config.note_model == "anthropic/claude-haiku-3-20240307"


# resolve_prompt tests

def test_resolve_default_with_no_tags():
    configs = (ConfidentConfidantConfig(note_prompts={"default": "my default"}),)
    assert resolve_prompt(configs, []) == "my default"


def test_resolve_specific_tag_excludes_default():
    configs = (ConfidentConfidantConfig(note_prompts={"default": "nope", "meeting": "yes"}),)
    assert resolve_prompt(configs, ["meeting"]) == "yes"


def test_resolve_hierarchical_concatenation():
    root = ConfidentConfidantConfig(note_prompts={"meeting": "I am Peter."})
    leaf = ConfidentConfidantConfig(note_prompts={"meeting": "Focus on action items."})
    assert resolve_prompt((root, leaf), ["meeting"]) == "I am Peter.\n\nFocus on action items."


def test_resolve_multi_tag():
    config = ConfidentConfidantConfig(note_prompts={"meeting": "meeting stuff", "followup": "followup stuff"})
    result = resolve_prompt((config,), ["meeting", "followup"])
    assert result == "meeting stuff\n\nfollowup stuff"


def test_resolve_multi_tag_with_hierarchy():
    root = ConfidentConfidantConfig(note_prompts={"meeting": "root meeting"})
    leaf = ConfidentConfidantConfig(note_prompts={"meeting": "leaf meeting", "followup": "leaf followup"})
    result = resolve_prompt((root, leaf), ["meeting", "followup"])
    assert result == "root meeting\n\nleaf meeting\n\nleaf followup"


def test_resolve_fallback_to_builtin():
    """No configs define any prompts → fall back to DEFAULT_NOTE_PROMPT."""
    configs = (ConfidentConfidantConfig(),)
    assert resolve_prompt(configs, []) == DEFAULT_NOTE_PROMPT


def test_resolve_empty_configs():
    assert resolve_prompt((), []) == DEFAULT_NOTE_PROMPT


def test_resolve_tag_not_found_anywhere():
    """Tag requested but no config defines it → fall back to builtin."""
    configs = (ConfidentConfidantConfig(note_prompts={"default": "exists"}),)
    assert resolve_prompt(configs, ["nonexistent"]) == DEFAULT_NOTE_PROMPT


def test_read_config_walks_up_from_relative_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "cc-config.md").write_text("""\
# Confident Confidant Config

## Base Config
```hjson
transcription_model: parakeet-mlx/mlx-community/parakeet-tdt-0.6b-v3
```""")
    (tmp_path / "todo").mkdir()
    monkeypatch.chdir(tmp_path)

    config = read_config_from_directory_hierarchy(Path("todo/memo.m4a"))

    assert config.transcription_model == "parakeet-mlx/mlx-community/parakeet-tdt-0.6b-v3"
