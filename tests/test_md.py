from cc.md import extract_headings_by_prefix


def test_bare_note_prompt_maps_to_default():
    md = "## Note Prompt\nsome prompt content"
    assert extract_headings_by_prefix(md, "Note Prompt") == {"default": "some prompt content"}


def test_named_note_prompt():
    md = "## Note Prompt: meeting\nmeeting instructions here"
    assert extract_headings_by_prefix(md, "Note Prompt") == {"meeting": "meeting instructions here"}


def test_multiple_named_prompts():
    md = """\
## Note Prompt
default stuff

## Note Prompt: meeting
meeting stuff

## Note Prompt: standup
standup stuff"""
    result = extract_headings_by_prefix(md, "Note Prompt")
    assert result == {
        "default": "default stuff",
        "meeting": "meeting stuff",
        "standup": "standup stuff",
    }


def test_heading_inside_code_block_ignored():
    md = """\
## Note Prompt: real
real content

```
## Note Prompt: fake
not a real heading
```

## Other Heading
unrelated"""
    result = extract_headings_by_prefix(md, "Note Prompt")
    assert "fake" not in result
    assert result["real"] == "real content\n\n```\n## Note Prompt: fake\nnot a real heading\n```"


def test_no_matching_headings():
    md = "## Something Else\ncontent"
    assert extract_headings_by_prefix(md, "Note Prompt") == {}


def test_colon_in_name_preserved():
    md = "## Note Prompt: my special: prompt\ncontent here"
    result = extract_headings_by_prefix(md, "Note Prompt")
    assert result == {"my special: prompt": "content here"}


def test_stops_at_same_level_heading():
    md = """\
## Note Prompt: meeting
meeting content
## Base Config
other stuff"""
    result = extract_headings_by_prefix(md, "Note Prompt")
    assert result == {"meeting": "meeting content"}


def test_different_heading_levels():
    md = """\
### Note Prompt
deep content
### Note Prompt: meeting
meeting at level 3"""
    result = extract_headings_by_prefix(md, "Note Prompt")
    assert result == {"default": "deep content", "meeting": "meeting at level 3"}
