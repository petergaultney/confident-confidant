from pathlib import Path

import pytest

from cc.vault import (
    VaultIndex,
    _find_links_to_file,
    _Link,
    _clean_context,
    extract_prompt_tags,
    link_line_has_tag,
    _markdown_link_matches_target,
    _obsidian_link_matches_target,
    build_vault_index,
    find_link_context,
)


class Test_build_vault_index:
    """Tests for build_vault_index function."""

    def test_empty_vault(self, tmp_path: Path) -> None:
        """Empty directory produces empty index."""
        index = build_vault_index(tmp_path)
        assert index == {}

    def test_single_file(self, tmp_path: Path) -> None:
        """Single file is indexed by its stem."""
        (tmp_path / "note.md").write_text("content")
        index = build_vault_index(tmp_path)
        assert index == {"note": {tmp_path / "note.md"}}

    def test_multiple_files_different_stems(self, tmp_path: Path) -> None:
        """Multiple files with different stems are indexed separately."""
        (tmp_path / "note1.md").write_text("content1")
        (tmp_path / "note2.md").write_text("content2")
        index = build_vault_index(tmp_path)
        assert index == {
            "note1": {tmp_path / "note1.md"},
            "note2": {tmp_path / "note2.md"},
        }

    def test_same_stem_different_extensions(self, tmp_path: Path) -> None:
        """Files with same stem but different extensions are grouped."""
        (tmp_path / "recording.m4a").touch()
        (tmp_path / "recording.md").touch()
        index = build_vault_index(tmp_path)
        assert index["recording"] == {
            tmp_path / "recording.m4a",
            tmp_path / "recording.md",
        }

    def test_nested_directories(self, tmp_path: Path) -> None:
        """Files in nested directories are indexed."""
        (tmp_path / "a" / "b").mkdir(parents=True)
        (tmp_path / "a" / "note.md").touch()
        (tmp_path / "a" / "b" / "other.md").touch()
        index = build_vault_index(tmp_path)
        assert index == {
            "note": {tmp_path / "a" / "note.md"},
            "other": {tmp_path / "a" / "b" / "other.md"},
        }

    def test_duplicate_stems_in_different_directories(self, tmp_path: Path) -> None:
        """Same filename in different directories creates multiple entries."""
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        (tmp_path / "a" / "note.md").touch()
        (tmp_path / "b" / "note.md").touch()
        index = build_vault_index(tmp_path)
        assert index["note"] == {
            tmp_path / "a" / "note.md",
            tmp_path / "b" / "note.md",
        }

    def test_ignores_directories(self, tmp_path: Path) -> None:
        """Directories are not indexed, only files."""
        (tmp_path / "subdir").mkdir()
        (tmp_path / "file.md").touch()
        index = build_vault_index(tmp_path)
        assert "subdir" not in index


class Test_obsidian_link_matches_target:
    """Tests for _obsidian_link_matches_target function."""

    @pytest.fixture
    def index_with_audio_file(self, tmp_path: Path) -> tuple[VaultIndex, Path]:
        """Create a simple vault with one unique file at recordings/Recording 123.m4a."""
        audio = tmp_path / "recordings" / "Recording 123.m4a"
        audio.parent.mkdir(parents=True)
        audio.touch()
        index = build_vault_index(tmp_path)
        return index, audio

    @pytest.mark.parametrize(
        ("link_target_str", "expected"),
        [
            pytest.param("Recording 123", True, id="matches by stem only"),
            pytest.param("Recording 123.m4a", True, id="matches by stem with extension"),
            pytest.param("recordings/Recording 123", True, id="matches with partial path"),
            pytest.param(
                "recordings/Recording 123.m4a", True, id="matches with partial path and extension"
            ),
            pytest.param("  Recording 123  ", True, id="strips whitespace"),
            pytest.param("/recordings/Recording 123", True, id="handles leading slash"),
            pytest.param("Recording 456", False, id="no match wrong stem"),
            pytest.param("other/Recording 123", False, id="no match wrong path"),
            pytest.param("nonexistent", False, id="no match not in index"),
        ],
    )
    def test_link_matching(
        self,
        index_with_audio_file: tuple[VaultIndex, Path],
        link_target_str: str,
        expected: bool,
    ) -> None:
        """Test various link target strings against the audio file."""
        index, audio = index_with_audio_file
        assert _obsidian_link_matches_target(index, link_target_str, audio) == expected

    def test_ambiguous_link_raises_when_target_involved(self, tmp_path: Path) -> None:
        """Ambiguous link raises ValueError when target is among candidates."""
        audio_a = tmp_path / "a" / "recording.m4a"
        audio_b = tmp_path / "b" / "recording.m4a"
        for p in (audio_a, audio_b):
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()
        index = build_vault_index(tmp_path)

        with pytest.raises(ValueError, match="Ambiguous obsidian link"):
            _obsidian_link_matches_target(index, "recording", audio_a)

    def test_ambiguous_link_returns_false_when_target_not_involved(self, tmp_path: Path) -> None:
        """Ambiguous link returns False when target is not among candidates."""
        audio_a = tmp_path / "a" / "recording.m4a"
        audio_b = tmp_path / "b" / "recording.m4a"
        other = tmp_path / "other.m4a"
        for p in (audio_a, audio_b, other):
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()
        index = build_vault_index(tmp_path)

        # _Link is ambiguous between a/recording and b/recording, but target is "other"
        assert not _obsidian_link_matches_target(index, "recording", other)

    def test_partial_path_disambiguates(self, tmp_path: Path) -> None:
        """Partial path can disambiguate files with same stem."""
        audio_a = tmp_path / "a" / "recording.m4a"
        audio_b = tmp_path / "b" / "recording.m4a"
        for p in (audio_a, audio_b):
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()
        index = build_vault_index(tmp_path)

        assert _obsidian_link_matches_target(index, "a/recording", audio_a)
        assert not _obsidian_link_matches_target(index, "a/recording", audio_b)

    def test_extension_in_link_filters_candidates(self, tmp_path: Path) -> None:
        """Extension in link filters to exact match."""
        audio = tmp_path / "recording.m4a"
        note = tmp_path / "recording.md"
        for p in (audio, note):
            p.touch()
        index = build_vault_index(tmp_path)

        for link_target_str, target_path, expected in (
            ("recording.m4a", audio, True),
            ("recording.m4a", note, False),
            ("recording.md", audio, False),
            ("recording.md", note, True),
        ):
            assert _obsidian_link_matches_target(index, link_target_str, target_path) is expected


class Test_markdown_link_matches_target:
    """Tests for _markdown_link_matches_target function."""

    def test_relative_path_matches(self, tmp_path: Path) -> None:
        """Relative path with ./ prefix matches."""
        src = tmp_path / "notes" / "daily.md"
        target = tmp_path / "notes" / "recording.m4a"
        for p in (src, target):
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()

        assert _markdown_link_matches_target(src, "./recording.m4a", target)

    def test_relative_path_parent_dir(self, tmp_path: Path) -> None:
        """Relative path with ../ matches."""
        src = tmp_path / "notes" / "sub" / "daily.md"
        target = tmp_path / "notes" / "recording.m4a"
        for p in (src, target):
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()

        assert _markdown_link_matches_target(src, "../recording.m4a", target)

    def test_absolute_path_matches(self, tmp_path: Path) -> None:
        """Absolute path matches."""
        src = tmp_path / "notes" / "daily.md"
        target = tmp_path / "recordings" / "audio.m4a"
        for p in (src, target):
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()

        assert _markdown_link_matches_target(src, str(target), target)

    def test_file_uri_matches(self, tmp_path: Path) -> None:
        """file:// URI matches."""
        src = tmp_path / "daily.md"
        target = tmp_path / "recording.m4a"
        for p in (src, target):
            p.touch()

        assert _markdown_link_matches_target(src, f"file://{target}", target)

    def test_url_encoded_path(self, tmp_path: Path) -> None:
        """URL-encoded paths are decoded."""
        src = tmp_path / "daily.md"
        target = tmp_path / "my recording.m4a"
        for p in (src, target):
            p.touch()

        assert _markdown_link_matches_target(src, "./my%20recording.m4a", target)

    def test_no_match_without_path_prefix(self, tmp_path: Path) -> None:
        """Link without path prefix (./, ../, /, file://) doesn't match."""
        src = tmp_path / "daily.md"
        target = tmp_path / "recording.m4a"
        for p in (src, target):
            p.touch()

        # Plain filename without prefix - not a markdown file link
        assert not _markdown_link_matches_target(src, "recording.m4a", target)

    def test_no_match_wrong_target(self, tmp_path: Path) -> None:
        """Link to different file doesn't match."""
        src = tmp_path / "daily.md"
        target = tmp_path / "recording.m4a"
        other = tmp_path / "other.m4a"
        for p in (src, target, other):
            p.touch()

        assert not _markdown_link_matches_target(src, "./other.m4a", target)


class Test_find_links_to_file:
    """Tests for _find_links_to_file function."""

    def test_finds_obsidian_link(self, tmp_path: Path) -> None:
        """Finds obsidian-style link to target."""
        target = tmp_path / "recording.m4a"
        note = tmp_path / "daily.md"
        target.touch()
        note.write_text("Check out this recording: [[recording]]")
        index = build_vault_index(tmp_path)

        links = _find_links_to_file(index=index, in_md_file=note, target_file=target)
        assert links == [
            _Link(full_match="[[recording]]", target=target, is_embed=False, style="obsidian", text=None)
        ]

    def test_finds_obsidian_embed(self, tmp_path: Path) -> None:
        """Finds obsidian embed to target."""
        target = tmp_path / "recording.m4a"
        note = tmp_path / "daily.md"
        target.touch()
        note.write_text("![[recording.m4a]]")
        index = build_vault_index(tmp_path)

        links = _find_links_to_file(index=index, in_md_file=note, target_file=target)
        assert links == [
            _Link(
                full_match="![[recording.m4a]]",
                target=target,
                is_embed=True,
                style="obsidian",
                text=None,
            )
        ]

    def test_finds_markdown_link(self, tmp_path: Path) -> None:
        """Finds markdown-style link to target."""
        target = tmp_path / "recording.m4a"
        note = tmp_path / "daily.md"
        target.touch()
        note.write_text("Check out [my recording](./recording.m4a)")
        index = build_vault_index(tmp_path)

        links = _find_links_to_file(index=index, in_md_file=note, target_file=target)
        assert links == [
            _Link(
                full_match="[my recording](./recording.m4a)",
                target=target,
                is_embed=False,
                style="markdown",
                text="my recording",
            )
        ]

    def test_finds_multiple_links(self, tmp_path: Path) -> None:
        """Finds multiple links to same target."""
        target = tmp_path / "recording.m4a"
        note = tmp_path / "daily.md"
        target.touch()
        note.write_text("First: [[recording]] and second: [[recording.m4a]]")
        index = build_vault_index(tmp_path)

        links = _find_links_to_file(index=index, in_md_file=note, target_file=target)
        assert links == [
            _Link(
                full_match="[[recording]]",
                target=target,
                is_embed=False,
                style="obsidian",
                text=None,
            ),
            _Link(
                full_match="[[recording.m4a]]",
                target=target,
                is_embed=False,
                style="obsidian",
                text=None,
            ),
        ]

    def test_ignores_links_to_other_files(self, tmp_path: Path) -> None:
        """Only returns links to the specified target."""
        target = tmp_path / "recording.m4a"
        other = tmp_path / "other.m4a"
        note = tmp_path / "daily.md"
        target.touch()
        other.touch()
        note.write_text("Target: [[recording]] Other: [[other]]")
        index = build_vault_index(tmp_path)

        links = _find_links_to_file(index=index, in_md_file=note, target_file=target)
        assert links == [
            _Link(full_match="[[recording]]", target=target, is_embed=False, style="obsidian", text=None)
        ]

    def test_returns_empty_for_no_matches(self, tmp_path: Path) -> None:
        """Returns empty list when no links match target."""
        target = tmp_path / "recording.m4a"
        note = tmp_path / "daily.md"
        target.touch()
        note.write_text("No links here, just text.")
        index = build_vault_index(tmp_path)

        links = _find_links_to_file(index=index, in_md_file=note, target_file=target)
        assert links == []

    def test_preserves_link_text(self, tmp_path: Path) -> None:
        """Preserves display text from links."""
        target = tmp_path / "recording.m4a"
        note = tmp_path / "daily.md"
        target.touch()
        note.write_text("[[recording|My Audio Note]]")
        index = build_vault_index(tmp_path)

        links = _find_links_to_file(index=index, in_md_file=note, target_file=target)
        assert links == [
            _Link(
                full_match="[[recording|My Audio Note]]",
                target=target,
                is_embed=False,
                style="obsidian",
                text="My Audio Note",
            )
        ]


def test_link_context_same_line(tmp_path: Path) -> None:
    target = tmp_path / "recording.m4a"
    note = tmp_path / "daily.md"
    target.touch()
    note.write_text(" - meeting with Grant: ![[recording.m4a]]")
    index = build_vault_index(tmp_path)

    contexts = find_link_context(index, in_md_file=note, target_file=target)
    assert len(contexts) == 1
    assert contexts[0].context == "meeting with Grant"


def test_link_context_previous_line(tmp_path: Path) -> None:
    target = tmp_path / "recording.m4a"
    note = tmp_path / "daily.md"
    target.touch()
    note.write_text(" - meeting with Grant about policy stuff:\n![[recording.m4a]]")
    index = build_vault_index(tmp_path)

    contexts = find_link_context(index, in_md_file=note, target_file=target)
    assert len(contexts) == 1
    assert contexts[0].context == "meeting with Grant about policy stuff"


def test_link_context_both_lines(tmp_path: Path) -> None:
    target = tmp_path / "recording.m4a"
    note = tmp_path / "daily.md"
    target.touch()
    note.write_text(" - meeting with Grant:\nextra context ![[recording.m4a]]")
    index = build_vault_index(tmp_path)

    contexts = find_link_context(index, in_md_file=note, target_file=target)
    assert len(contexts) == 1
    assert "meeting with Grant" in contexts[0].context
    assert "extra context" in contexts[0].context


def test_link_context_strips_diarize_tag(tmp_path: Path) -> None:
    target = tmp_path / "recording.m4a"
    note = tmp_path / "daily.md"
    target.touch()
    note.write_text(" - meeting with Grant: #diarize ![[recording.m4a]]")
    index = build_vault_index(tmp_path)

    contexts = find_link_context(index, in_md_file=note, target_file=target)
    assert len(contexts) == 1
    assert "#diarize" not in contexts[0].context
    assert "meeting with Grant" in contexts[0].context


def test_link_context_no_context(tmp_path: Path) -> None:
    target = tmp_path / "recording.m4a"
    note = tmp_path / "daily.md"
    target.touch()
    note.write_text("![[recording.m4a]]")
    index = build_vault_index(tmp_path)

    contexts = find_link_context(index, in_md_file=note, target_file=target)
    assert len(contexts) == 1
    assert contexts[0].context == ""


def test_link_context_no_links(tmp_path: Path) -> None:
    target = tmp_path / "recording.m4a"
    note = tmp_path / "daily.md"
    target.touch()
    note.write_text("no links here")
    index = build_vault_index(tmp_path)

    assert find_link_context(index, in_md_file=note, target_file=target) == []


def test_link_tag_on_link_line(tmp_path: Path) -> None:
    target = tmp_path / "recording.m4a"
    note = tmp_path / "daily.md"
    target.touch()
    note.write_text(" - meeting: #diarize ![[recording.m4a]]")
    index = build_vault_index(tmp_path)

    assert link_line_has_tag(index, in_md_file=note, target_file=target, tag="#diarize")


def test_link_tag_on_previous_line(tmp_path: Path) -> None:
    target = tmp_path / "recording.m4a"
    note = tmp_path / "daily.md"
    target.touch()
    note.write_text(" - meeting: #diarize\n![[recording.m4a]]")
    index = build_vault_index(tmp_path)

    assert link_line_has_tag(index, in_md_file=note, target_file=target, tag="#diarize")


def test_link_tag_absent(tmp_path: Path) -> None:
    target = tmp_path / "recording.m4a"
    note = tmp_path / "daily.md"
    target.touch()
    note.write_text(" - meeting with Grant:\n![[recording.m4a]]")
    index = build_vault_index(tmp_path)

    assert not link_line_has_tag(index, in_md_file=note, target_file=target, tag="#diarize")


# extract_prompt_tags tests


def test_extract_prompt_tags_single_tag(tmp_path: Path) -> None:
    target = tmp_path / "recording.m4a"
    note = tmp_path / "daily.md"
    target.touch()
    note.write_text("![[recording.m4a]] #meeting")
    index = build_vault_index(tmp_path)

    assert extract_prompt_tags(index, in_md_file=note, target_file=target) == ["meeting"]


def test_extract_prompt_tags_multiple_tags(tmp_path: Path) -> None:
    target = tmp_path / "recording.m4a"
    note = tmp_path / "daily.md"
    target.touch()
    note.write_text("![[recording.m4a]] #meeting #followup")
    index = build_vault_index(tmp_path)

    assert extract_prompt_tags(index, in_md_file=note, target_file=target) == ["meeting", "followup"]


def test_extract_prompt_tags_ignores_meta_diarize(tmp_path: Path) -> None:
    target = tmp_path / "recording.m4a"
    note = tmp_path / "daily.md"
    target.touch()
    note.write_text("![[recording.m4a]] #diarize #meeting")
    index = build_vault_index(tmp_path)

    assert extract_prompt_tags(index, in_md_file=note, target_file=target) == ["meeting"]


def test_extract_prompt_tags_no_tags(tmp_path: Path) -> None:
    target = tmp_path / "recording.m4a"
    note = tmp_path / "daily.md"
    target.touch()
    note.write_text("![[recording.m4a]]")
    index = build_vault_index(tmp_path)

    assert extract_prompt_tags(index, in_md_file=note, target_file=target) == []


def test_extract_prompt_tags_from_prev_line(tmp_path: Path) -> None:
    target = tmp_path / "recording.m4a"
    note = tmp_path / "daily.md"
    target.touch()
    note.write_text("meeting with Grant #meeting\n![[recording.m4a]]")
    index = build_vault_index(tmp_path)

    assert extract_prompt_tags(index, in_md_file=note, target_file=target) == ["meeting"]


def test_clean_context_strips_all_tags() -> None:
    assert _clean_context("meeting with Grant #meeting #followup") == "meeting with Grant"


def test_clean_context_strips_diarize_tag() -> None:
    assert _clean_context("something #diarize") == "something"


def test_link_tag_far_away_not_detected(tmp_path: Path) -> None:
    target = tmp_path / "recording.m4a"
    note = tmp_path / "daily.md"
    target.touch()
    note.write_text("#diarize\n\nsome other stuff\n![[recording.m4a]]")
    index = build_vault_index(tmp_path)

    assert not link_line_has_tag(index, in_md_file=note, target_file=target, tag="#diarize")
