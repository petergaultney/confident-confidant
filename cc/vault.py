import difflib
import logging
import re
import typing as ty
import urllib
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path

logger = logging.getLogger(__name__)


@lru_cache
def _find_vault_root_recursive(any_path: Path) -> Path:
    # recursive
    """Find the root of the Vault by looking for the .obsidian or .git directory.

    This is really only used for deciding where to look for links to your audio files.

    If it resolves to the root of the filesystem, we'll ignore this and instead use
    whatever path you provided at the root of the script.
    """
    if (
        (any_path / ".obsidian").is_dir()
        or (any_path / ".loqseq").is_dir()
        or (any_path / ".root").is_dir()
        or (any_path / ".git").is_dir()
    ):
        return any_path

    if any_path.parent == any_path:
        return any_path

    return _find_vault_root_recursive(any_path.parent)


def find_vault_root(any_path: Path) -> Path:
    vault_root = _find_vault_root_recursive(any_path)
    if vault_root == vault_root.parent:
        logger.warning(
            f"Could not find a vault root for '{any_path}'. Using that path as the vault root."
        )
        return any_path
    logger.info(f"Found vault root: {vault_root}")
    return vault_root


# Maps filename stem (without extension) -> set of all paths with that stem
VaultIndex = dict[str, set[Path]]


def build_vault_index(vault_root: Path) -> VaultIndex:
    """Build a reverse index of all files in the vault: stem -> set of paths.
    Stem is used instead of filename because Obsidian allows you to reference
    files by stem (without extension)."""
    index: VaultIndex = defaultdict(set)
    for path in vault_root.rglob("*"):
        if path.is_file():
            index[path.stem].add(path)
    return index


@dataclass
class _Link:
    """Represents a Markdown or Obsidian link found in a note."""

    full_match: str
    target: Path
    is_embed: bool
    style: ty.Literal["obsidian", "markdown"]
    text: str | None = None


# A single regex to find both Obsidian and Markdown links.
# It uses named capture groups for easier parsing.
_LINK_PATTERN = re.compile(
    r"""
    (?P<obsidian>
        (?P<obs_embed>!)?\[\[
        (?P<obs_target>[^|\]]+)
        (?:\|(?P<obs_text>[^\]]+))?
        \]\]
    )
    |
    (?P<markdown>
        (?P<md_embed>!)?
        \[(?P<md_text>[^\]]*)\]
        \((?P<md_target>[^)]+)\)
    )
""",
    re.VERBOSE,
)


def _obsidian_link_matches_target(
    index: VaultIndex, link_target_str: str, expected_target: Path
) -> bool:
    """Check if an obsidian link target string matches the expected target.

    Obsidian hrefs are permissive: you can refer to a file with just its
    filename, with a relpath, with a _partial_ relpath, or with an absolute path.

    Raises ValueError if the link is ambiguous and expected_target is among candidates.
    """
    link_target_str = link_target_str.strip()

    # Extract the stem (filename without extension) from the target string.
    # For "subfolder/note" or "note.md" or "note", we want the final component's stem.
    target_path = Path(link_target_str)
    stem = target_path.stem

    candidates = index.get(stem, set())
    if not candidates:
        return False

    # Filter candidates based on path and/or extension in the link target
    if "/" in link_target_str:
        # Link has path components (e.g., "subfolder/note") - filter by path suffix
        pattern = "/" + link_target_str.lstrip("/")
        if target_path.suffix:
            # Link includes extension (e.g., "subfolder/note.md") - match exactly
            filtered = {p for p in candidates if str(p).endswith(pattern)}
        else:
            # Link has no extension (e.g., "subfolder/note") - match any extension
            filtered = {p for p in candidates if str(p.with_suffix("")).endswith(pattern)}
        candidates = filtered
    elif target_path.suffix:
        # No path but has extension (e.g., "note.md") - filter by extension
        filtered = {p for p in candidates if p.suffix == target_path.suffix}
        candidates = filtered

    if not candidates:
        return False

    if len(candidates) == 1:
        return expected_target in candidates

    # Multiple candidates - ambiguous link
    if expected_target in candidates:
        raise ValueError(
            f"Ambiguous obsidian link '{link_target_str}' could refer to multiple files "
            f"including the target '{expected_target}': {candidates}"
        )

    # Ambiguous but doesn't involve our target - ignore this link
    return False


def _markdown_link_matches_target(src_file: Path, link_target_str: str, expected_target: Path) -> bool:
    """Check if a markdown link target string (with path prefix) matches the expected target."""
    link_target_str = link_target_str.strip()
    if not any(link_target_str.startswith(p) for p in ("./", "../", "/", "file://")):
        return False

    link_target_str = link_target_str.removeprefix("file://")
    link_target_str = urllib.parse.unquote(link_target_str)
    # ^ you can link to files with url-encoding, and in fact this is the default
    # behavior for obsidian when there are spaces in the file name (it converts
    # ' ' to '%20').  we unquote it here in order to have a valid file name.

    target = Path(link_target_str)
    if target.is_absolute():
        resolved = target
    else:
        resolved = (src_file.parent / link_target_str).resolve()

    return resolved == expected_target


def _link_from_match_if_target(
    index: VaultIndex, match: re.Match, *, src_file: Path, expected_target: Path
) -> _Link | None:
    """Return a Link if the match points to expected_target, None otherwise.

    Raises ValueError if an obsidian link is ambiguous and expected_target is a candidate.
    """
    if match.group("obsidian"):
        obs_target_str = match.group("obs_target").strip()
        if _obsidian_link_matches_target(index, obs_target_str, expected_target):
            return _Link(
                full_match=match.group(0),
                target=expected_target,
                is_embed=bool(match.group("obs_embed")),
                style="obsidian",
                text=match.group("obs_text"),
            )

    if match.group("markdown"):
        md_target_str = match.group("md_target")
        if _markdown_link_matches_target(src_file, md_target_str, expected_target):
            return _Link(
                full_match=match.group(0),
                target=expected_target,
                is_embed=bool(match.group("md_embed")),
                style="markdown",
                text=match.group("md_text"),
            )

    return None


def _find_links_to_file(index: VaultIndex, *, in_md_file: Path, target_file: Path) -> list[_Link]:
    """Find all links in in_md_file that point to target_file."""
    content = in_md_file.read_text(encoding="utf-8")
    return [
        link
        for match in _LINK_PATTERN.finditer(content)
        if (
            link := _link_from_match_if_target(
                index, match, src_file=in_md_file, expected_target=target_file
            )
        )
    ]


@dataclass
class LinkContext:
    link: _Link
    line_text: str  # full line containing the link
    prev_line: str  # line immediately before the link (empty if first line)
    context: str  # surrounding text with link, tags, and list markers stripped


def _clean_context(raw: str) -> str:
    return raw.replace("#diarize", "").strip().lstrip("- ").rstrip(":").strip()


def find_link_context(
    index: VaultIndex, *, in_md_file: Path, target_file: Path
) -> list[LinkContext]:
    """Find links to target_file and extract surrounding text as context.

    Grabs both same-line text and the previous line (for the common pattern
    where context sits on the line above an embed).
    """
    content = in_md_file.read_text(encoding="utf-8")
    lines = content.splitlines()
    results: list[LinkContext] = []

    for match in _LINK_PATTERN.finditer(content):
        link = _link_from_match_if_target(
            index, match, src_file=in_md_file, expected_target=target_file
        )
        if not link:
            continue

        context_parts: list[str] = []
        prev_line = ""
        for i, line in enumerate(lines):
            if link.full_match not in line:
                continue

            if i > 0:
                prev_line = lines[i - 1]
                prev_ctx = _clean_context(prev_line)
                if prev_ctx and not _LINK_PATTERN.search(prev_line):
                    context_parts.append(prev_ctx)

            same_line = _clean_context(line.replace(link.full_match, ""))
            if same_line:
                context_parts.append(same_line)
            break

        results.append(LinkContext(
            link=link, line_text=line, prev_line=prev_line,
            context=" ".join(context_parts),
        ))

    return results


def link_line_has_tag(
    index: VaultIndex, *, in_md_file: Path, target_file: Path, tag: str
) -> bool:
    """Check if any link to target_file has the given tag on or near the same line."""
    return any(
        tag in ctx.line_text or tag in ctx.prev_line
        for ctx in find_link_context(index, in_md_file=in_md_file, target_file=target_file)
    )


def find_linking_notes(index: VaultIndex, vault_root: Path, audio_path: Path) -> list[Path]:
    """Find all text notes that link to the given audio file."""
    logger.info(f"Looking for notes linking to: {audio_path.name}")
    linking_notes = []
    for note_path in vault_root.rglob("*.md"):
        try:
            links = _find_links_to_file(index=index, in_md_file=note_path, target_file=audio_path)
            if links:
                linking_notes.append(note_path)
                logger.info(f"Found linking note: {note_path}")
        except Exception as e:
            logger.warning(f"Could not read {note_path}: {e}")
    return linking_notes


def _print_diff(diff: ty.Iterable[str]) -> None:
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "cyan": "\033[96m",
        "endc": "\033[0m",
    }
    for line in diff:
        if line.startswith("+"):
            print(f"{colors['green']}{line}{colors['endc']}", end="")
        elif line.startswith("-"):
            print(f"{colors['red']}{line}{colors['endc']}", end="")
        elif line.startswith("@@"):
            print(f"{colors['cyan']}{line}{colors['endc']}", end="")
        else:
            print(line, end="")


def replace_links_in_notes(
    index: VaultIndex,
    vault_root: Path,
    linking_notes: list[Path],
    old_filepath: Path,
    transcript_note_path: Path,
    transcript_title: str,
    dry_run: bool = False,
) -> None:
    """Replace audio links with transcript links, preserving link style."""
    new_link_target = transcript_note_path.relative_to(vault_root).as_posix()

    def replacer(linking_note_path: Path, match: re.Match) -> str:
        """This function is called by re.sub for each link found."""
        link = _link_from_match_if_target(
            index, match, src_file=linking_note_path, expected_target=old_filepath
        )
        if not link:
            return match.group(0)

        # Build the new link in the same style as the original.
        if link.style == "obsidian":
            # Note: We are intentionally converting audio embeds `![[...]]`
            # to normal note links `[[...]]`.
            return f"[[{new_link_target}|{transcript_title}]]"
        elif link.style == "markdown":
            return f"[{transcript_title}]({new_link_target})"

        return link.full_match  # Failsafe

    for note_path in linking_notes:
        try:
            content = note_path.read_text(encoding="utf-8")
            new_content = _LINK_PATTERN.sub(partial(replacer, note_path), content)

            if content != new_content:
                logger.info(f"Links in {note_path} need updating:")
                _print_diff(
                    difflib.unified_diff(
                        content.splitlines(keepends=True),
                        new_content.splitlines(keepends=True),
                        fromfile=f"{note_path.name} (original)",
                        tofile=f"{note_path.name} (modified)",
                    )
                )
                if not dry_run:
                    note_path.write_text(new_content, encoding="utf-8")
                    logger.info(f"Updated links in: {note_path}")
                else:
                    logger.info(f"DRY RUN: Would update links in: {note_path}")
            else:
                logger.warning(
                    f"No changes made to {note_path}, though it was identified as a linking note."
                )
        except Exception as e:
            logger.exception(f"Failed to update links in {note_path}: {e}")
