"""E2E: coco (cc.__main__.process_audio_file / process_vault_recordings)

Sets up a minimal vault with an audio file linked from a note, then runs the
full pipeline: transcribe → summarize → create note → rewrite links.
"""

import shutil
from pathlib import Path

import pytest

from cc.__main__ import process_audio_file, process_vault_recordings
from cc.vault import build_vault_index

from tests.e2e.conftest import CHEAP_CONFIG

pytestmark = pytest.mark.e2e


def _setup_vault(tmp_path: Path, audio_src: Path) -> tuple[Path, Path]:
    """Create a minimal vault:

        vault_root/
          .root/            ← vault marker
          journal/
            Recording 123.m4a
            2025-01-29.md   ← links to the recording

    Returns (vault_root, audio_path).
    """
    vault_root = tmp_path / "vault"
    (vault_root / ".root").mkdir(parents=True)
    journal = vault_root / "journal"
    journal.mkdir()

    # Copy real audio into the vault
    audio_path = journal / "Recording 123.m4a"
    shutil.copy2(audio_src, audio_path)

    # Create a note that embeds the recording (Obsidian style)
    note = journal / "2025-01-29.md"
    note.write_text("# Wednesday\n\n![[Recording 123.m4a]]\n", encoding="utf-8")

    # Write a cc-config that uses cheap models
    config_md = vault_root / "cc-config.md"
    config_md.write_text(
        f"""\
## Confident Confidant Config

### Base Config

```hjson
transcription_model: {CHEAP_CONFIG.transcription_model}
reformat_model: {CHEAP_CONFIG.reformat_model}
note_model: {CHEAP_CONFIG.note_model}
```
""",
        encoding="utf-8",
    )

    return vault_root, audio_path


def test_process_audio_file(short_audio: Path, tmp_path: Path):
    """Run the full coco pipeline on a single audio file inside a vault."""
    vault_root, audio_path = _setup_vault(tmp_path, short_audio)
    index = build_vault_index(vault_root)

    result = process_audio_file(index, vault_root, dry_run=False, audio_path=audio_path)

    assert result is not None, "process_audio_file should return the transcript note path"
    assert result.exists()
    note_text = result.read_text(encoding="utf-8")
    assert note_text.startswith("# ")
    assert "sha256:" in note_text

    # The original audio should have been deleted (not dry_run)
    assert not audio_path.exists(), "Original audio should be removed after processing"

    # The linking note should have been updated
    linking_note = vault_root / "journal" / "2025-01-29.md"
    updated = linking_note.read_text(encoding="utf-8")
    assert "Recording 123" not in updated, "Link should have been replaced"


def test_process_vault_recordings_dry_run(short_audio: Path, tmp_path: Path):
    """Dry-run mode should process but not move files or rewrite links."""
    vault_root, audio_path = _setup_vault(tmp_path, short_audio)

    process_vault_recordings(vault_root / "journal", dry_run=True)

    # Audio file should still be in place
    assert audio_path.exists(), "Dry-run should not delete audio"
