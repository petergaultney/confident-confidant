import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def create_transcript_note(
    vault_root: Path,
    new_audio_path: Path,
    transcript_note_path: Path,
    title: str,
    prompt_response: str,
    file_hash: str,
) -> None:
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
