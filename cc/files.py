import hashlib
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def hash_file(file_path: Path) -> str:
    """Generate SHA-256 hash of file contents."""
    logger.info(f"Hashing file: {file_path}")
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _sanitize_title(title: str) -> str:
    """Convert title to filesystem-safe filename."""
    # Replace spaces with dashes, remove unsafe characters
    sanitized = re.sub(r"[^\w\s-]", "", title)
    sanitized = re.sub(r"\s+", "-", sanitized)
    sanitized = sanitized.strip("-").lower()
    return sanitized


def generate_new_filename(datetime_fmt: str, input_path: Path, title: str) -> str:
    """Generate new filename with input_file creation timestamp and title."""
    # Get the creation time of the audio file
    creation_time = datetime.fromtimestamp(input_path.stat().st_ctime)
    timestamp = creation_time.strftime(datetime_fmt)
    return f"{timestamp}_{_sanitize_title(title)}"


def create_unique_file_path(original_path: Path, target_dir: Path, new_filename: str) -> Path:
    new_path = target_dir / f"{new_filename}{original_path.suffix}"

    # Handle filename conflicts
    counter = 1
    while new_path.exists():
        new_path = target_dir / f"{new_filename}-{counter}{original_path.suffix}"
        counter += 1

    return new_path


def copy_file(original_path: Path, new_path: Path, dry_run: bool = True) -> None:
    if original_path.resolve() == new_path.resolve():
        logger.info(f"Skipping copy to same location: {original_path}")
        return
    if not dry_run:
        logger.info(f"Copying audio file: {original_path} -> {new_path}")
        new_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original_path, new_path)
    else:
        logger.info(f"DRY RUN: Would copy {original_path} -> {new_path}")
