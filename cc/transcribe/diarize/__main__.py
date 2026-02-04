"""CLI entry point for GPT-4o based transcribe-diarize."""

import argparse
import shutil
import logging
from pathlib import Path
from cc.config import read_config_from_directory_hierarchy
from cc.transcribe.diarize.core import transcribe_audio_diarized

logger = logging.getLogger(__name__)


def cli() -> None:
    parser = argparse.ArgumentParser(
        prog="transcribe-diarize",
        description="Transcribe audio with GPT-4o speaker diarization.",
    )
    parser.add_argument("input", help="Input audio/video file", type=Path)
    parser.add_argument(
        "-o", "--out", help="Where does the output go? (provide a directory)", type=Path
    )
    args = parser.parse_args()

    config = read_config_from_directory_hierarchy(args.input)
    output = transcribe_audio_diarized(
        input_file=args.input,
        diarization_model=config.diarization_model,
        split_audio_approx_every_s=config.split_audio_approx_every_s,
    )

    dest: Path | None = args.out
    if dest:
        if dest.is_file():
            logger.warning(f"-o argument {dest} is a file; cannot output to it")
            return

        dest.mkdir(parents=True, exist_ok=True)
        for file in output:
            shutil.copy2(file, dest / file.name)


if __name__ == "__main__":
    cli()
