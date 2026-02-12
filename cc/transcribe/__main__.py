"""Main CLI entry point for transcribe-long."""

from pathlib import Path
import shutil
import argparse
from cc.config import read_config_from_directory_hierarchy
from cc.transcribe.core import transcribe_audio_file


def cli() -> None:
    parser = argparse.ArgumentParser(
        prog="transcribe",
        description="Transcribe large audio files by splitting, transcribing chunks, and stitching.",
    )
    parser.add_argument("input", help="Input audio/video file", type=Path)
    parser.add_argument("-o", "--out", help="Where does the transcript go?", type=Path)
    args = parser.parse_args()

    config = read_config_from_directory_hierarchy(args.input)
    output = transcribe_audio_file(
        input_file=args.input,
        transcription_model=config.transcription_model,
        transcription_context=config.transcription_context,
        reformat_model=config.reformat_model,
        split_audio_approx_every_s=config.split_audio_approx_every_s,
    )

    if args.out:
        shutil.copy2(output, args.out)


if __name__ == "__main__":
    cli()
