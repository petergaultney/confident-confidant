"""CLI entry point for GPT-4o based transcribe-diarize."""

import argparse
from pathlib import Path
from cc.config import read_config_from_directory_hierarchy
from cc.transcribe.diarize.core import transcribe_audio_diarized


def cli() -> None:
    parser = argparse.ArgumentParser(
        prog="transcribe-diarize",
        description="Transcribe audio with GPT-4o speaker diarization.",
    )
    parser.add_argument("input", help="Input audio/video file", type=Path)
    args = parser.parse_args()

    config = read_config_from_directory_hierarchy(args.input)
    transcribe_audio_diarized(
        input_file=args.input,
        diarization_model=config.diarization_model,
        split_audio_approx_every_s=config.split_audio_approx_every_s,
    )


if __name__ == "__main__":
    cli()
