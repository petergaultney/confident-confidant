# 2.0.0

- Support transcribing audio files larger than 25 MB by splitting on silence,
  transcribing chunks in parallel, and stitching the results back together.
- Add speaker diarization pipeline (`transcribe-diarize`) using GPT-4o, with a
  labeling step (`transcribe-label`) for mapping chunk-local speaker IDs to names.
- New standalone CLI entry points: `transcribe`, `transcribe-diarize`, `transcribe-label`.
- New config options: `reformat_model`, `split_audio_approx_every_s`, `diarization_model`.
- New dependency: `ffmpeg` (install via `brew install ffmpeg`).
- Python version requirement bumped to 3.13+.

# 1.0.0

- Modularizes and librarifies `confident-confidant`

## 0.3.0

- Move default directories to be under a `cc` directory. Rename `transcripts` directory to `notes`.
- Expose the datetime format as a configurable option, `datetime_fmt`.

## 0.2.1

- `--no-mutate` now avoids making a copy of the audio file, so that things are easier to
  clean up - this potentially hints at a different workflow for people who don't want
  their audio files moved in the first place?

# 0.2.0

- Fully (I think?) support non-Obsidian (e.g., Logseq, folders full of Markdown notes -- whatever, really!) Markdown vaults.

# 0.1.0

- Initial release
