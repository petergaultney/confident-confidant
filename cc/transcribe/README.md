# cc.transcribe

Transcribe audio files using OpenAI's transcription API, or on-device with NVIDIA
Parakeet. Long files are split on silence boundaries, transcribed chunk by chunk,
and stitched back together.

## Requirements

- `ffmpeg` (`brew install ffmpeg`)
- OpenAI API key (set `OPENAI_API_KEY` or place in `~/.keys/openai-api`), unless
  you transcribe on-device

## On-device transcription

Set `transcription_model` to `parakeet-mlx/` followed by a Hugging Face repo, e.g.
`parakeet-mlx/mlx-community/parakeet-tdt-0.6b-v3`. We then transcribe with
[parakeet-mlx](https://github.com/senstella/parakeet-mlx) instead of calling OpenAI.
v3 covers English and 24 other European languages, and it runs only on Apple
Silicon.

This needs the `parakeet` extra:

```bash
uv tool install -e '.[parakeet]'
```

The first run downloads the model weights (~1.2 GB) into the Hugging Face cache.

Parakeet takes no prompt, so we do not use `transcription_context` when
transcribing on-device. Long files still go through the `reformat_model` stitch
step below.

## Usage

```bash
transcribe path/to/recording.m4a
```

The pipeline is also available as a library function:

```python
from cc.transcribe import transcribe_audio_file

transcript_path = transcribe_audio_file(Path("recording.m4a"))
```

When invoked via the `transcribe` CLI, configuration is read from the Markdown
vault hierarchy (see the root README for config format). When called as a
library, parameters are passed directly.

## Pipeline

1. **Extract audio** — Strips the video track (if any) and copies the audio to
   `audio.m4a` using ffmpeg.
2. **Split** (long files only) — If the audio duration exceeds
   `split_audio_approx_every_s + 90s` (default ~21.5 minutes), ffmpeg's
   `silencedetect` finds silent intervals (noise < -35 dB, duration >= 0.4s).
   Cut points are chosen at the silence midpoint nearest each target interval,
   within a 90-second window. The file is then segmented at those points. Silent
   chunks are discarded. Short files skip this step entirely.
3. **Transcribe** — Each chunk is sent to the OpenAI transcription API in
   parallel (2 workers), or transcribed on-device one at a time with Parakeet.
   Individual results are saved as JSON for troubleshooting.
4. **Stitch** — Chunk transcripts are concatenated in order. If there was only
   one chunk, the text is used as-is. Otherwise, an LLM (`reformat_model`,
   default `anthropic/claude-haiku-5-5`) cleans up capitalization and punctuation at fragment
   boundaries and inserts paragraph breaks.

All pipeline steps are memoized (`thds.mops.pure.magic`), so re-running the
same command skips already-completed work.

## Output

Output is written to `.out/transcribe/<filename>/<content-hash>/`:

- `audio.m4a` — Extracted audio track
- `silence.log` — ffmpeg silence detection output (long files only)
- `chunks/` — Split audio files (long files only)
- `chunk-transcripts/` — Individual chunk transcripts as JSON
- `transcript.raw.txt` — Joined chunk text before reformatting (long files only)
- `transcript.txt` — Final transcript

## Configuration

These fields from the `coco` config are used by the transcribe pipeline:

| Key | Default | Description |
|-----|---------|-------------|
| `transcription_model` | `gpt-4o-transcribe` | OpenAI transcription model, or `parakeet-mlx/<hf-repo>` for [on-device](#on-device-transcription) |
| `transcription_prompt` | `""` | Optional prompt to guide transcription |
| `reformat_model` | `anthropic/claude-haiku-5-5` | LLM used to clean up stitched transcripts |
| `split_audio_approx_every_s` | `1200` (20 min) | Target chunk duration in seconds |

## Speaker Diarization

For multi-speaker audio, see [diarize/README.md](diarize/README.md) for speaker
identification support.
