# cc.transcribe.diarize

Transcribe audio with speaker diarization using OpenAI's GPT-4o diarization
model, then label speakers with human-readable names.

Speaker identity is determined per-chunk (OpenAI labels speakers `A`, `B`, `C`,
etc. independently for each chunk), so the same person may be `CHUNK_0_A` in one
chunk and `CHUNK_1_B` in another. The workflow below handles this with a manual
labeling step.

## Workflow

### 1. Transcribe with diarization

```bash
transcribe-diarize path/to/meeting.m4a
```

This runs the same split pipeline as `cc.transcribe` (extract audio, detect
silence, split on silence boundaries — see [../README.md](../README.md)), then
transcribes each chunk with diarization enabled
(`response_format="diarized_json"`).

The formatter merges consecutive same-speaker segments within each chunk into
single blocks and inserts `--- CHUNK_N ---` separators between chunks.

Output is written to `.out/transcribe-gpt-diarize/<filename>/<content-hash>/`:

- `audio.m4a` — Extracted audio track
- `chunks/` — Split audio files (long files only)
- `diarized-transcripts/` — Per-chunk diarized JSON (for troubleshooting)
- `transcript.txt` — Formatted transcript with speaker labels like `CHUNK_0_A`
- `speakers.toml` — Template listing all distinct speakers (commented out)

### 2. Review and label speakers

Open `transcript.txt` and identify who each speaker is. Then edit
`speakers.toml` to map chunk-local speaker IDs to real names:

```toml
# Before (auto-generated):
# CHUNK_0_A
# CHUNK_0_B
# CHUNK_1_A

# After (you fill in):
Caleb = ["CHUNK_0_A", "CHUNK_1_A"]
Andrew = ["CHUNK_0_B"]
```

A name can also map to a single label as a bare string:

```toml
Andrew = "CHUNK_0_B"
```

### 3. Apply labels

```bash
transcribe-label transcript.txt speakers.toml
```

This writes `transcript.labeled.txt` with:

- Speaker labels replaced (`CHUNK_0_A` -> `Caleb`)
- Consecutive same-speaker blocks merged (including across chunk boundaries
  where the same name appears on both sides)

## Commands

| Command | Description |
|---------|-------------|
| `transcribe-diarize <audio>` | Transcribe with diarization |
| `transcribe-label --speakers <transcript>` | List distinct speakers in a transcript |
| `transcribe-label <transcript> <labels.toml>` | Apply speaker labels |

## Configuration

| Key | Default | Description |
|-----|---------|-------------|
| `diarization_model` | `gpt-4o-transcribe-diarize` | OpenAI diarization model |
| `split_audio_approx_every_s` | `1200` (20 min) | Target chunk duration in seconds |

## Example

```bash
# 1. Transcribe
transcribe-diarize meeting.m4a

# 2. Edit speakers.toml in the output directory
vim .out/transcribe-gpt-diarize/meeting/*/speakers.toml

# 3. Apply labels
transcribe-label \
  .out/transcribe-gpt-diarize/meeting/*/transcript.txt \
  .out/transcribe-gpt-diarize/meeting/*/speakers.toml

# 4. View result
cat .out/transcribe-gpt-diarize/meeting/*/transcript.labeled.txt
```

All pipeline steps are memoized (`thds.mops.pure.magic`), so re-running
`transcribe-diarize` on the same file skips already-completed work.
