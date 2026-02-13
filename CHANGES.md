# 3.0.0

- Interactive speaker identification for `coco-meeting -i`: walk through diarized
  utterances and assign speakers by name, then let the LLM fix up and summarize.
- Textual TUI (default) with scrollable table, colored emoji per speaker, conflict
  markers, word count, expanded text panel, directional auto-advance, and undo.
  Old sequential terminal flow preserved via `--no-tui`.
- `--speakers "Peter,Eby,Grant"` flag to skip the name prompt.
- Per-utterance inline transcript annotation (`CHUNK_0_A - Peter:`) so the LLM
  sees exactly which utterances were identified, including conflicting labels.
- Named prompts with tag-based selection: define `## Note Prompt: meeting` (or any name)
  in config, then select it with `#meeting` on the link line in your note.
- `coco-meeting`: two-phase vault-aware diarized meeting transcription. Phase 1 transcribes
  and writes `speakers.toml`; phase 2 applies speaker labels and summarizes.
- Hierarchical prompt resolution: named prompts are concatenated root-to-file, so vault-wide
  context composes with directory-specific instructions.
- `transcription_context` (renamed from `transcription_prompt`; old name still accepted):
  passed to both the transcription API and the summarizer as background context.
- Meeting notes now extract full section context from the heading above the audio link,
  giving the summarizer richer context about the meeting.
- `#diarize` tag on link lines tells batch `coco` to skip the file.
- Session logging to `/tmp/coco-meeting-logs/` for debugging identification decisions.
- Default `note_model` changed to `gpt-5.2`.
- Fix: `speakers.toml` is no longer overwritten on re-runs if it already exists.

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
