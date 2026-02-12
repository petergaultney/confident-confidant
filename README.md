# `coco` - the (overly?) Confident Confidant

(Named `coco` because `cc` is the C compiler.)

Turn audio files within Markdown folders into helpful Markdown notes and transcripts.

Originally built for use with Obsidian[^1] and its bare-bones (but useful!) Audio recorder
core plugin, but it will work with any folder that contains Markdown files and `.m4a` or
`.webm` files.

[^1]: I recommend setting Obsidian's default location for new attachments (under Options
    -> Files and Links) to 'Same folder as current file' - this tends to make for nicer
    overall organization of things even though this utility will end up reorganizing these
    audio recordings according to its own configuration.

Will find any `Recording*.m4a` or `Recording*.webm` file within a directory (recursively)
that is linked to from within any `.md` file. The link can be a "wikilink"
(`[[audio.m4a|optional name]]`), an embed (wikilink prefixed with `!`), or a standard
Markdown link (`[name](audio.m4a)`).

Does not need to be pointed at your entire Vault - can run on any subdirectory within it.

## Setup

You'll need `uv` [installed](https://docs.astral.sh/uv/getting-started/installation/) and
`ffmpeg` (required for splitting long audio files):

```sh
brew install ffmpeg
```

Install `coco` as an editable tool:

```sh
uv tool install -e .
```

You'll need an OpenAI API key and an Anthropic API key. Place them in `~/.keys/openai-api`
and `~/.keys/anthropic-api`, or set `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` environment variables.

If you wish to use only OpenAI, you can skip the Anthropic key, but you'll need to
configure one of OpenAI's LLMs as the `note_model`, using the config format shown
below. As of this writing, I'm not aware of an Anthropic transcription model that we could
be supporting.

## Usage

### `coco` - Process audio recordings

```sh
coco <your-vault-dir>
```

To 'test' (avoid modifying any of your existing notes), add the `--no-mutate` flag -
this will create the output note, but will not move the audio file and will not
mutate any existing notes.

Run in an infinite loop ('server mode') with `--loop`. The sleep is hardcoded to 10
seconds because I am lazy.

### `coco-meeting` - Process diarized meeting recordings

```sh
coco-meeting <audio-file>
```

Two-phase workflow for multi-speaker recordings:

1. **Phase 1**: Transcribes and diarizes the audio, writing a `speakers.toml` with
   placeholder labels (`CHUNK_0_A`, `CHUNK_1_B`, etc.). Prints the file paths and exits.
2. **Edit `speakers.toml`** to map labels to real names:
   ```toml
   Peter = ["CHUNK_0_B", "CHUNK_1_A"]
   Grant = ["CHUNK_0_A", "CHUNK_1_B"]
   ```
3. **Phase 2**: Re-run the same command. It detects that `speakers.toml` has mappings,
   applies them, and produces a labeled summary note.

Re-running phase 1 is free (transcription results are `mops`-cached). Use `--no-mutate` to
skip moving files and modifying vault notes.

Mark recordings for `coco-meeting` by adding `#diarize` to the link line in your daily
note — batch `coco` will skip these files so they don't get processed as solo voice memos.

### `coco-summarize` - Summarize an existing transcript

```sh
coco-summarize <transcript-file>
```

Takes an existing transcript `.txt` file and generates a summary note. Optionally
specify an output path with `--output`/`-o`.

### Config

Reads its config from the Markdown 'vault' itself. An example config markdown that mostly mirrors
the default config would look like [the file linked here](cc-config.md).

For any given audio file discovered by `coco`, will look recursively 'upward' to find
config. For scalar fields (`note_model`, `audio_dir`, etc.), the nearest config to the
file wins. For named prompts, see [hierarchical prompt resolution](#hierarchical-prompt-resolution)
below.

Config will be looked for in files with these names as we recurse upward:

- `.cc-config.md`
- `cc-config.md`
- `<current-dirname>.md`

The last follows the convention of the ['Folder Notes' plugin for
Obsidian](https://github.com/LostPaul/obsidian-folder-notes).

#### Named prompts and tags

The `## Note Prompt` heading defines the default summarization prompt. You can also define
**named prompts** by appending a colon and a name:

```markdown
## Note Prompt: meeting

Summarize this multi-speaker meeting. Identify action items and decisions.

## Note Prompt: standup

Brief bullet points of what each person said.
```

Select a prompt by adding **tags** to the link or embed line in your Markdown note:

```markdown
![[Recording-20260212.webm]] #diarize #meeting
```

- `#diarize` is a **meta tag** — it tells batch `coco` to skip the file (use `coco-meeting`
  instead). It does not select a prompt.
- All other tags (e.g. `#meeting`, `#standup`) are **prompt selectors**.
- **No tags** → the `default` prompt is used. **Specific tags present** → only those named
  prompts are used (not the default).
- Tags can appear on the same line as the link or on the line immediately above it.

#### Hierarchical prompt resolution

When multiple config files exist in the directory hierarchy (e.g., vault root and a
subdirectory), prompts are **concatenated root-to-file** for each tag. This lets you put
shared context like "I am Peter, a software engineer" in a root config while adding
structure-specific instructions in leaf configs.

Scalar config fields (`note_model`, `audio_dir`, etc.) still use nearest-to-file-wins.

#### Transcription context

The `## Transcription Context` heading (or legacy `## Transcription Prompt`) provides names,
terms, and pronunciations. This text is passed to both the transcription API (as a prompt
hint) and to the summarizer (as background context).

# Future work

- [x] Support standard Markdown style links in addition to the Obsidian piped links
- [x] Discover config at arbitrary level of header nesting
- [x] Better support for non-Obsidian vaults - 'vault' root discovery is now more robust in these situations.
- [ ] Auto-configure default LLM based on discovered API key(s)
- [ ] Provide an outline/summary refinement mode, where an already-transcribed recording
      and its existing note can be iterated on with refined LLM prompts?
- [x] Pick up partial configs recursively — hierarchical prompt resolution now concatenates
      named prompts from root-to-file, while scalar fields use nearest-to-file-wins.
- [ ] Potentially (configurably) allow embedding the resulting note _within_ the note
      where the audio file was linked. This would probably require some Markdown
      cleverness to respect headings.
- [x] Support audio recordings larger than 25 MB. This is a low priority feature for me
      and would likely require taking on new dependencies.
