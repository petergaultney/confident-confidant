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

You'll need `uv` [installed](https://docs.astral.sh/uv/getting-started/installation/).

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
config. The first found file that contains any config will override the default config,
and the tool stops recursing upward for further config.

Config will be looked for in files with these names as we recurse upward:

- `.cc-config.md`
- `cc-config.md`
- `<current-dirname>.md`

The last follows the convention of the ['Folder Notes' plugin for
Obsidian](https://github.com/LostPaul/obsidian-folder-notes).

# Future work

- [x] Support standard Markdown style links in addition to the Obsidian piped links
- [x] Discover config at arbitrary level of header nesting
- [x] Better support for non-Obsidian vaults - 'vault' root discovery is now more robust in these situations.
- [ ] Auto-configure default LLM based on discovered API key(s)
- [ ] Provide an outline/summary refinement mode, where an already-transcribed recording
      and its existing note can be iterated on with refined LLM prompts?
- [ ] Pick up partial configs recursively? E.g., allow defining the desired model and
      directory config at the root of a vault, while still having specific prompts per
      directory.
- [ ] Potentially (configurably) allow embedding the resulting note _within_ the note
      where the audio file was linked. This would probably require some Markdown
      cleverness to respect headings.
- [ ] Support audio recordings larger than 25 MB. This is a low priority feature for me
      and would likely require taking on new dependencies.
