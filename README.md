# The (overly?) Confident Confidant

Turn audio files within Markdown folders into helpful Obsidian notes and transcripts.

Will find any `.m4a` or `.webm` file within a directory (recursively) that is linked to
from within any `.md` file. The link currently has to be an Obsidian-style embed,
e.g. `![[the-audio.m4a]]` - but we can change this eventually.

Does not need to be pointed at your entire Vault - can run on any subdirectory within it.

## Usage


Use the default prompt and settings by just running the script. You'll need an OpenAI API
key and an Anthropic API key.

`./cc.py <your-vault-dir>`

To 'test' (avoid modifying any of your existing notes), run as `./cc.py --no-mutate` -
this will create a new note and will make a copy of your audio file, but will not change
existing notes.

### Config

Reads its config from the Markdown 'vault' itself. An example config markdown that mostly mirrors
the default config would look like [the file linked here](example_config.md).

For any given audio file discovered by `cc`, will look recursively 'upward' to find
config. The first found file that contains any config will override the default config,
and the tool stops recursing upward for further config.
