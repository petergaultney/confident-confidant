# Confident Confidant Config

## Base Config

```hjson
transcription_model: gpt-4o-transcribe
note_model: anthropic/claude-sonnet-4-20250514
audio_dir: ./audio
transcripts_dir: ./transcripts
```

## Transcription Prompt

Audio notes discovered here are related to my work as a software developer.

## Note Prompt

```
2. A concise summary (2-3 sentences) - I am the speaker, so use first-person perspective
3. A complete and organized markdown-native outline,
	with the highest level of heading being `##`,
	because it will be embedded inside an existing markdown document.
	If it makes sense for the content, try to orient around high level categories like
	"intuitions", "constraints", "assumptions",
	"alternatives or rejected ideas", "tradeoffs", and "next steps",
	though these don't necessarily need to be present or even the headings.

    If the audio is more like a retelling of my day, then write the outline
    in three distinct named sections as:
	  - a markdown list (-) of activites I engaged in, or tasks I accomplished
	  - specific 'tasks' on my plate (not someone else's);
	    formatted as Markdown tasks, e.g. ` - [ ] <task text>`
	  - a list of other insights, through-lines, or points to ponder.

    If you think it fits neither of these categories, use your best judgment
	on the outline structure and format.
4. A readable transcript of the audio, broken up into paragraphs.
	Never leave the most key thoughts buried in long paragraphs.
	Change ONLY whitespace!

Format your response as:
# Summary

{2. summary}

# Outline

{3. outline}

# Transcript

{4. full_readable_transcript}
```
