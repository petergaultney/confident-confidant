"""Simple Python formatter for diarized transcripts (replaces LLM stitch)."""

import logging

from thds.core.source import Source
from thds.mops import pure

from cc.transcribe.diarize.llm.transcribe_chunks import DiarizedChunkTranscript
from cc.transcribe.workdir import workdir

logger = logging.getLogger(__name__)


@pure.magic()
def format_diarized_transcripts(transcripts: list[DiarizedChunkTranscript]) -> Source:
    """
    Format diarized transcripts into readable text without LLM.

    This replaces the LLM-based stitch step with simple Python logic:
    - Merges consecutive same-speaker segments
    - Adds paragraph breaks between speaker changes
    - Adds "---" markers between chunks
    """
    lines: list[str] = []
    transcripts = sorted(transcripts, key=lambda t: t.index)

    for i, transcript in enumerate(transcripts):
        current_speaker: str | None = None
        current_texts: list[str] = []

        def flush() -> None:
            nonlocal current_speaker, current_texts
            if current_texts and current_speaker:
                text = " ".join(current_texts)
                lines.append(f"{current_speaker}: {text}")

        for seg in transcript.segments:
            if not seg.text.strip():
                continue

            if seg.speaker == current_speaker:
                # Same speaker - accumulate text
                current_texts.append(seg.text)
            else:
                # Speaker changed - flush previous and start new
                flush()
                if current_speaker is not None:
                    lines.append("")  # Paragraph break
                current_speaker = seg.speaker
                current_texts = [seg.text]

        flush()

        # Add separator between chunks
        if i < len(transcripts) - 1:
            lines.append("")
            lines.append(f"--- CHUNK_{i + 1} ---")
            lines.append("")

    # Write output
    out_txt = workdir() / "transcript.txt"
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"Wrote: {out_txt}")

    return Source.from_file(out_txt)
