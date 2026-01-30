import logging

from thds.core.source import Source
from thds.mops import pure

from cc.transcribe.llm.transcribe_chunks import ChunkTranscript
from cc.transcribe.workdir import workdir
from cc.transcribe import llm

logger = logging.getLogger(__name__)


@pure.magic()
def stitch_transcripts(chunk_transcripts: list[ChunkTranscript], model: str) -> Source:
    chunk_transcripts = sorted(chunk_transcripts, key=lambda t: t.index)

    out_raw = workdir() / "transcript.raw.txt"
    out_txt = workdir() / "transcript.txt"

    if len(chunk_transcripts) == 1 and (txt := chunk_transcripts[0].text.strip()):
        out_txt.write_text(txt + "\n", encoding="utf-8")
        logger.info(f"Received 1 chunk to stitch; wrote: {out_txt}")
        return Source.from_file(out_txt)

    txt_parts = [txt for ct in chunk_transcripts if (txt := ct.text.strip())]
    if not txt_parts:
        raise ValueError("No text found in transcripts.")

    raw_text = "\n\n".join(txt_parts)
    out_raw.write_text(raw_text + "\n", encoding="utf-8")  # for troubleshooting
    logger.info(f"Wrote raw joined-transcript: {out_raw}")

    logger.info(f"Reformatting with {model}...")
    reformatted = llm.reformat_stitched_transcript(raw_text, model)
    out_txt.write_text(reformatted + "\n", encoding="utf-8")
    logger.info(f"Wrote reformatted transcript: {out_txt}")

    return Source.from_file(out_txt)
