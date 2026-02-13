import logging
import re
import textwrap
import time
import typing as ty

from litellm import completion

from cc.config import DEFAULT_NOTE_PROMPT
from cc.env import activate_api_keys

logger = logging.getLogger(__name__)


def _test_transcript_equivalence(original: str, modified: str) -> bool:
    """Check if the second string contains the first, ignoring whitespace and capitalization."""
    original_cleaned = re.sub(r"\s+", " ", original).strip().lower()
    modified_cleaned = re.sub(r"\s+", " ", modified).strip().lower()
    return original_cleaned in modified_cleaned


class SummaryNote(ty.NamedTuple):
    title: str
    note: str


def summarize_transcript(
    ll_model: str,
    transcript: str,
    prompt: str,
    context: str = "",
) -> SummaryNote:
    """Get title and summary note from LLM; the summary note will be formatted as returned by the LLM.

    Your prompt MUST NOT redefine the first line of output from the LLM, which is
    specified to be a short title that can also be used as a filename.

    The function also tacks on the raw transcript if your resulting note does not include
    some whitespace-compressed version of the transcript.

    context is prepended as background info (e.g. transcription prompt with names/terms).
    """
    logger.info(f"Getting note and title from {ll_model}")

    context_section = f"Background context:\n{context}\n\n" if context.strip() else ""
    prompt = (
        context_section
        + textwrap.dedent(
            """
        Please analyze the raw transcript at the end and provide:

        1. A short title (3-7 words, suitable for a filename) - put this as the very first
           line, followed by a newline, regardless of any formatting instructions that follow.
           This must never be missing, and it must always be at least 3 words and no more than 7,
           and they must be as unique as possible using the content of the transcript, since this will
           be part of a filename.
        """
        )
        + (prompt or DEFAULT_NOTE_PROMPT)
        + (
            "\n\n" + "Remember - regardless of the rest of the format of your response,"
            " the very first line must be a 3-7 word title on a line by itself."
        )
        + f"\n\nRaw transcript to be analyzed:\n{transcript}"
    )

    activate_api_keys()
    logger.debug("Prompt:\n%s", prompt)
    t0 = time.monotonic()
    response = completion(model=ll_model, messages=[{"role": "user", "content": prompt}])
    logger.info(f"LLM response received in {time.monotonic() - t0:.1f}s")
    content = response["choices"][0]["message"]["content"]
    logger.debug("Response:\n%s", content)

    first_line_is_title, rest_of_note = content.split("\n", 1)
    if not _test_transcript_equivalence(transcript, rest_of_note):
        rest_of_note = rest_of_note.rstrip() + f"\n\n# Raw transcript\n\n{transcript}"
    return SummaryNote(first_line_is_title.strip(), rest_of_note)


def summarize_diarized_meeting(
    ll_model: str,
    annotated_transcript: str,
    prompt: str,
    context: str = "",
) -> SummaryNote:
    """Single-pass fixup + summarization for diarized meetings.

    Takes a transcript where a human reviewer has already replaced identified
    CHUNK_N_X labels with real speaker names inline. Remaining CHUNK labels
    are unidentified and should be inferred from context.
    """
    logger.info(f"Getting diarized meeting note from {ll_model}")

    context_section = f"Background context:\n{context}\n\n" if context.strip() else ""
    prompt = (
        context_section
        + textwrap.dedent(
            """\
        You are processing a diarized meeting transcript. A human reviewer has already
        identified most speakers — their real names appear directly as speaker labels.

        Any remaining CHUNK_N_X labels are speakers the reviewer did not identify.
        Labels marked "<uncertain>" mean the reviewer heard this speaker but could not
        determine who they are. Plain CHUNK_N_X labels (without "<uncertain>") were never
        evaluated — use conversational context and the identified speakers to infer them.

        IMPORTANT: The human reviewer's identifications are authoritative. Each utterance
        was individually reviewed. Do NOT change an utterance's speaker from the name the
        reviewer assigned. Only resolve the remaining CHUNK_N_X labels.

        Please provide:

        1. A short title (3-7 words, suitable for a filename) as the very first line.
        2. The summary/analysis as instructed below.
        3. A corrected transcript with all remaining CHUNK_N_X labels replaced by real
           speaker names. Merge consecutive same-speaker paragraphs. Preserve the
           original words.

        """
        )
        + (prompt or DEFAULT_NOTE_PROMPT)
        + (
            "\n\nRemember - regardless of the rest of the format of your response,"
            " the very first line must be a 3-7 word title on a line by itself."
        )
        + f"\n\nPartially-identified diarized transcript:\n{annotated_transcript}"
    )

    activate_api_keys()
    logger.debug("Prompt:\n%s", prompt)
    t0 = time.monotonic()
    response = completion(model=ll_model, messages=[{"role": "user", "content": prompt}])
    logger.info(f"LLM response received in {time.monotonic() - t0:.1f}s")
    content = response["choices"][0]["message"]["content"]
    logger.debug("Response:\n%s", content)

    first_line_is_title, rest_of_note = content.split("\n", 1)
    return SummaryNote(first_line_is_title.strip(), rest_of_note)
