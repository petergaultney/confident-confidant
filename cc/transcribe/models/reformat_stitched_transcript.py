import logging

from litellm import completion

from cc.env import activate_api_keys

logger = logging.getLogger(__name__)

REFORMAT_SYSTEM_PROMPT = """\
You are a transcript editor. The user will provide a transcript that has been \
stitched together from multiple audio fragments. Because of this fragmentation, \
some sentences may have incorrect capitalization or punctuation at the boundaries \
where fragments were joined.

Your task:
1. Correct any capitalization errors (e.g., lowercase letter at the start of a sentence)
2. Correct any punctuation errors (e.g., missing periods, incorrect comma placement at boundaries)
3. Add paragraph breaks where natural topic or speaker changes occur

CRITICAL RULES:
- You may ONLY change whitespace (spaces, newlines) and capitalization/punctuation
- Do NOT change any words, add words, remove words, or rephrase anything
- Do NOT add any commentary, headers, or formatting beyond paragraph breaks
- Return ONLY the corrected transcript text, nothing else
"""


def reformat_stitched_transcript(text: str, model: str) -> str:
    """
    Send the stitched transcript to an LLM to fix capitalization/punctuation
    and add paragraph breaks.
    """
    activate_api_keys()

    response = completion(
        model=model,
        messages=[
            {"role": "system", "content": REFORMAT_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    )
    content = response.choices[0].message.content
    return content.strip() if content else text
