import logging
from pathlib import Path

import openai

from cc.config import DEFAULT_CONFIG
from cc.env import activate_api_keys

logger = logging.getLogger(__name__)


def transcribe_audio(audio_path: Path, transcription_model: str, transcription_prompt: str) -> str:
    """Transcribe audio file using OpenAI Whisper or newer models."""
    transcription_model = transcription_model or DEFAULT_CONFIG.transcription_model
    logger.info(
        f"Transcribing audio: {audio_path} with {transcription_model}, using prompt: {transcription_prompt}"
    )
    activate_api_keys()
    client = openai.OpenAI()

    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model=transcription_model,
            file=audio_file,
            prompt=transcription_prompt,
        )

    return transcript.text
