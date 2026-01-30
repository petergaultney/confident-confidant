from thds.mops import pure

from .core import transcribe_audio_file

pure.magic.pipeline_id("transcribe")

__all__ = ["transcribe_audio_file"]
