"""On-device transcription with NVIDIA Parakeet, via parakeet-mlx (Apple Silicon only).

Needs the `parakeet` extra. The model weights (~1.2 GB for parakeet-tdt-0.6b-v3) download into the
Hugging Face cache on first use. Parakeet takes no prompt, so there is no transcription context here.
"""

import logging
import platform
import sys
import typing as ty

from thds.mops import pure

from cc.transcribe.split import Chunk
from cc.transcribe.workdir import workdir

from . import debug
from .types import ChunkTranscript

if ty.TYPE_CHECKING:
    from parakeet_mlx import BaseParakeet, DecodingConfig

logger = logging.getLogger(__name__)

_MODEL_PREFIX = "parakeet-mlx/"
_LOCAL_ATTENTION_CONTEXT_SIZE = 256  # parakeet-mlx's CLI default


def hf_repo_from_model_name(transcription_model: str) -> str | None:
    """`parakeet-mlx/mlx-community/parakeet-tdt-0.6b-v3` -> `mlx-community/parakeet-tdt-0.6b-v3`.

    None means the model is not a Parakeet model.
    """
    if not transcription_model.startswith(_MODEL_PREFIX):
        return None
    hf_repo = transcription_model.removeprefix(_MODEL_PREFIX)
    if not hf_repo:
        raise ValueError(
            f"transcription_model {transcription_model!r} names no Hugging Face repo;"
            f" e.g. {_MODEL_PREFIX}mlx-community/parakeet-tdt-0.6b-v3"
        )
    return hf_repo


def _load_parakeet(hf_repo: str) -> tuple["BaseParakeet", "DecodingConfig"]:
    # The extra's platform marker makes it install nothing elsewhere, so check this before
    # suggesting the user install it.
    if not (sys.platform == "darwin" and platform.machine() == "arm64"):
        raise RuntimeError("Parakeet transcription runs only on Apple Silicon (MLX).")
    try:
        from parakeet_mlx import Beam, DecodingConfig, from_pretrained
    except ImportError as err:
        raise ImportError(
            "Parakeet transcription needs the `parakeet` extra: uv tool install -e '.[parakeet]'"
        ) from err

    model = from_pretrained(hf_repo)
    # We transcribe each chunk in one pass; our silence-based split already chunks long files.
    # Local attention keeps memory bounded.
    model.encoder.set_attention_model(
        "rel_pos_local_attn", (_LOCAL_ATTENTION_CONTEXT_SIZE, _LOCAL_ATTENTION_CONTEXT_SIZE)
    )

    # Greedy decoding can get stuck partway through a chunk and silently emit nothing for the rest
    # of it; beam decoding recovers. These are the parakeet-mlx CLI's beam settings, which differ
    # from the `Beam` dataclass defaults.
    decoding_config = DecodingConfig(
        decoding=Beam(beam_size=5, length_penalty=0.013, patience=3.5, duration_reward=0.67)
    )

    return model, decoding_config


@pure.magic()
def transcribe_chunks(chunks: ty.Sequence[Chunk], hf_repo: str) -> list[ChunkTranscript]:
    """mops memoizes on the arguments only, so bump the key whenever the decoding or attention
    settings in _load_parakeet change.

    function-logic-key: beam-v1
    """
    out_dir = workdir() / "chunk-transcripts"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Transcribing {len(chunks)} chunks on-device, using {hf_repo}...")
    model, decoding_config = _load_parakeet(hf_repo)

    # One chunk at a time: the chunks share the single loaded model and the GPU.
    transcripts: list[ChunkTranscript] = []
    for chunk in chunks:
        result = model.transcribe(chunk.audio_src.path(), decoding_config=decoding_config)
        transcript = ChunkTranscript(index=chunk.index, text=result.text, audio_src=chunk.audio_src)
        debug.write_chunk_transcript(out_dir, transcript)
        logger.info(f"ok  {chunk.audio_src.path().name}")
        transcripts.append(transcript)

    return transcripts
