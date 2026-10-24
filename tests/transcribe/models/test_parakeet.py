import pytest

from cc.transcribe.models.parakeet import hf_repo_from_model_name


@pytest.mark.parametrize(
    "transcription_model, expected",
    [
        ("parakeet-mlx/mlx-community/parakeet-tdt-0.6b-v3", "mlx-community/parakeet-tdt-0.6b-v3"),
        ("gpt-4o-transcribe", None),
        ("whisper-1", None),
        ("mlx-community/parakeet-tdt-0.6b-v3", None),
    ],
)
def test_hf_repo_from_model_name(transcription_model: str, expected: str | None):
    assert hf_repo_from_model_name(transcription_model) == expected


def test_hf_repo_from_model_name_rejects_missing_repo():
    with pytest.raises(ValueError, match="names no Hugging Face repo"):
        hf_repo_from_model_name("parakeet-mlx/")
