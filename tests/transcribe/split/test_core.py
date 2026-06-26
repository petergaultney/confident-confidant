from pathlib import Path

import pytest

from cc.transcribe.split.core import _build_extract_audio_cmd


@pytest.mark.parametrize(
    "n_audio_streams, expected_stream_args",
    [
        (1, ["-map", "0:a:0"]),
        (2, ["-filter_complex", "[0:a:0][0:a:1]amix=inputs=2:duration=longest", "-ac", "1"]),
        (
            3,
            ["-filter_complex", "[0:a:0][0:a:1][0:a:2]amix=inputs=3:duration=longest", "-ac", "1"],
        ),
    ],
)
def test_maps_first_stream_for_mono_and_mixes_down_multitrack_inputs(
    n_audio_streams, expected_stream_args
):
    cmd = _build_extract_audio_cmd(Path("/tmp/in.m4a"), Path("/tmp/out.m4a"), n_audio_streams)

    assert cmd == [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", "/tmp/in.m4a",
        "-vn", *expected_stream_args, "-c:a", "aac",
        "/tmp/out.m4a",
    ]
