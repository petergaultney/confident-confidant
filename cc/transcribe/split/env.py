import shutil

from thds.core.lazy import lazy


@lazy
def which_ffmpeg_or_raise() -> str:
    if ffmpeg := shutil.which("ffmpeg"):
        return ffmpeg
    raise EnvironmentError("ffmpeg not installed; use homebrew to install it")
