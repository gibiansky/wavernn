"""
__init__.py for WaveRNN package.
"""
from wavernn.main import main
from wavernn.model import InferenceWaveRNN


def load(path: str, clip_frames: int) -> InferenceWaveRNN:
    """
    Load a WaveRNN for inference.

    Args:
        path: Path to an exported WaveRNN JIT file.
        clip_frames: How many frames to synthesize in each step.

    Returns:
      A WaveRNN which can be used for inference.
    """
    return InferenceWaveRNN(path=path, clip_frames=clip_frames)
