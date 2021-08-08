"""
Inference entrypoint for WaveRNN.
"""
import os

from omegaconf import OmegaConf
from tqdm import tqdm  # type: ignore
import click
import numpy as np
import soundfile  # type: ignore

from wavernn.util import die_if
from wavernn.model import Model, Config
from wavernn.train import CONFIG_PATH, CHECKPOINTS_DIR, BEST_CHECKPOINT
from wavernn.dataset import AudioDataset

INFERENCE_BAR_FMT = "{l_bar}{bar}| {n:.02f}/{total:.2f} [{rate_noinv_fmt}]"


@click.command("infer")
@click.option(
    "--path", required=True, type=click.Path(file_okay=False), help="Model directory"
)
@click.option(
    "--input",
    "input_file",
    required=True,
    type=click.Path(dir_okay=False, exists=True),
    help="Input WAV file",
)
@click.option(
    "--output",
    "output_file",
    required=True,
    type=click.Path(dir_okay=False),
    help="Output WAV file",
)
def infer(  # pylint: disable=missing-param-doc
    path: str, input_file: str, output_file: str
) -> None:
    """
    Infer with a trained WaveRNN.
    """
    config_path = os.path.join(path, CONFIG_PATH)
    die_if(not os.path.exists(config_path), f"Missing config path {config_path}")

    checkpoint_path = os.path.join(path, CHECKPOINTS_DIR, BEST_CHECKPOINT + ".ckpt")
    die_if(
        not os.path.exists(checkpoint_path),
        f"Missing checkpoint path {checkpoint_path}",
    )

    die_if(not input_file.endswith(".wav"), "--input argument must have .wav extension")
    die_if(
        not output_file.endswith(".wav"), "--output argument must have .wav extension"
    )

    # Create a model with the config.
    model_config: Config = OmegaConf.structured(Config)
    model_config.merge_with(OmegaConf.load(config_path))  # type: ignore
    model = Model.load_from_checkpoint(checkpoint_path, config=model_config)

    # Load the data using a dataset.
    dataset = AudioDataset([input_file], model_config.data)
    clips = list(dataset.load_samples_from(input_file))

    sample_rate = model_config.data.mel.sample_rate
    total_samples: int = sum(int(clip.waveform.size) for clip in clips)  # type: ignore
    total_secs = total_samples / sample_rate

    synthesized_clips = []
    with tqdm(total=total_secs, bar_format=INFERENCE_BAR_FMT, unit="sec") as progress:
        for synthesized in model.infer(
            (clip.spectrogram for clip in clips), native=True
        ):
            synthesized_clips.append(synthesized)
            progress.update(synthesized.size / sample_rate)

    soundfile.write(output_file, np.concatenate(synthesized_clips), sample_rate)
