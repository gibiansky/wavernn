"""
Dataset-handling entrypoint for WaveRNN package.
"""
from dataclasses import dataclass
import glob
import os
import json
from typing import Iterator, Optional, NamedTuple

from omegaconf import MISSING
import click
import pytorch_lightning as pl
import torch
import librosa  # type: ignore
import numpy as np

from wavernn.util import download, die_if, cmd

# Public-facing dataset names.
NAME_LJSPEECH: str = "ljspeech"

# Where to store dataset information.
DATASET_JSON: str = "dataset.json"

# Keys required in dataset.json
TRAIN_KEY: str = "train"
VALID_KEY: str = "valid"
TEST_KEY: str = "test"

# Dictionary mapping dataset to download URL.
DATASETS: dict[str, str] = {
    NAME_LJSPEECH: "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
}


@dataclass
class MelConfig:
    """Configuration for log mel spectrogram extraction."""

    # Sample rate of the audio. If the audio is not this sample rate, it will
    # be up- or downsampled to this sample rate when it is loaded.
    sample_rate: int = MISSING

    # Number of Fourier coefficients in the spectrogram.
    n_fft: int = MISSING

    # Number of bands in the mel spectrogram.
    n_mels: int = MISSING

    # Minimum frequency to include in the mel spectrogram.
    fmin: float = MISSING

    # Maximum frequency to include in the mel spectrogram.
    fmax: float = MISSING

    # How many samples to shift between each consecutive frame.
    hop_length: int = MISSING

    # How many samples to include in the window used for STFT extraction.
    win_length: int = MISSING

    # Minimum spectrogram magnitude to enforce to avoid log underflow.
    # Log mel spectrogram is ln(clip(mel_spectrogram, log_epsilon)).
    log_epsilon: float = MISSING


@dataclass
class DataConfig:
    """
    Configuration for a training dataset.
    """

    # Path where to load the dataset.
    path: str = MISSING

    # Configuration for mel spectrogram extraction.
    mel: MelConfig = MISSING

    # How many frames to include in each audio sample.
    # The number of audio samples is clip_frames * mel.hop_length.
    clip_frames: int = MISSING

    # How many frames of "padding" data to include in the spectrogram. Padding
    # frames are included so that, if your sample starts at frame t, the data
    # starts at frame t - padding_frames and ends at t + clip_frames + padding_frames.
    # This allows a convolutional conditioning network to have proper data on
    # the left and right edges, instead of zero-padding, which would make it so
    # that the inference behavior differs from the training behavior. The
    # padding frames are computed as if the data in the waveform was zeros.
    padding_frames: int = MISSING

    # How many samples to include in a batch.
    batch_size: int = MISSING


class AudioSample(NamedTuple):
    """
    A training sample.
    """

    # A float32 waveform tensor of shape [num_samples].
    # After batching, the shape will be [batch_size, num_samples].
    waveform: torch.Tensor

    # A float32 mel spectrogram tensor of shape [n_mels, num_frames].
    # After batching, the shape will be [batch_size, n_mels, num_frames].
    spectrogram: torch.Tensor


class AudioDataset(torch.utils.data.IterableDataset):
    """
    A dataset of audio waveforms and mel spectrograms.
    """

    def __init__(self, globs: list[str], config: DataConfig) -> None:
        """
        Create a new dataset.

        Args:
          globs: The filename globs to use for the input data.
          config: The dataset configuration.
        """
        super().__init__()

        self.config = config

        # Collect all files in this dataset.
        self.filenames: list[str] = []
        for glb in globs:
            self.filenames.extend(glob.glob(os.path.join(config.path, glb)))

    def __iter__(self) -> Iterator[AudioSample]:
        """
        Iterate over this dataset.

        Yields:
          Audio samples containing mel spectrograms and waveforms.
        """
        # Split files by worker.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            filenames = self.filenames[:]
        else:
            filenames = [
                filename
                for i, filename in enumerate(self.filenames)
                if i % worker_info.num_workers == worker_info.id  # type:ignore
            ]

        for filename in filenames:
            yield from self.load_samples_from(filename)

    def load_samples_from(
        self, filename: str, clip_frames: Optional[int] = None
    ) -> Iterator[AudioSample]:
        """
        Load training samples from a file.

        Args:
          filename: Path to load from.
          clip_frames: How many spectrogram frames to include in each clip.
            If not provided, uses the default in the data configuration.

        Yields:
          All the training samples from a given file.
        """
        if clip_frames is None:
            clip_frames = self.config.clip_frames

        # Load waveform from disk.
        mel = self.config.mel

        raw_waveform: np.ndarray
        sr: int
        raw_waveform, sr = librosa.load(filename, sr=mel.sample_rate)

        # Ensure samples are all in (-1, 1).
        raw_waveform = np.clip(raw_waveform, -0.9999999, 0.9999999)

        # Pad waveform with silence at the end. Our spectrogram frames are
        # non-centered, and we'd like the number of samples in the waveform to
        # be exactly the hop length times the number of spectrogram frames. In
        # order to do this, we compute the number of frames we want, then pad
        # the waveform so that its length is equal to
        #   waveform_length = (num_frames - 1) * hop_length + win_length
        # (This assumes that win_length > hop_length.)
        # Althought the final waveform length is as above, we want our
        # spectrogram length to have extra padding frames on the left and
        # right. In order to achieve this, we pad the waveform with zero
        # samples equal to the number of desired padding frames.
        assert mel.win_length >= mel.hop_length, "Window cannot be less than hop"
        num_frames = (raw_waveform.size + mel.hop_length - 1) // mel.hop_length
        needed_length = (num_frames - 1) * mel.hop_length + mel.win_length
        padding_frames = self.config.padding_frames
        padding_samples = padding_frames * mel.hop_length
        padded_waveform = np.pad(
            raw_waveform,
            (padding_samples, padding_samples + needed_length - raw_waveform.size),
        )
        spectrogram: np.ndarray
        spectrogram = librosa.feature.melspectrogram(
            padded_waveform,
            sr,
            n_fft=mel.n_fft,
            hop_length=mel.hop_length,
            win_length=mel.win_length,
            center=False,
            n_mels=mel.n_mels,
            fmin=mel.fmin,
            fmax=mel.fmax,
        )
        log_epsilon = torch.tensor(mel.log_epsilon, dtype=torch.float32)
        for i in range(0, spectrogram.shape[1], clip_frames):
            desired_frames = clip_frames + 2 * padding_frames
            clip_spectrogram = spectrogram[:, i : i + desired_frames]
            if clip_spectrogram.shape[1] != desired_frames:
                continue

            start_sample = (i + padding_frames) * mel.hop_length
            end_sample = start_sample + clip_frames * mel.hop_length
            clip_waveform = padded_waveform[start_sample:end_sample]
            log_spectrogram = torch.log(torch.maximum(clip_spectrogram, log_epsilon))
            yield AudioSample(waveform=clip_waveform, spectrogram=log_spectrogram)


class AudioDataModule(pl.LightningDataModule):
    """
    Data loading module for audio data.
    """

    def __init__(self, config: DataConfig) -> None:
        """
        Create a new data module.

        Args:
          config: Configuration for the data module.
        """
        super().__init__()
        self.config = config

        self.train_set: Optional[AudioDataset] = None
        self.valid_set: Optional[AudioDataset] = None
        self.test_set: Optional[AudioDataset] = None

    def setup(self, stage: Optional[str] = None):
        """
        Set up this data module.

        Args:
          stage: Which stage to set up. If stage is None, set up all stages.
            Stage can also be 'fit', 'validate', or 'test'.
        """
        # Load the listing.
        listing_path = os.path.join(self.config.path, DATASET_JSON)
        with open(listing_path, "r") as handle:
            listing = json.load(handle)

        self.train_set = AudioDataset(listing[TRAIN_KEY], self.config)
        self.valid_set = AudioDataset(listing[VALID_KEY], self.config)
        self.test_set = AudioDataset(listing[TEST_KEY], self.config)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Create a training data loader.

        Returns:
          A training data loader.
        """
        dset = self.train_set
        assert dset is not None
        return torch.utils.data.DataLoader(
            dset, batch_size=self.config.batch_size, num_workers=16
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Create a validation data loader.

        Returns:
          A validation data loader.
        """
        dset = self.valid_set
        assert dset is not None
        return torch.utils.data.DataLoader(
            dset, batch_size=self.config.batch_size, num_workers=2
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """Create a test data loader.

        Returns:
          A test data loader.
        """
        dset = self.test_set
        assert dset is not None
        return torch.utils.data.DataLoader(dset, batch_size=self.config.batch_size)


def download_ljspeech(destination: str) -> None:
    """
    Download LJSpeech dataset.

    Args:
      destination: Where to download dataset to.
    """
    if os.path.exists(destination):
        is_empty_dir = os.path.isdir(destination) and len(os.listdir(destination)) == 0
        die_if(not is_empty_dir, f"{destination} already exists")
    else:
        os.makedirs(destination)

    path = download(DATASETS[NAME_LJSPEECH], destination)

    print("Extracting dataset (this may take a while)...")
    cmd("tar", "--directory", destination, "-xf", path)

    print("Removing compressed version...")
    os.unlink(path)

    listing_path = os.path.join(destination, DATASET_JSON)
    with open(listing_path, "w") as handle:
        json.dump(
            {
                TRAIN_KEY: [f"LJSpeech-1.1/wavs/LJ{i:03d}*.wav" for i in range(2, 51)],
                VALID_KEY: [
                    f"LJSpeech-1.1/wavs/LJ001-00{i}*.wav" for i in range(2, 10)
                ],
                TEST_KEY: [
                    "LJSpeech-1.1/wavs/LJ001-000*.wav",
                    "LJSpeech-1.1/wavs/LJ001-001*.wav",
                ],
            },
            handle,
            indent=2,
        )


def verify_dataset(path: str) -> None:
    """
    Check that a dataset is correctly formatted.

    Args:
      path: Path to the dataset.
    """
    # Check that the dataset listing JSON exists.
    listing_path = os.path.join(path, DATASET_JSON)
    die_if(not os.path.exists(path), f"Dataset directory {path} missing")
    die_if(not os.path.exists(listing_path), f"Dataset listing {listing_path} missing")

    # Check that we can load the JSON file.
    with open(listing_path, "r") as handle:
        try:
            listing = json.load(handle)
        except json.decoder.JSONDecodeError as err:
            print(f"ERROR: malformed JSON in {listing_path}, '{err}'")

    # Check that every set is non-empty and sets are non-overlapping.
    seen_files = set()
    for key in (TRAIN_KEY, VALID_KEY, TEST_KEY):
        die_if(key not in listing, f"Missing key '{key}' in {listing_path}")
        files = set()
        for glob_path in listing[key]:
            files.update(glob.glob(os.path.join(path, glob_path)))

        die_if(not files, f"Globs for key '{key}' are empty")
        for filename in files:
            die_if(filename in seen_files, f"File {filename} appears in multiple sets")

        seen_files.update(files)

    # Check that all matched files are WAV files.
    for filename in seen_files:
        die_if(not filename.endswith(".wav"), f"{filename} is missing .wav extension")

    print(f"Successfully verified dataset at {path}. All checks passed.")


@click.group()
def dataset() -> None:
    """
    WaveRNN dataset commands.
    """


@dataset.command("download")
@click.argument("name", type=click.Choice(list(DATASETS)))
@click.option("--destination", required=True, help="Directory to download to")
def cmd_download(  # pylint: disable=missing-param-doc
    name: str, destination: str
) -> None:
    """
    Download a dataset.
    """
    if name == NAME_LJSPEECH:
        download_ljspeech(destination)

    verify_dataset(destination)


@dataset.command("verify")
@click.argument("path")
def cmd_verify(path: str) -> None:  # pylint: disable=missing-param-doc
    """
    Verify a dataset.
    """
    verify_dataset(path)
