"""
Dataset preparation and loading for WaveRNN.
"""
import glob
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterator, List, NamedTuple, Optional

import click
import librosa  # type: ignore
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import MISSING

from wavernn.lpc import LPC, LinearPredictionConfig, MelConfig
from wavernn.pqmf import PQMF
from wavernn.util import cmd, die_if, download

# Where to store dataset information. A valid dataset, as far as this package
# is concerned, is a directory with a dataset.json file in it which points to
# WAV files stored inside that directory. See the list of required keys below.
DATASET_JSON: str = "dataset.json"

# Keys required in dataset.json. Each of the keys delineates a subset of the
# dataset as specified by a collection of globs. Globs are expanded using
# Python's glob.glob() function inside the dataset directory. For example, a
# reasonable data format would have a train/, validation/, and test/
# subdirectory, in which case the dataset.json would look like this:
#
#     {
#          "train": "train/*.wav",
#          "valid": "validation/*.wav",
#          "test": "test/*.wav"
#     }
TRAIN_KEY: str = "train"
VALID_KEY: str = "valid"
TEST_KEY: str = "test"

# Public-facing dataset names. These are used by the 'download' command.
NAME_LJSPEECH: str = "ljspeech"
NAME_VCTK: str = "vctk"
NAME_LIBRITTS: str = "libritts"

# List of LibriTTS subsets.
LIBRITTS_CLEAN_CHUNKS: List[str] = [
    "dev-clean.tar.gz",
    "test-clean.tar.gz",
    "train-clean-100.tar.gz",
    "train-clean-360.tar.gz",
]

# Dictionary mapping dataset to download URL. These datasets can be downloaded
# with the 'download' command. For example, to download LJSpeech, you can run:
#
#     wavernn dataset download ljspeech --destination ~/ljspeech
#
# This will download and unpack the dataset into ~/ljspeech.
DATASETS: Dict[str, str] = {
    NAME_LJSPEECH: "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
    NAME_VCTK: "http://www.udialogue.org/download/VCTK-Corpus.tar.gz",
    NAME_LIBRITTS: "https://www.openslr.org/resources/60/",
}


def download_ljspeech(destination: str) -> None:
    """
    Download and unpack LJSpeech dataset. Generate a dataset.json in the target
    directory with a reasonable train / validation / test split.

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

    # Generate dataset.json with train / validation / test split.
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


def download_vctk(destination: str) -> None:
    """
    Download and unpack VCTK dataset. Generate a dataset.json in the target
    directory with a reasonable train / validation / test split.

    Args:
      destination: Where to download dataset to.
    """
    if os.path.exists(destination):
        is_empty_dir = os.path.isdir(destination) and len(os.listdir(destination)) == 0
        die_if(not is_empty_dir, f"{destination} already exists")
    else:
        os.makedirs(destination)

    path = download(DATASETS[NAME_VCTK], destination)

    print("Extracting dataset (this may take a while)...")
    cmd("tar", "--directory", destination, "-xf", path)

    print("Removing compressed version...")
    os.unlink(path)

    # Generate dataset.json with train / validation / test split.
    listing_path = os.path.join(destination, DATASET_JSON)
    with open(listing_path, "w") as handle:
        json.dump(
            {
                TRAIN_KEY: [
                    f"VCTK-Corpus/wav48/*/*_{i:03d}.wav" for i in range(11, 503)
                ],
                VALID_KEY: [f"VCTK-Corpus/wav48/*/*_{i:03d}.wav" for i in range(4, 11)],
                TEST_KEY: [
                    "VCTK-Corpus/wav48/*/*_001.wav",
                    "VCTK-Corpus/wav48/*/*_002.wav",
                    "VCTK-Corpus/wav48/*/*_003.wav",
                ],
            },
            handle,
            indent=2,
        )


def download_libritts(destination: str) -> None:
    """
    Download and unpack the clean subsets of the LibriTTS dataset. Generate a dataset.json
    in the target directory with a reasonable train / validation / test split.

    Args:
      destination: Where to download dataset to.
    """
    if os.path.exists(destination):
        is_empty_dir = os.path.isdir(destination) and len(os.listdir(destination)) == 0
        die_if(not is_empty_dir, f"{destination} already exists")
    else:
        os.makedirs(destination)

    paths = [
        download(os.path.join(DATASETS[NAME_LIBRITTS], chunk), destination)
        for chunk in LIBRITTS_CLEAN_CHUNKS
    ]

    for chunk, path in zip(LIBRITTS_CLEAN_CHUNKS, paths):
        print(f"Extracting dataset {chunk} (this may take a while)...")
        cmd("tar", "--directory", destination, "-xf", path)

        print(f"Removing compressed version of {chunk}...")
        os.unlink(path)

    # Collect all the speakers in LibriTTS.
    speakers = []
    for chunk in LIBRITTS_CLEAN_CHUNKS:
        chunk = chunk.replace(".tar.gz", "")
        speakers.extend(glob.glob(os.path.join(destination, "LibriTTS", chunk, "*")))

    rand = random.Random(1234)
    speakers.sort()
    rand.shuffle(speakers)

    # Check the number of speakers.
    assert len(speakers) == 1230, f"Expected 1230 speakers, found {len(speakers)}"

    # Speakers to use for testing (and not training) and speakers to use for
    # training and/or validation. This is split to ensure there are some
    # speakers that are never seen for training.
    test_speakers = speakers[:10]
    train_speakers = speakers[10:]

    # Collect all known files.
    seen_speaker_files = [
        os.path.relpath(filename, destination)
        for speaker_dir in train_speakers
        for filename in glob.glob(os.path.join(speaker_dir, "*", "*.wav"))
    ]
    seen_speaker_files.sort()
    rand.shuffle(seen_speaker_files)

    # Add unseen speakers to test files.
    test_globs = [
        os.path.join(os.path.relpath(speaker_dir, destination), "*", "*.wav")
        for speaker_dir in test_speakers
    ]
    test_globs.extend(seen_speaker_files[0:200])

    valid_globs = seen_speaker_files[200:500]
    train_globs = seen_speaker_files[500:]

    test_globs.sort()
    valid_globs.sort()
    train_globs.sort()

    # Generate dataset.json with train / validation / test split.
    listing_path = os.path.join(destination, DATASET_JSON)
    with open(listing_path, "w") as handle:
        json.dump(
            {
                TRAIN_KEY: train_globs,
                VALID_KEY: valid_globs,
                TEST_KEY: test_globs,
            },
            handle,
            indent=2,
        )


def verify_dataset(path: str) -> None:
    """
    Check that a dataset is correctly formatted. A correctly formatted dataset
    is a directory with a dataset.json file which contains keys for training,
    validation, and testing datasets. The values are lists of globs referring
    to WAV files; the different sets of files must not overlap and must not be
    empty.

    If an error is detected, a message is printed and we exit with an error code.

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
    elif name == NAME_VCTK:
        download_vctk(destination)
    elif name == NAME_LIBRITTS:
        download_libritts(destination)

    verify_dataset(destination)


@dataset.command("verify")
@click.argument("path")
def cmd_verify(path: str) -> None:  # pylint: disable=missing-param-doc
    """
    Verify a dataset.
    """
    verify_dataset(path)


@dataset.command("list")
def cmd_list() -> None:
    """
    List available datasets.
    """
    for dset in DATASETS:
        print(dset)


@dataclass
class MultibandConfig:
    """
    Configuration for multi-band decomposition of training signal.
    """

    # How many bands to split into.
    bands: int = MISSING

    # Number of taps for the filter.
    taps: int = MISSING

    # The cutoff ratio hyperparameter for prototype filter design.
    cutoff_ratio: float = MISSING

    # Beta parameter for the Kaiser window.
    beta: float = MISSING


@dataclass
class DataConfig:
    """
    Configuration for a training dataset.
    """

    # Configuration for mel spectrogram extraction.
    mel: MelConfig = MISSING

    # Configuration for subband filtering.
    multiband: MultibandConfig = MISSING

    # Configuration for linear prediction.
    lpc: LinearPredictionConfig = MISSING

    # How many spectrogram frames to include in each audio sample.
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

    # How many clips to include in a batch.
    batch_size: int = MISSING


class AudioSample(NamedTuple):
    """
    A training sample or a batch of training samples.
    """

    # If only one subband, a float32 waveform tensor of shape [num_samples].
    # After batching, the shape will be [batch_size, num_samples].
    #
    # If subband modeling is enabled via multiband.bands > 1, then this will
    # instead of a float32 waveform tensor of shape [num_samples, num_bands].
    # After batching, the shape will be [batch_size, num_samples, num_bands].
    #
    # If linear prediction is enabled via lpc > 0, then this will be the
    # excitation signal. The linear prediction will be in 'prediction'.
    waveform: torch.Tensor

    # If linear prediction is enabled via lpc > 0, a float32 tensor containing
    # the linearly predicted waveform of shape [num_samples].
    # After batching, the shape will be [batch_size, num_samples].
    #
    # If multiband modeling is enabled via multiband.bands > 1, then this will
    # instead of a float32 waveform tensor of shape [num_samples, num_bands].
    # After batching, the shape will be [batch_size, num_samples, num_bands].
    #
    # If linear prediction is disabled, this will be a float32 tensor of zeros.
    prediction: torch.Tensor

    # A float32 mel spectrogram tensor of shape [n_mels, num_frames].
    # After batching, the shape will be [batch_size, n_mels, num_frames].
    spectrogram: torch.Tensor


class AudioDataset(torch.utils.data.IterableDataset):
    """
    An iterable PyTorch dataset of audio waveforms and mel spectrograms.
    """

    def __init__(
        self, path: str, globs: List[str], config: DataConfig, shuffle: bool = False
    ) -> None:
        """
        Create a new iterable dataset.

        Args:
          path: Path to the dataset directory.
          globs: The filename globs to use for the input data.
          config: The dataset configuration.
          shuffle: Whether to shuffle the files during iteration.
            Should be set to True for the training dataset.
        """
        super().__init__()

        self.config = config
        self.shuffle = shuffle
        self.pqmf = PQMF(
            config.multiband.bands,
            config.multiband.taps,
            config.multiband.cutoff_ratio,
            config.multiband.beta,
        )
        self.lpc = LPC(config.lpc, config.mel, self.pqmf)

        # Collect all files in this dataset.
        self.filenames: List[str] = []
        for glb in globs:
            self.filenames.extend(glob.glob(os.path.join(path, glb)))

    def __iter__(self) -> Iterator[AudioSample]:
        """
        Iterate over this dataset. If running with multiple data loading
        workers, iterate over an even split of the dataset.

        Yields:
          Audio samples containing mel spectrograms and waveforms. A single
          file will result in many generated samples, since each sample will be
          split into non-overlapping clips as indicated by the 'clip_frames'
          configuration value.
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

        if self.shuffle:
            random.shuffle(filenames)

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
            This option exists so that we can easily use this dataset to do data
            loading for purposes besides training, such as copy-synthesis.

        Yields:
          The audio samples in a file and mel spectrograms extracted from them,
          split into AudioSample tuples with 'clip_frames' spectrogram frames
          in each sample.
        """
        mel = self.config.mel
        multiband = self.config.multiband
        if clip_frames is None:
            clip_frames = self.config.clip_frames

        # Load waveform from disk.
        raw_waveform: np.ndarray
        sr: int
        raw_waveform, sr = librosa.load(filename, sr=mel.sample_rate)

        # Apply pre-emphasis on the raw data only if we aren't going to
        # decompose it into multiple subbands as well. If we *do* want to
        # decompose it, the pre-emphasis should be applied on the filtered
        # signal.
        if multiband.bands == 1 and mel.pre_emphasis > 0:
            raw_waveform = librosa.effects.preemphasis(
                raw_waveform, coef=mel.pre_emphasis
            )

        # Ensure samples are all in (-1, 1).
        # This happens very rarely on well-formatted raw audio but if you apply
        # pre-emphasis then it happens to a small fraction (0.01% or less) of a
        # small fraction (5-10%) of audio files.
        raw_waveform = np.clip(raw_waveform, -0.9999999, 0.9999999)

        # Pad waveform with silence at the start and end.
        #
        # Our spectrogram frames are non-centered, and we'd like the number of
        # samples in the waveform to be exactly the hop length times the number
        # of spectrogram frames we want our short-time Fourier transform to
        # generate. In order to do this, we compute the number of frames we
        # want, then pad the waveform so that its length is equal to
        #
        #   waveform_length = (num_frames - 1) * hop_length + win_length
        #            (Assuming win_length > hop_length.)
        #
        # Additionally, we want our spectrogram length to have extra padding
        # frames on the left and right. To achieve this we pad the waveform
        # with zero samples equal to the number of desired padding frames.
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

        # Compute the log mel spectrogram with a minimum value to avoid underflow.
        log_epsilon = torch.tensor(mel.log_epsilon, dtype=torch.float32)
        torch_spectrogram = torch.from_numpy(spectrogram)
        log_spectrogram = torch.log(torch_spectrogram + log_epsilon)

        # Now that we have computed the spectrogram on the fullband signal,
        # apply subband decomposition to generate the audio signal.
        if multiband.bands == 1:
            band_hop_length = mel.hop_length
        else:
            assert (
                mel.hop_length % multiband.bands == 0
            ), "# of bands must divide mel hop length"
            band_hop_length = mel.hop_length // multiband.bands
            torch_waveform = torch.from_numpy(padded_waveform.reshape((1, 1, -1)))
            analyzed = self.pqmf.analysis(torch_waveform).reshape((multiband.bands, -1))
            padded_waveform = analyzed.t().numpy()

            # Apply pre-emphasis to each band.
            if mel.pre_emphasis > 0:
                for b in range(multiband.bands):
                    padded_waveform[:, b] = librosa.effects.preemphasis(
                        padded_waveform[:, b], coef=mel.pre_emphasis
                    )

            # Ensure that after subband filtering and pre-emphasis, everything
            # is in a reasonable audio range.
            padded_waveform = np.clip(padded_waveform, -0.9999999, 0.9999999)

        if self.config.lpc == 0:
            padded_prediction = np.zeros_like(padded_waveform)
        else:
            padded_prediction = self.lpc.estimate(
                log_spectrogram.numpy(), padded_waveform
            )
            padded_waveform = np.clip(
                padded_waveform - padded_prediction, -0.9999999, 0.9999999
            )

        # Split the audio and spectrogram into chunks. Each chunk has
        # 'clip_frames' spectrogram frames and the corresponding samples.
        # Additionally, 'padding_frames' spectrogram frames are included on the
        # left and right of the spectrogram; these padding frames overlap with
        # the padding frames of previous and next generated chunks.
        for i in range(0, spectrogram.shape[1], clip_frames):
            desired_frames = clip_frames + 2 * padding_frames
            clip_spectrogram = log_spectrogram[:, i : i + desired_frames]
            if clip_spectrogram.shape[1] != desired_frames:
                continue

            start_sample = (i + padding_frames) * band_hop_length
            end_sample = start_sample + clip_frames * band_hop_length
            clip_waveform = torch.from_numpy(padded_waveform[start_sample:end_sample])
            clip_prediction = torch.from_numpy(
                padded_prediction[start_sample:end_sample]
            )
            if clip_waveform.shape[0] != clip_frames * band_hop_length:
                continue
            yield AudioSample(
                waveform=clip_waveform,
                spectrogram=clip_spectrogram,
                prediction=clip_prediction,
            )


class AudioDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module for audio data.
    """

    def __init__(self, path: str, config: DataConfig) -> None:
        """
        Create a new data module.

        Args:
          path: A path to the dataset directory.
          config: Configuration for the data module.
        """
        super().__init__()

        self.config = config

        # Load the dataset and its splits.
        listing_path = os.path.join(path, DATASET_JSON)
        with open(listing_path, "r") as handle:
            listing = json.load(handle)

        self.train_set = AudioDataset(path, listing[TRAIN_KEY], config, shuffle=True)
        self.valid_set = AudioDataset(path, listing[VALID_KEY], config)
        self.test_set = AudioDataset(path, listing[TEST_KEY], config)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Create a training data loader.

        Returns:
          A training data loader.
        """
        return torch.utils.data.DataLoader(
            self.train_set, batch_size=self.config.batch_size, num_workers=16
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Create a validation data loader.

        Returns:
          A validation data loader.
        """
        return torch.utils.data.DataLoader(
            self.valid_set, batch_size=self.config.batch_size, num_workers=8
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Create a test data loader.

        Returns:
          A test data loader.
        """
        return torch.utils.data.DataLoader(
            self.test_set, batch_size=self.config.batch_size
        )
