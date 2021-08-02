"""
Dataset-handling entrypoint for WaveRNN package.
"""
import glob
import os
import json

import click

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
    cmd("tar", "--directory", destination, "-xzf", path)

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
