"""
Utility functions for WaveRNN.
"""
import os
import shlex
import subprocess
import sys

import requests
import tqdm  # type: ignore


def download(url: str, target: str) -> str:
    """
    Downloads a file.

    Args:
      url: The URL of the file to download.
      target: Where to download file to.

    Returns:
      Path to the downloaded file.
    """
    _, basename = url.rsplit("/", 1)
    path = os.path.join(target, basename)

    print(f"Downloading {url} to {target} ...")
    with requests.get(url, stream=True) as response:
        total_bytes = int(response.headers.get("content-length", 0))
        with open(path, "wb") as handle:
            # Set up progress bar for long downloads.
            with tqdm.tqdm(
                desc=basename,
                total=total_bytes,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as progress:
                for data in response.iter_content(chunk_size=1024):
                    progress.update(handle.write(data))

    return path


def die_if(condition: bool, message: str) -> None:
    """
    Die if a condition is met. Prints an error message.

    Args:
      condition: Condition for termination.
      message: Message to print upon termination.
    """
    if condition:
        print("ERROR:", message)
        sys.exit(1)


def cmd(*args: str) -> None:
    """
    Run a subprocess. If it fails, exit.

    Args:
      args: Command to run.
    """
    print(f"Running command '{shlex.join(args)}' ...")
    subprocess.check_output(args)
