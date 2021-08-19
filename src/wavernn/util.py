"""
Utility functions for WaveRNN.
"""
import importlib
import os
import shlex
import subprocess
import sys

import requests
import torch
import tqdm  # type: ignore


def download(url: str, target: str) -> str:
    """
    Downloads a file while displaying a progress bar.

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
    Die if a condition is met.

    If so, prints an error message and exits with a non-zero exit code.

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


def load_extension_module() -> None:
    """
    Load the C++ extension module.

    We use setuptools to build the extension module. Normally, a C++ extension
    called 'x' built with setuptools can be loaded by just using 'import x';
    however, the PyTorch extension modules which define new ops don't actually
    define a valid Python module. (They don't have a PyInit_module_whatever
    function in them.)

    Instead of doing 'import x', we use the Python import machinery to find the
    op, but then we import it using PyTorch's shared object loader.
    """
    # Find the shared object using Python's standard import machinery.
    spec = importlib.util.find_spec("wavernn_kernel")
    assert spec is not None, "Could not import WaveRNN C++ kernel"
    path = spec.origin

    # Load it into this process.
    torch.ops.load_library(path)
