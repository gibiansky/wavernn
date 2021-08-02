"""
Main entrypoint for WaveRNN package.
"""
import click

import wavernn.dataset

# import wavernn.train


@click.group()
def main() -> None:
    """
    WaveRNN neural vocoder.
    """


main.add_command(wavernn.dataset.dataset)
