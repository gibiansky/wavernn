"""
Main entrypoint for WaveRNN package.
"""
import click

import wavernn.dataset
import wavernn.train
import wavernn.infer


@click.group()
def main() -> None:
    """
    WaveRNN neural vocoder.
    """


main.add_command(wavernn.dataset.dataset)
main.add_command(wavernn.train.train)
main.add_command(wavernn.infer.infer)
