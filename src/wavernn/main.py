"""
Main entrypoint for WaveRNN package.
"""
import click

import wavernn.dataset
import wavernn.infer
import wavernn.train

# The WaveRNN package is intended to be used through a main command-line
# interface ("wavernn") driven by Click. Check out README.md for examples of
# how to use this command-line interface. In this module, we import all the
# subcommands and attach them to the top-level command. The top-level command
# is executed by the main script.


@click.group()
def main() -> None:
    """
    WaveRNN neural vocoder.
    """


# Most Click examples will use decorators to attach subcommands to groups.
# However, doing so requires jumping through hoops to import the group in each
# file that defines the subcommands, so we instead import all the subcommands
# here and use add_command() to group them together.

main.add_command(wavernn.dataset.dataset)
main.add_command(wavernn.train.train)
main.add_command(wavernn.infer.infer)
main.add_command(wavernn.infer.benchmark)
main.add_command(wavernn.infer.gemv)
