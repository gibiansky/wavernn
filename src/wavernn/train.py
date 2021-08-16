"""
Training entrypoint for WaveRNN.
"""
import os
import shutil
from typing import Optional

import click
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger

from wavernn.dataset import AudioDataModule
from wavernn.model import VALIDATION_LOSS_KEY, Config, Model
from wavernn.util import die_if

# Constants related to the model directory organization.
#
# A model directory contains everything associated with a single model. This
# includes the model config (config.yaml), the checkpoints directory
# (checkpoints) with the best checkpoint (best.ckpt) in it, a logs directory
# for Tensorboard logs, and anything else the model may need. All operations
# with models are done by passing a --path argument pointing to a model directory.

# Name of the config file to store.
CONFIG_PATH: str = "config.yaml"

# Where to store checkpoints.
CHECKPOINTS_DIR: str = "checkpoints"

# Where to write best checkpoint.
BEST_CHECKPOINT: str = "best"


@click.command("train")
@click.option(
    "--config", type=click.Path(exists=True, dir_okay=False), help="YAML model config"
)
@click.option(
    "--path", required=True, type=click.Path(file_okay=False), help="Model directory"
)
@click.option(
    "--data", required=True, type=click.Path(file_okay=False), help="Dataset directory"
)
@click.option(
    "--test-every", default=5000, help="How often to run validation during training"
)
@click.argument("overrides", multiple=True, help="Dotlist option overrides")
def train(  # pylint: disable=missing-param-doc
    config: Optional[str], path: str, data: str, test_every: int, overrides: list[str]
) -> None:
    """
    Train a WaveRNN.
    """
    die_if(
        config is None and not os.path.exists(path),
        f"Since --config is not passed, directory {path} must exist",
    )

    # If this is the first time a model is being trained, create its directory
    # and populate it with a config file. Otherwise, use the existing
    # directory and the existing config file.
    saved_config_path = os.path.join(path, CONFIG_PATH)
    if config is None or os.path.exists(saved_config_path):
        config = saved_config_path
        die_if(not os.path.exists(config), f"Missing config file {config}")
    else:
        os.makedirs(path, exist_ok=True)
        shutil.copyfile(config, saved_config_path)

    # Create a model with the config.
    model_config: Config = OmegaConf.structured(Config)
    model_config.merge_with(OmegaConf.load(config))  # type: ignore
    model_config.merge_with_dotlist(overrides)  # type: ignore
    model = Model(model_config)

    # Load the dataset from the config.
    data_module = AudioDataModule(data, model_config.data)

    best_path = os.path.join(path, CHECKPOINTS_DIR, BEST_CHECKPOINT + ".ckpt")
    if os.path.exists(best_path):
        model = Model.load_from_checkpoint(best_path, config=model_config)
    else:
        # If this model has never been initialized before, compute the input
        # stats from the dataset. The input stats are used for normalizing the
        # input features. Doing this on the first run makes our model less
        # error-prone, as it is impossible to set an incorrect feature
        # normalization.
        model = Model(model_config)
        data_module.setup()
        model.initialize_input_stats(data_module.train_dataloader())

    # Train the model.
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=VALIDATION_LOSS_KEY,
        dirpath=os.path.join(path, CHECKPOINTS_DIR),
        filename=BEST_CHECKPOINT,
        mode="min",
        save_last=True,
    )
    logger = TensorBoardLogger(save_dir=path, version="logs", name=None)
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        val_check_interval=test_every,
        logger=logger,
        gpus=1,
    )
    trainer.fit(model, data_module)
