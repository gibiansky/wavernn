"""
Pruning utilities for WaveRNN training.
"""
from dataclasses import dataclass
from typing import Tuple

import torch
from omegaconf import MISSING


@dataclass
class PruneConfig:
    """
    Configuration indicating how to prune throughout training.
    """

    # What fraction of the weights to drop after pruning.
    final_sparsity: float = MISSING

    # How many iterations to wait to start pruning.
    start_iteration: int = MISSING

    # By which iteration should pruning finish?
    stop_iteration: int = MISSING

    # Block shape for block sparisty.
    block: Tuple[int, int] = MISSING


@torch.no_grad()
def prune(config: PruneConfig, parameters: list[torch.Tensor], step: int) -> None:
    """
    Prune parameters according to the provided configuration and update them in-place.

    Args:
      config: The pruning configuration.
      parameters: The parameters to prune.
      step: The current step of training.
    """
    if step <= config.start_iteration:
        return

    # Compute the fraction to prune right now.
    if step >= config.stop_iteration:
        fraction = config.final_sparsity
    else:
        prune_iterations = config.stop_iteration - config.start_iteration
        remaining = (config.stop_iteration - step) / prune_iterations
        fraction = config.final_sparsity * (1 - remaining ** 3)

    # Compute the magnitude of each block of each parameter.
    parameter_block_magnitudes = [
        torch.nn.functional.max_pool2d(
            parameter.abs().unsqueeze(0).unsqueeze(0),
            kernel_size=tuple(config.block),
            stride=tuple(config.block),
        )
        .squeeze(0)
        .squeeze(0)
        for parameter in parameters
    ]

    # Find the pruning threshold.
    all_magnitudes = torch.cat([mags.flatten() for mags in parameter_block_magnitudes])
    num_remaining_blocks = int(all_magnitudes.numel() * (1 - fraction))
    prune_threshold = torch.topk(
        all_magnitudes, num_remaining_blocks, sorted=False, largest=True
    ).values.min()

    for param, block_magnitudes in zip(parameters, parameter_block_magnitudes):
        block_mask = (block_magnitudes >= prune_threshold).to(torch.float32)
        param_mask = torch.repeat_interleave(
            torch.repeat_interleave(block_mask, config.block[0], dim=0),
            config.block[1],
            dim=1,
        )
        param *= param_mask
