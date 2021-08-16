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

    # What fraction of the weights to drop after pruning. The sparsity is
    # gradually ramped up starting from zero during a warmup period and
    # eventually reaches the value indicated by this configuration option. For
    # example, if this value is 0.95, then approximately 95% of the weights in
    # the final model parameters will be zeros.
    final_sparsity: float = MISSING

    # How many iterations to wait to start pruning.
    start_iteration: int = MISSING

    # By which iteration should pruning finish?
    stop_iteration: int = MISSING

    # Block shape for block sparsity.
    block: Tuple[int, int] = MISSING


def prune_fraction(step: int, config: PruneConfig) -> float:
    """
    Compute what fraction of weights should be pruned. The fraction is zero for
    a warmup period and then ramps up gradually until it reaches a final max
    value, after which it is constant for the rest of training.

    Args:
      step: The current training step.
      config: The pruning configuration for the model.

    Returns:
      The fraction of weights that should be set to zero after this step.
    """
    if step <= config.start_iteration:
        return 0.0

    # Compute the fraction to prune right now.
    if step >= config.stop_iteration:
        return config.final_sparsity

    prune_iterations = config.stop_iteration - config.start_iteration
    remaining = (config.stop_iteration - step) / prune_iterations
    return config.final_sparsity * (1 - remaining ** 3)


@torch.no_grad()
def prune(config: PruneConfig, parameters: list[torch.Tensor], step: int) -> None:
    """
    Prune parameters according to the provided configuration and update them
    in-place.

    Pruning fraction starts at zero (a dense model) and stays there for an
    initial warmup period. After the warmup period, the pruning fraction
    gradually ramps up to its final value using a cubic interpolation curve.
    Once it reaches its final value, it remains there for the rest of training.

    Args:
      config: The pruning configuration.
      parameters: The parameters to prune. These parameters must be 2D tensors.
      step: The current step of training.
    """
    fraction = prune_fraction(step, config)
    if fraction == 0:
        return

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
