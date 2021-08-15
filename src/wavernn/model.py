"""
WaveRNN model definition.
"""
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, NamedTuple, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import MISSING
from torch import Tensor

from wavernn.dataset import AudioSample, DataConfig
from wavernn.prune import PruneConfig, prune
from wavernn.util import die_if

# Key under which to log validation loss.
VALIDATION_LOSS_KEY: str = "validation_loss"

# Key under which to log train loss.
TRAIN_LOSS_KEY: str = "train_loss"


@dataclass
class ConditionerConfig:
    """Configuration for WaveRNN conditioner."""

    # Number of convolutional layers to use.
    layers: int = MISSING

    # How many output channels each convolutional layer should have.
    channels: int = MISSING

    # The kernel width for each of the convolutional layers.
    width: int = MISSING

    # How much to shift (subtract) the input mels by prior to scaling and convolving.
    normalization_shift: float = MISSING

    # How much to scale (divide) the input mels by prior to convolving after shifting.
    normalization_scale: float = MISSING


class Conditioner(torch.nn.Module):
    """The conditioning network for WaveRNN."""

    def __init__(self, config: ConditionerConfig, n_mels: int) -> None:
        """Create a new conditioning network.

        Args:
          config: Configuration for this network.
          n_mels: How many input channels the data will have.
        """
        super().__init__()

        layers: list[torch.nn.Module] = []
        for idx in range(config.layers):
            input_channels = n_mels if idx == 0 else config.channels
            layers.extend(
                (
                    torch.nn.Conv1d(input_channels, config.channels, config.width),
                    torch.nn.Tanh(),
                )
            )

        self.model: torch.nn.Module = torch.nn.Sequential(*layers)

        self.normalization_shift: float = config.normalization_shift
        self.normalization_scale: float = config.normalization_scale

    def forward(self, mels: Tensor) -> Tensor:
        """Run the network.

        Args:
          mels: Input spectrograms of shape [batch, n_mels, input_timesteps].

        Returns:
          Network outputs of shape [batch, channels, output_timesteps].
        """
        return self.model((mels - self.normalization_shift) / self.normalization_scale)


@dataclass
class AutoregressiveConfig:
    """Configuration for WaveRNN autoregressive network."""

    # State dimension for the GRU.
    gru_dimension: int = MISSING

    # Output dimension of the linear layer after the GRU.
    hidden_dimension: int = MISSING


class AutoregressiveRNN(torch.nn.Module):
    """The autoregressive network for WaveRNN."""

    def __init__(self, config: AutoregressiveConfig, input_channels: int) -> None:
        """Create a new autoregressive RNN.

        Args:
          config: Configuration for this network.
          input_channels: How many channels the conditioning network outputs.
        """
        super().__init__()

        self.gru: torch.nn.Module = torch.nn.GRU(
            input_size=input_channels,
            hidden_size=config.gru_dimension,
            num_layers=1,
            batch_first=True,
        )
        self.post_gru: torch.nn.Module = torch.nn.Linear(
            config.gru_dimension, config.hidden_dimension
        )

    def forward(
        self, x: Tensor, state: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """Run the network.

        Args:
          x: Input tensor of shape [batch, timesteps, input_channels].
          state: Optional initial state of GRU.

        Returns:
          Tuple of tensors containing:
            - Network outputs of shape [batch, timesteps, hidden_dimension].
            - Final state of shape [batch, gru_dimension].
        """
        x, state = self.gru(x, state)
        assert state is not None
        return torch.nn.functional.relu(self.post_gru(x)), state


@dataclass
class OutputConfig:
    """Configuration for the input and output domain."""

    # What type of input / output domain to use.
    # Allowed values:
    #   - "discretized-mu-law": Standard WaveRNN with one discretized prediction.
    domain: str = "discretized-mu-law"

    # How many buckets to discretize audio sample into.
    buckets: int = MISSING

    # Mu to use for mu law companding and expanding.
    mu: float = MISSING


class DiscretizedMuLaw(torch.nn.Module):
    """A discretized mu-law input and output domain."""

    def __init__(
        self, config: OutputConfig, embedding_channels: int, input_channels: int
    ) -> None:
        """Create a new input and output domain.

        Args:
          config: Configuration for this input and output domain.
          input_channels: How many inputs are produced by the autoregressive network.
        """
        super().__init__()

        assert config.domain == "discretized-mu-law", "Invalid domain type"

        self.mu: float = config.mu
        self.buckets: int = config.buckets
        self.embedding: torch.nn.Module = torch.nn.Embedding(
            self.buckets, embedding_channels
        )
        self.linear: torch.nn.Module = torch.nn.Linear(input_channels, self.buckets)

    def quantize(self, waveform: Tensor) -> Tensor:
        """Quantize a waveform with mu-law encoding and quantization.

        Args:
          waveform: The float32 waveform to quantize.

        Returns:
          A long tensor of the same shape was `waveform`.
        """
        if not waveform.is_floating_point():
            waveform = waveform.to(torch.float)
        mu = torch.tensor(self.mu, dtype=waveform.dtype)
        x_mu = (
            torch.sign(waveform)
            * torch.log1p(mu * torch.abs(waveform))
            / torch.log1p(mu)
        )
        x_mu = ((x_mu + 1) / 2 * (self.buckets - 1) + 0.5).to(torch.int64)
        return x_mu

    def dequantize(self, waveform: Tensor) -> Tensor:
        """Dequantize a waveform with mu-law encoding and quantization.

        Args:
          waveform: The int64 quantized waveform.

        Returns:
          A float32 tensor of the same shape was `waveform`.
        """
        if not waveform.is_floating_point():
            waveform = waveform.to(torch.float)
        mu = torch.tensor(self.mu, dtype=waveform.dtype)
        x = (waveform / (self.buckets - 1)) * 2 - 1.0
        x = torch.sign(x) * (torch.exp(torch.abs(x) * torch.log1p(mu)) - 1.0) / mu
        return x

    def embed(self, waveform: Tensor) -> Tensor:
        """
        Create a waveform embedding tensor.

        Args:
          waveform: Input waveform of shape [batch, num_samples].

        Returns:
          Embedding of shape [batch, num_samples, embedding_channels].
        """
        return self.embedding(self.quantize(waveform))

    def sample(self, hidden: Tensor) -> Tensor:
        """
        Sample from this domain.

        Args:
          hidden: Output of the autoregressive layer of shape.

        Returns:
          Sample of shape [1].
        """
        logits = self.linear(hidden)
        distribution = torch.nn.functional.softmax(logits, dim=0)
        return torch.multinomial(distribution, num_samples=1)

    def loss(self, hidden: Tensor, target: Tensor) -> Tensor:
        """
        Compute the final prediction outputs from the final hidden layer and
        then compute loss against the target waveform.

        The input activations and target waveform samples must be aligned; that
        is, for an autoregressive network, the timesteps need to already be
        shifted by one to make the network properly autoregressive.

        Args:
          hidden: Input activations of shape [batch, num_samples, input_channels].
          target: Target waveform of shape [batch, num_samples].

        Returns:
          The average loss of the prediction.
        """
        logits = self.linear(hidden).transpose(1, 2)
        quantized_targets = self.quantize(target)
        return torch.nn.functional.cross_entropy(logits, quantized_targets)


@dataclass
class OptimizerConfig:
    """Configuration for the optimizer."""

    learning_rate: float = MISSING
    decay_rate: float = MISSING
    decay_iterations: list[float] = MISSING


@dataclass
class Config:
    """
    Configuration for a WaveRNN model.
    """

    data: DataConfig = MISSING
    conditioner: ConditionerConfig = MISSING
    autoregressive: AutoregressiveConfig = MISSING
    output: OutputConfig = MISSING
    optimizer: OptimizerConfig = MISSING
    prune: PruneConfig = MISSING


class ModelWeights(NamedTuple):
    """
    Weight and bias matrices for WaveRNN.
    """

    gru_weight_ih: Tensor
    gru_weight_hh: Tensor
    gru_bias_ih: Tensor
    gru_bias_hh: Tensor
    hidden_weight: Tensor
    hidden_bias: Tensor
    sample_embeddings: Tensor
    output_weight: Tensor
    output_bias: Tensor


class Model(pl.LightningModule):
    """
    A WaveRNN model.
    """

    def __init__(self, config: Config) -> None:
        """
        Create a new model.

        Args:
          config: Configuration for the model.
        """

        super().__init__()
        self.config = config

        self.conditioner = Conditioner(config.conditioner, config.data.mel.n_mels)
        self.upsample = torch.nn.Upsample(scale_factor=config.data.mel.hop_length)
        self.autoregressive = AutoregressiveRNN(
            config.autoregressive, config.conditioner.channels
        )
        self.domain = DiscretizedMuLaw(
            config.output,
            config.conditioner.channels,
            config.autoregressive.hidden_dimension,
        )

    def configure_optimizers(self) -> torch.optim.Adam:
        """
        Create optimizers for this model.

        Returns:
          A properly-configured optimizer.
        """
        return torch.optim.Adam(
            self.parameters(), lr=self.config.optimizer.learning_rate
        )

    def training_step(  # type: ignore # pylint: disable=unused-argument
        self, batch: AudioSample, batch_idx: int
    ) -> Tensor:
        """
        Compute loss for one training step.

        Args:
          batch: Input waveforms and spectrograms.
          batch_idx: Unused.

        Returns:
          The computed loss for this training batch.
        """
        loss = self.loss(batch)
        self.log(TRAIN_LOSS_KEY, loss)
        return loss

    def validation_step(  # type:ignore # pylint: disable=unused-argument
        self, batch: AudioSample, batch_idx: int
    ) -> Tensor:
        """
        Compute loss for one validation step.

        Args:
          batch: Input waveforms and spectrograms.
          batch_idx: Unused.

        Returns:
          The computed loss for this training batch.
        """
        loss = self.loss(batch)
        self.log(VALIDATION_LOSS_KEY, loss)
        return loss

    @torch.no_grad()
    def on_train_batch_start(
        self, *args: Any  # pylint: disable=unused-argument
    ) -> None:
        """
        Called before each training batch.

        Used to set learning rate.

        Args:
          args: Unused arguments we don't need.
        """
        opt = self.config.optimizer
        lr = opt.learning_rate
        for it in opt.decay_iterations:
            if self.global_step >= it:
                lr *= opt.decay_rate

        optimizer = self.optimizers().optimizer  # type: ignore
        for pg in optimizer.param_groups:
            pg["lr"] = lr

    @torch.no_grad()
    def on_train_batch_end(self, *args: Any) -> None:  # pylint: disable=unused-argument
        """
        Called after each training batch.

        Used to perform pruning.

        Args:
          args: Unused arguments we don't need.
        """
        weights = self.weights()
        sparse_matrices = [
            weights.output_weight,
            weights.gru_weight_hh,
            weights.hidden_weight,
        ]
        prune(
            config=self.config.prune,
            parameters=sparse_matrices,
            step=self.global_step,
        )

        # Log sparsity fraction occasionally.
        if self.global_step % 100 == 0:
            nonzero_params = torch.tensor(0.0)
            total_params = torch.tensor(0.0)
            for matrix in sparse_matrices:
                total_params += matrix.numel()
                nonzero_params += torch.sum(matrix.abs() > 0.0).cpu()
            self.log("remaining_fraction", nonzero_params / total_params)

    def weights(self) -> ModelWeights:
        """
        Extract the weights from this model.

        Returns:
          A ModelWeights tuple containing all the labeled model weights.
        """
        return ModelWeights(
            gru_weight_ih=self.autoregressive.gru.weight_ih_l0,  # type: ignore
            gru_weight_hh=self.autoregressive.gru.weight_hh_l0,  # type: ignore
            gru_bias_ih=self.autoregressive.gru.bias_ih_l0,  # type: ignore
            gru_bias_hh=self.autoregressive.gru.bias_hh_l0,  # type: ignore
            hidden_weight=self.autoregressive.post_gru.weight,  # type: ignore
            hidden_bias=self.autoregressive.post_gru.bias,  # type: ignore
            sample_embeddings=self.domain.embedding.weight,  # type: ignore
            output_weight=self.domain.linear.weight,  # type: ignore
            output_bias=self.domain.linear.bias,  # type: ignore
        )

    def loss(self, batch: AudioSample) -> Tensor:
        """
        Compute loss for one training or validation step.

        Args:
          batch: Input waveforms and spectrograms.

        Returns:
          The computed loss for this training batch.
        """
        hidden = self.conditioner(batch.spectrogram)

        # Check that sizes match. If they don't, it may be a poorly set
        # padding_frames value, and we warn the user.
        num_frames = hidden.shape[2]
        num_samples = batch.waveform.shape[1]
        hop_length = self.config.data.mel.hop_length
        die_if(
            num_frames * hop_length != num_samples,
            f"Number of frames ({num_frames}) in sample does not match "
            f"number of samples ({num_samples}) times hop length "
            f"({hop_length}). Is padding_frames set correctly?",
        )

        # Run the autoregressive chunk of the network.
        hidden = self.upsample(hidden).transpose(1, 2)
        hidden += self.domain.embed(batch.waveform)
        predictions, _ = self.autoregressive(hidden)

        return self.domain.loss(predictions[:, :-1, :], batch.waveform[:, 1:])

    def pytorch_inference(
        self, conditioning: Tensor, prev_sample: Tensor, init_state: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Run slow autoregressive inference implemented in Python.

        Args:
          conditioning: The conditioning information of shape [num_frames, channels].
          prev_sample: A tensor of shape [1] containing the last generated sample.
          init_state: A tensor of shape [gru_dimension] containing the last state.

        Returns:
          A tuple of tensors containing:
            - the output samples
            - the final generated sample
            - the final GRU state
        """
        hop_length = self.config.data.mel.hop_length
        num_frames = conditioning.size(0)
        num_samples = hop_length * num_frames
        outputs = torch.zeros((num_samples,), dtype=torch.int64)
        state = init_state[None, None, :]
        for timestep in range(num_samples):
            frame_idx = timestep // hop_length
            gru_input = conditioning[
                frame_idx : frame_idx + 1, :
            ] + self.domain.embedding(prev_sample)
            hidden, state = self.autoregressive(gru_input.unsqueeze(0), state=state)
            prev_sample = self.domain.sample(hidden.flatten())
            outputs[timestep] = prev_sample[0]
        return outputs, prev_sample, state.flatten()

    @torch.no_grad()
    def infer(
        self, spectrograms: Iterable[Tensor], native: bool = True, timing: bool = False
    ) -> Iterator[np.ndarray]:
        """Run inference to generate audio.

        Args:
          spectrograms: An iterable of spectrograms. These should be
              overlapping appropriately so that running through the conditioner
              yields non-overlapping chunks that cover the input sequence. The
              easiest way to get this is to load an AudioDataset (as if training)
              and extract the samples that way.
          native: Whether to use native inference (C++) or Pytorch inference (Python).
          timing: Whether to print timing information from the kernel.

        Yields:
          Generated audio samples, one per input spectrogram. Output arrays
          are of shape [num_samples] and are float32 in the range [-1, 1].
        """
        # Ensure the C++ library is loaded.
        torch.ops.load_library("build/lib.linux-x86_64-3.9/wavernn_kernel.so")

        # Extract hyperparameters and weights from the model.
        hop_length = self.config.data.mel.hop_length
        weights = self.weights()

        final_sample = torch.full((1,), 128, dtype=torch.int64)
        final_state = torch.zeros((self.config.autoregressive.gru_dimension))

        sample_activations = torch.mm(
            weights.sample_embeddings, weights.gru_weight_ih.t()
        )
        for spectrogram in spectrograms:
            hidden = self.conditioner(spectrogram.unsqueeze(0))
            conditioning = hidden.squeeze(0).transpose(0, 1)
            if native:
                activations_ih = torch.addmm(
                    weights.gru_bias_ih, conditioning, weights.gru_weight_ih.t()
                )
                samples = torch.ops.wavernn.wavernn_inference(
                    activations_ih,
                    final_sample,
                    final_state,
                    sample_activations,
                    weights.gru_weight_hh,
                    weights.gru_bias_hh,
                    weights.hidden_weight,
                    weights.hidden_bias,
                    weights.output_weight,
                    weights.output_bias,
                    hop_length,
                    timing,
                )
            else:
                samples, final_sample, final_state = self.pytorch_inference(
                    conditioning, final_sample, final_state
                )
            yield self.domain.dequantize(samples).numpy()
