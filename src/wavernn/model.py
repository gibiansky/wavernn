"""
WaveRNN model definition.
"""
import os
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import scipy.signal  # type: ignore
import torch
from omegaconf import MISSING
from torch import Tensor

from wavernn.dataset import (
    AudioDataset,
    AudioSample,
    DataConfig,
    MelConfig,
    MultibandConfig,
)
from wavernn.prune import PruneConfig, prune
from wavernn.util import die_if, load_extension_module

# Key under which to log validation loss.
VALIDATION_LOSS_KEY: str = "validation_loss"

# Key under which to log train loss.
TRAIN_LOSS_KEY: str = "train_loss"

# Number of frames needed to compute normalization statistics.
NORMALIZATION_FRAMES: int = 100_000


@dataclass
class ConditionerConfig:
    """Configuration for WaveRNN conditioner."""

    # WaveRNN consists of a conditioner network and an autoregressive network.
    # The conditioner network takes as input log mel spectrograms (or other
    # low-frequency acoustic features) and processes them with a fast neural
    # network to generate conditioning inputs for the autoregressive network.
    # These inputs guide the autoregressive network to produce speech matching
    # the input features.
    #
    # This implementation of WaveRNN uses a simple stack of 1D convolutional
    # layers with interspersed nonlinearities as a conditioner network.

    # Number of convolutional layers to use.
    layers: int = MISSING

    # How many output channels each convolutional layer should have.
    channels: int = MISSING

    # The kernel width for each of the convolutional layers.
    width: int = MISSING


class Conditioner(torch.nn.Module):
    """The conditioner network for WaveRNN."""

    def __init__(self, config: ConditionerConfig, n_mels: int) -> None:
        """
        Create a new conditioner network.

        Args:
          config: Configuration for this network.
          n_mels: How many input channels the data will have.
        """
        super().__init__()

        layers: List[torch.nn.Module] = []
        for idx in range(config.layers):
            input_channels = n_mels if idx == 0 else config.channels
            layers.extend(
                (
                    torch.nn.Conv1d(input_channels, config.channels, config.width),
                    torch.nn.Tanh(),
                )
            )

        self.model: torch.nn.Module = torch.nn.Sequential(*layers)

        # The input minimum and maximum value. Must be computed from data and
        # set later. Will error in forward() if it's not set.
        self.register_buffer("input_range", torch.zeros(2))

    def set_input_range(self, low: float, high: float) -> None:
        """
        Set the input feature range. This be used for feature
        normalization prior to running the conditioner subnetwork.

        Args:
          low: The minimum expected feature value.
          high: The maximum expected feature value.
        """
        self.input_range[0] = low  # type: ignore
        self.input_range[1] = high  # type: ignore

    def forward(self, mels: Tensor) -> Tensor:
        """Normalize the input features and then run the network.

        Args:
          mels: Input spectrograms of shape [batch, n_mels, input_timesteps].

        Returns:
          Network outputs of shape [batch, channels, output_timesteps].
        """
        low = self.input_range[0]  # type: ignore
        high = self.input_range[1]  # type: ignore

        center = (low + high) / 2
        scale = (high - low) / 2
        assert scale.cpu().item() > 0

        normalized = (mels + center) / scale
        return self.model(normalized)


@dataclass
class AutoregressiveConfig:
    """Configuration for WaveRNN autoregressive network."""

    # The main part of WaveRNN is an autoregressive recurrent neural network
    # (RNN) which outputs a probability distribution over audio samples. The
    # input to the network is the conditioning information (from the
    # conditioner network) and the values of all past samples.
    #
    # The current autoregressive network for WaveRNN includes:
    #
    #   1. A sample embedding layer.
    #   2. A GRU RNN.
    #   3. A linear layer with a ReLU activation.
    #   4. A linear layer with a softmax activation.

    # State dimension for the GRU used in the autoregressive network.
    gru_dimension: int = MISSING

    # Output dimension of the linear layer after the GRU.
    hidden_dimension: int = MISSING


class AutoregressiveRNN(torch.nn.Module):
    """The autoregressive network for WaveRNN."""

    def __init__(self, config: AutoregressiveConfig, input_channels: int) -> None:
        """
        Create a new autoregressive RNN.

        Args:
          config: Configuration for this network.
          input_channels: How many channels the conditioner network outputs.
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

    # WaveRNN variants differ significantly by how they represent audio. For
    # example, the simplest variant uses the original representation from the
    # WaveNet paper, applying µ-law companding and quantization to the raw
    # audio signal and modeling it with a discrete distribution. Other variants
    # use a Mixture-of-Logistics (MoL) distribution, Gaussian distribution, or
    # Mixture-of-Gaussians (MoG) distribution to model the companded (but not
    # quantized) audio. Multi-band or linear predictive coding based WaveRNN
    # variants replace the audio with a different representation entirely.
    # Bit-bunching and sample-bunching allow sampling multiple times per
    # timestep of the RNN.
    #
    # In order to easily mix all these variants in the same codebase without
    # frustrating code duplication, we use the concept of an input / output
    # domain. The input / output domain defines how audio is represented, how
    # the probability distribution is predicted, and how to sample from that
    # probability distribution.

    # What type of input / output domain to use.
    # Allowed values:
    #   - "discretized-mu-law": Standard WaveRNN with one discretized prediction.
    #   - "multiband-discretized-mu-law": WaveRNN with multiple bands.
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
        """
        Create a new input and output domain.

        Args:
          config: Configuration for this input and output domain.
          embedding_channels: How many channels to output for sample embeddings.
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

        # Apply µ-law companding.
        mu = torch.tensor(self.mu, dtype=waveform.dtype)
        x_mu = (
            torch.sign(waveform)
            * torch.log1p(mu * torch.abs(waveform))
            / torch.log1p(mu)
        )

        # Quantize the resulting companded signal. If the number of buckets is
        # even, then the quantization will be symmetric around zero, with one
        # bucket being centered around zero.
        assert self.buckets % 2 == 0, "Expected even number of buckets"
        x_mu = ((x_mu + 1) / 2 * (self.buckets - 1)).to(torch.int64)

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

        # Dequantize the integer-valued signal. Since the middle bucket is
        # centered around zero, we need to add 0.5 to the bucket boundary in
        # order to properly treat zero. With this shift, zero can be quantized
        # and dequantized without error.
        x = ((waveform + 0.5) / (self.buckets - 1)) * 2 - 1.0

        mu = torch.tensor(self.mu, dtype=waveform.dtype)
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
        if torch.is_floating_point(waveform):
            waveform = self.quantize(waveform)
        return self.embedding(waveform)

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


class MultibandDiscretizedMuLaw(torch.nn.Module):
    """A multiband discretized mu-law input and output domain."""

    def __init__(
        self,
        config: OutputConfig,
        bands: int,
        embedding_channels: int,
        input_channels: int,
    ) -> None:
        """
        Create a new input and output domain.

        Args:
          config: Configuration for this input and output domain.
          bands: Number of subbands for audio prediction.
          embedding_channels: How many channels to output for sample embeddings.
          input_channels: How many inputs are produced by the autoregressive network.
        """
        super().__init__()

        assert config.domain == "discretized-mu-law", "Invalid domain type"

        self.bands: torch.nn.ModuleList = torch.nn.ModuleList(
            DiscretizedMuLaw(config, embedding_channels, input_channels)
            for _ in range(bands)
        )

    def dequantize(self, waveform: Tensor) -> Tensor:
        """Dequantize a waveform with mu-law encoding and quantization.

        Args:
          waveform: The int64 quantized waveform.

        Returns:
          A float32 tensor of the same shape was `waveform`.
        """
        return self.bands[0].dequantize(waveform)

    def embed(self, waveform: Tensor) -> Tensor:
        """
        Create a waveform embedding tensor.

        Args:
          waveform: Input waveform of shape [batch, num_samples, num_bands].

        Returns:
          Embedding of shape [batch, num_samples, embedding_channels].
        """
        embeddings = [
            band.embed(sub_waveform)
            for band, sub_waveform in zip(self.bands, waveform.unbind(-1))
        ]
        return torch.stack(embeddings).sum(dim=0)

    def sample(self, hidden: Tensor) -> Tensor:
        """
        Sample from this domain.

        Args:
          hidden: Output of the autoregressive layer of shape.

        Returns:
          Sample of shape [num_bands].
        """
        return torch.cat([band.sample(hidden) for band in self.bands])

    def loss(self, hidden: Tensor, target: Tensor) -> Tensor:
        """
        Compute the final prediction outputs from the final hidden layer and
        then compute loss against the target waveform.

        The input activations and target waveform samples must be aligned; that
        is, for an autoregressive network, the timesteps need to already be
        shifted by one to make the network properly autoregressive.

        Args:
          hidden: Input activations of shape [batch, num_samples, input_channels].
          target: Target waveform of shape [batch, num_samples, num_bands].

        Returns:
          The average loss of the prediction.
        """
        losses = [
            band.loss(hidden, sub_target)
            for band, sub_target in zip(self.bands, target.unbind(2))
        ]
        return torch.stack(losses).sum(dim=0)


@dataclass
class OptimizerConfig:
    """Configuration for the optimizer."""

    # An AdamOptimier is used for optimization. The learning rate is updated at
    # the specified iterations to decay it by the given rate.

    # Base learning rate for the optimizer.
    learning_rate: float = MISSING

    # When decaying, how much to multiply the learning rate by.
    # For example, a value of 0.1 will make the learning rate 10x smaller.
    decay_rate: float = MISSING

    # When to decay the learning rate. Measured in number of iterations.
    decay_iterations: List[float] = MISSING


@dataclass
class Config:
    """
    Configuration for a WaveRNN model.
    """

    # How to load data.
    data: DataConfig = MISSING

    # Configuration for the conditioner subnetwork.
    conditioner: ConditionerConfig = MISSING

    # Configuration for the autoregressive subnetwork.
    autoregressive: AutoregressiveConfig = MISSING

    # Configuration for the input / output domain.
    output: OutputConfig = MISSING

    # Configuration for the optimizer.
    optimizer: OptimizerConfig = MISSING

    # Configuration for pruning the WaveRNN weight matrices.
    prune: PruneConfig = MISSING


class ModelWeights(NamedTuple):
    """
    Weight and bias matrices for WaveRNN.
    """

    # Input-to-hidden weight matrix of the GRU.
    gru_weight_ih: Tensor

    # Hidden-to-hidden weight matrix of the GRU.
    gru_weight_hh: Tensor

    # Input-to-hidden bias vector of the GRU.
    gru_bias_ih: Tensor

    # Hidden-to-hidden bias vector of the GRU.
    gru_bias_hh: Tensor

    # GRU-output-to-hidden-layer weight matrix.
    hidden_weight: Tensor

    # GRU-output-to-hidden-layer bias vector.
    hidden_bias: Tensor

    # Hidden-layer-to-output-logits weight matrix.
    output_weights: List[Tensor]

    # Hidden-layer-to-output-logits bias vector.
    output_biases: List[Tensor]

    # Sample embeddings matrix (one embedding row per quantization value).
    sample_embeddings: List[Tensor]


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

        num_bands = config.data.multiband.bands
        upsample_factor = config.data.mel.hop_length // num_bands
        self.conditioner = Conditioner(config.conditioner, config.data.mel.n_mels)
        self.upsample = torch.nn.Upsample(scale_factor=upsample_factor)
        self.autoregressive = AutoregressiveRNN(
            config.autoregressive, config.conditioner.channels
        )
        if num_bands == 1:
            self.domain: Union[
                DiscretizedMuLaw, MultibandDiscretizedMuLaw
            ] = DiscretizedMuLaw(
                config.output,
                config.conditioner.channels,
                config.autoregressive.hidden_dimension,
            )
        else:
            self.domain = MultibandDiscretizedMuLaw(
                config.output,
                num_bands,
                config.conditioner.channels,
                config.autoregressive.hidden_dimension,
            )

    def configure_optimizers(self) -> torch.optim.Adam:
        """
        Create optimizers for this model.

        Returns:
          A properly-configured optimizer.
        """
        # This method is required by PyTorch Lightning.
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
        # This method is required by PyTorch Lightning.
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
        # This method is required by PyTorch Lightning.
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
        # This method is called by PyTorch Lightning.
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
        # This method is called by PyTorch Lightning.
        weights = self.weights()
        sparse_matrices = [
            *weights.output_weights,
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
        if isinstance(self.domain, MultibandDiscretizedMuLaw):
            sample_embeddings = [
                domain.embedding.weight for domain in self.domain.bands
            ]
            output_weights = [domain.linear.weight for domain in self.domain.bands]
            output_biases = [domain.linear.bias for domain in self.domain.bands]
        elif isinstance(self.domain, DiscretizedMuLaw):
            sample_embeddings = [self.domain.embedding.weight]
            output_weights = [self.domain.linear.weight]
            output_biases = [self.domain.linear.bias]
        else:
            raise ValueError("Unknown type for domain")

        return ModelWeights(
            gru_weight_ih=self.autoregressive.gru.weight_ih_l0,  # type: ignore
            gru_weight_hh=self.autoregressive.gru.weight_hh_l0,  # type: ignore
            gru_bias_ih=self.autoregressive.gru.bias_ih_l0,  # type: ignore
            gru_bias_hh=self.autoregressive.gru.bias_hh_l0,  # type: ignore
            hidden_weight=self.autoregressive.post_gru.weight,  # type: ignore
            hidden_bias=self.autoregressive.post_gru.bias,  # type: ignore
            output_weights=output_weights,  # type: ignore
            output_biases=output_biases,  # type: ignore
            sample_embeddings=sample_embeddings,  # type: ignore
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
        hop_length = self.config.data.mel.hop_length // self.config.data.multiband.bands
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

    @torch.no_grad()
    def initialize_input_stats(self, data: torch.utils.data.DataLoader) -> None:
        """
        Compute spectrogram mean and variance for each spectrogram band. These
        means and variances are then used during training to normalize the
        input features. Computing mean and variance from data gets us more
        precise values and avoids requiring the user to do it as a separate
        step.

        Args:
          data: The training data data loader.
        """
        print("Computing dataset feature statistics...")

        padding_frames = self.config.data.padding_frames
        n_mels = self.config.data.mel.n_mels

        # Collect sufficient frames to compute statistics.
        num_frames = 0
        frames = []
        for batch in data:
            spectrogram = batch.spectrogram[:, :, padding_frames:-padding_frames]
            spectrogram = spectrogram.transpose(1, 2).reshape((-1, n_mels))
            num_frames += spectrogram.shape[0]
            frames.append(spectrogram)
            if num_frames >= NORMALIZATION_FRAMES:
                break

        # Compute statistics and put them in the conditioner.
        spectrograms = torch.cat(frames, dim=0)
        low = spectrograms.min().item()
        high = spectrograms.max().item()
        self.conditioner.set_input_range(low, high)

    def pytorch_inference(
        self, conditioning: Tensor, prev_sample: Tensor, init_state: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Run slow autoregressive inference implemented in Python.

        Args:
          conditioning: The conditioning information of shape [num_frames, channels].
          prev_sample: A tensor of shape [num_bands] containing the last generated sample.
          init_state: A tensor of shape [gru_dimension] containing the last state.

        Returns:
          A tuple of tensors containing:
            - the output samples
            - the final generated sample
            - the final GRU state
        """
        num_bands = self.config.data.multiband.bands
        hop_length = self.config.data.mel.hop_length // num_bands
        num_frames = conditioning.size(0)
        num_samples = hop_length * num_frames
        outputs = torch.zeros((num_samples, num_bands), dtype=torch.int64)
        state = init_state[None, None, :]
        for timestep in range(num_samples):
            frame_idx = timestep // hop_length
            sample_embedding = self.domain.embed(prev_sample)
            gru_input = conditioning[frame_idx : frame_idx + 1, :] + sample_embedding
            hidden, state = self.autoregressive(gru_input.unsqueeze(0), state=state)
            prev_sample = self.domain.sample(hidden.flatten())
            outputs[timestep] = prev_sample.reshape((num_bands,))
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
        load_extension_module()

        # Extract hyperparameters and weights from the model.
        hop_length = self.config.data.mel.hop_length
        num_bands = self.config.data.multiband.bands
        weights = self.weights()

        final_sample = torch.full(
            (num_bands,), (self.config.output.buckets - 1) // 2, dtype=torch.int64
        )
        final_state = torch.zeros((self.config.autoregressive.gru_dimension))

        sample_activations = [
            torch.mm(sample_embeddings, weights.gru_weight_ih.t())
            for sample_embeddings in weights.sample_embeddings
        ]
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
                    weights.output_weights,
                    weights.output_biases,
                    hop_length // num_bands,
                    timing,
                )
            else:
                samples, final_sample, final_state = self.pytorch_inference(
                    conditioning, final_sample, final_state
                )
            yield self.domain.dequantize(samples).numpy()


class ExportableWaveRNN(torch.nn.Module):
    """
    A WaveRNN exported for inference.
    """

    def __init__(self, train_wavernn: Model) -> None:
        """
        Create a new exportable WaveRNN.

        Args:
          train_wavernn: The trained WaveRNN model.
        """
        super().__init__()
        load_extension_module()

        config = train_wavernn.config
        weights = train_wavernn.weights()

        # Create the initial state used by this WaveRNN.
        self.init_sample = torch.full(
            (1,), (config.output.buckets - 1) // 2, dtype=torch.int64
        )
        self.init_state = torch.zeros((config.autoregressive.gru_dimension))

        # Create the weights needed for autoregressive inference.
        self.sample_activations = torch.mm(
            weights.sample_embeddings, weights.gru_weight_ih.t()
        )
        self.gru_weight_hh = weights.gru_weight_hh
        self.gru_bias_hh = weights.gru_bias_hh
        self.hidden_weight = weights.hidden_weight
        self.hidden_bias = weights.hidden_bias
        self.output_weight = weights.output_weight
        self.output_bias = weights.output_bias
        self.hop_length = config.data.mel.hop_length
        self.gru_bias_ih = weights.gru_bias_ih
        self.gru_weight_ih_t = weights.gru_weight_ih.t()

        self.conditioner = train_wavernn.conditioner
        self.domain = train_wavernn.domain

        # Store model input and output config information.
        self.padding_frames = config.data.padding_frames
        self.sample_rate = config.data.mel.sample_rate
        self.n_fft = config.data.mel.n_fft
        self.n_mels = config.data.mel.n_mels
        self.fmin = config.data.mel.fmin
        self.fmax = config.data.mel.fmax
        self.hop_length = config.data.mel.hop_length
        self.win_length = config.data.mel.win_length
        self.log_epsilon = config.data.mel.log_epsilon
        self.pre_emphasis = config.data.mel.pre_emphasis

    @torch.jit.export
    @torch.no_grad()
    def synthesize(
        self, spectrogram: Tensor, state: Optional[Tuple[Tensor, Tensor]]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Run synthesis using the exported WaveRNN.

        Args:
          spectrogram: A spectrogram of shape [timesteps, n_mels].
          state: If passed, the previous state of the WaveRNN.
            State is a tuple of (final_sample, final_rnn_state).

        Returns:
          A tuple containing the synthesized data and the final state.
        """
        if state is None:
            sample = self.init_sample.clone()
            gru_state = self.init_state.clone()
        else:
            sample, gru_state = state

        # Run conditioner network.
        hidden = self.conditioner(spectrogram.unsqueeze(0))
        conditioning = hidden.squeeze(0).transpose(0, 1)
        activations_ih = torch.addmm(
            self.gru_bias_ih, conditioning, self.gru_weight_ih_t
        )

        # Run autoregressive network.
        output = torch.ops.wavernn.wavernn_inference(
            activations_ih,
            sample,
            gru_state,
            self.sample_activations,
            self.gru_weight_hh,
            self.gru_bias_hh,
            self.hidden_weight,
            self.hidden_bias,
            self.output_weight,
            self.output_bias,
            self.hop_length,
            False,
        )
        output = self.domain.dequantize(output)

        return output, (sample, gru_state)


class InferenceState(NamedTuple):
    """
    State needed during WaveRNN synthesis.
    """

    pre_emphasis_state: float
    model_state: Optional[Tuple[Tensor, Tensor]]


class InferenceWaveRNN:
    """
    A class capable of running inference with an exported WaveRNN.
    """

    def __init__(self, path: str, clip_frames: int) -> None:
        """
        Create a WaveRNN inference runner.

        Args:
          path: Path to an exported WaveRNN JIT file.
          clip_frames: How many frames to synthesize in each step.
        """
        load_extension_module()

        self.model = torch.jit.load(path)
        self.sample_rate = self.model.sample_rate
        self.data_config = DataConfig(
            clip_frames=clip_frames,
            padding_frames=self.model.padding_frames,
            batch_size=1,
            mel=MelConfig(
                sample_rate=self.sample_rate,
                n_fft=self.model.n_fft,
                n_mels=self.model.n_mels,
                fmin=self.model.fmin,
                fmax=self.model.fmax,
                hop_length=self.model.hop_length,
                win_length=self.model.win_length,
                log_epsilon=self.model.log_epsilon,
                pre_emphasis=self.model.pre_emphasis,
            ),
            multiband=MultibandConfig(
                bands=self.model.multiband_bands,
                taps=self.model.multiband_taps,
                cutoff_ratio=self.model.multiband_cutoff_ratio,
                beta=self.model.multiband_beta,
            ),
        )

    def load_clips_from_wav(self, input_file: str) -> List[AudioSample]:
        """
        Load a waveform from a WAV file and break it into clips.

        Used for copy-synthesis testing of a trained WaveRNN.

        Args:
          input_file: Path to a .wav file to load.

        Returns:
          A list of audio samples. Each audio samples has a 'spectrogram' and
          'waveform' field.
        """
        dataset = AudioDataset(
            os.path.dirname(input_file),
            [os.path.basename(input_file)],
            self.data_config,
        )
        return list(dataset.load_samples_from(input_file))

    @torch.no_grad()
    def synthesize(
        self,
        spectrogram: Union[Tensor, np.ndarray],
        state: Optional[InferenceState],
    ) -> Tuple[np.ndarray, InferenceState]:
        """
        Run synthesis using this WaveRNN.

        Args:
          spectrogram: A spectrogram of shape [n_mels, timesteps].
            Can be a PyTorch Tensor or a NumPy ndarray.
          state: An optional state for the WaveRNN.
            On first call, this should be left as None.

        Returns:
          A tuple containing the synthesized waveform and the state to pass to
          the next call of 'synthesize'.
        """
        # Initialize empty state.
        if state is None:
            state = InferenceState(pre_emphasis_state=0.0, model_state=None)

        # Convert spectrogram from NumPy to Torch if needed.
        if isinstance(spectrogram, np.ndarray):
            spectrogram = torch.from_numpy(spectrogram)

        synthesized, model_state = self.model.synthesize(spectrogram, state.model_state)
        waveform = synthesized.numpy()

        # Implement our own de-emphasis to properly track state.
        if self.model.pre_emphasis > 0:
            waveform, pre_emphasis_state = scipy.signal.lfilter(
                [1.0],
                [1.0, -self.model.pre_emphasis],
                waveform,
                zi=[state.pre_emphasis_state],
            )
            pre_emphasis_state = pre_emphasis_state[0]
        else:
            pre_emphasis_state = 0.0

        waveform = np.clip(waveform, -0.9999, 0.9999)
        return waveform, InferenceState(
            pre_emphasis_state=pre_emphasis_state, model_state=model_state
        )
