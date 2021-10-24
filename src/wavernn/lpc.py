"""
Utilities for linear prediction on waveforms.
"""

from dataclasses import dataclass

import librosa  # type: ignore
import numpy as np
import scipy  # type: ignore
import torch
from omegaconf import MISSING

from wavernn.pqmf import PQMF


@dataclass
class LinearPredictionConfig:
    """
    Configuration for linear prediction.
    """

    # Number of coefficients used for linear prediction.
    order: int = MISSING

    # Hop length for frames used for LPC estimation.
    hop_length: int = MISSING

    # Window length for frames used for LPC estimation.
    win_length: int = MISSING

    # How many iteration of Griffin-Lim to use to convert spectrogram to audio.
    griffinlim_iterations: int = MISSING


@dataclass
class MelConfig:
    """Configuration for log mel spectrogram extraction."""

    # Sample rate of the audio. If the audio is not this sample rate, it will
    # be up- or downsampled to this sample rate when it is loaded.
    sample_rate: int = MISSING

    # Number of Fourier coefficients in the spectrogram. This must be greater
    # than or equal to the window size specified by win_length.
    n_fft: int = MISSING

    # Number of bands in the mel spectrogram.
    n_mels: int = MISSING

    # Minimum frequency to include in the mel spectrogram.
    # A reasonable value is 0 to include all frequencies in the audio.
    fmin: float = MISSING

    # Maximum frequency to include in the mel spectrogram.
    # A reasonable value is sample_rate / 2 to include all frequencies in the audio.
    fmax: float = MISSING

    # How many samples to shift between each consecutive frame. This must be
    # smaller than the window size specified by win_length. A reasonable value
    # is 5 - 25 milliseconds, that is, sample_rate * 0.005 or sample_rate * 0.025.
    hop_length: int = MISSING

    # How many samples to include in the window used for STFT extraction. This
    # must be greater than hop_length and a reasonable value is 2x or 4x hop_length.
    win_length: int = MISSING

    # Minimum spectrogram magnitude to enforce to avoid log underflow.
    # Log mel spectrogram is ln(clip(mel_spectrogram, log_epsilon)).
    # If a minimum value isn't enforced, then low-volume regions with a value
    # of zero will become negative infinity.
    log_epsilon: float = MISSING

    # Coefficient for a pre-emphasis filter to apply to the waveform prior to feature extraction.
    # A pre-emphasis filter replaces the signal x[t] with the modified signal x'[t]
    #
    #     x'[t] = x[t] - alpha * x[t - 1]
    #
    # Setting the coefficient value here to zero removes this filter. Setting
    # it to a higher value (commonly 0.97, for instance) effectively boosts
    # high frequencies in the training data. A de-emphasis (inverse of
    # pre-emphasis) filter is applied to the synthesized data, which in turn
    # attenuates high frequencies. Quantization noise is more audible in higher
    # frequencies and thus pre-emphasis can reduce the audible impact of
    # quantization noise.
    pre_emphasis: float = MISSING


class LPC:
    """
    Utility class for linear prediction.

    Algorithms based on LPCNet (https://arxiv.org/pdf/1810.11846.pdf).
    """

    def __init__(self, config: LinearPredictionConfig, mel: MelConfig, pqmf: PQMF):
        """Initilize LPC module.

        Args:
          config: Linear prediction config for this module.
          mel: Mel spectrogram extraction config used throughout.
          pqmf: PQMF module used for subband decomposition.
        """
        super().__init__()

        # Compute pseudo-inverse of mel filter.
        mel_filter = librosa.filters.mel(
            sr=mel.sample_rate,
            n_fft=mel.n_fft,
            n_mels=mel.n_mels,
            fmin=mel.fmin,
            fmax=mel.fmax,
        )
        self.mel_filter_pseudoinverse = np.linalg.pinv(mel_filter)

        self.config = config
        self.mel = mel
        self.pqmf = pqmf

    def mels_to_waveform(self, mels: np.ndarray) -> np.ndarray:
        """
        Convert log mel spectrogram to a waveform.

        This is done by:
          1. Undoing the logarithm and epsilon shift used to compute log mel
             spectrogram from mel spectrogram.
          2. Estimating linear spectrogram from mel spectrogram using a
             pseudo-inverse of the mel filter matrix.
          3. Taking the square root of the power spectrogram to get the
             spectrogram with power=1.
          4. Using the Griffin-Lim algorithm to estimate the audio phase from
             the linear magnitude spectrogram and generate a waveform.

        Args:
          mels: Mel spectrogram of shape [n_mels, n_frames],
            as computed by librosa.feature.melspectrogram().

        Returns:
            A waveform tensor of shape [n_samples].
        """
        # Convert log mel spectrogram (with epsilon shift) to just mel
        # spectrogram (without epsilon shift).
        mel_spec = np.exp(mels) - self.mel.log_epsilon

        # Convert mel spectrogram to linear spectrogram.
        linear_spec = np.maximum(0.0, self.mel_filter_pseudoinverse @ mel_spec)

        # Convert to audio using Griffin-Lim.
        audio = librosa.griffinlim(
            S=linear_spec ** 0.5,  # Undo power=2 in spectrogram
            n_iter=self.config.griffinlim_iterations,
            hop_length=self.mel.hop_length,
            win_length=self.mel.win_length,
            center=False,
        )

        return audio

    @torch.no_grad()
    def estimate(self, mels: np.ndarray, waveform: np.ndarray) -> np.ndarray:
        """
        Estimate the linear prediction based predicted waveform from the mel
        spectrogram and the ground truth waveform.

        To do this estimation:
        1. Convert log mel to linear spectrogram with a pseudo-inverse.
        2. Convert spectrogram to waveform using Griffin-Lim.
        3. If operating on sub-bands, split waveform into subbands with PQMF.
        4. Split waveform(s) into overlapping frames and compute linear
            prediction coefficients (LPCs) for each frame.
        5. Filter frames with LPCs to get output LPC predictions.

        Args:
          mels: A log mel spectrogram, same as the conditioning input to WaveRNN.
            A float32 tensor of shape [n_mels, n_frames].
          waveform: A waveform tensor. If operating on the fullband waveform, this
            is a float32 tensor of shape [n_samples]. If operating on subbands,
            this should instead have shape [n_samples, num_bands]. In all cases,
            the length should be exactly n_samples = n_mels * hop_length.

        Returns:
          A predicted waveform tensor with the same shape as `waveform`.
        """
        gf_waveform = self.mels_to_waveform(mels)

        # Apply subband decomposition to Griffin-Lim estimated waveform, if we
        # are using subband decomposition for the ground truth waveform.
        is_multiband = waveform.ndim == 2
        if is_multiband:
            gf_torch = torch.from_numpy(gf_waveform[None, None, :])
            gf_waveform = self.pqmf.analysis(gf_torch).squeeze(0).numpy().T
        else:
            # To have a uniform representation for LPC estimation.
            gf_waveform = gf_waveform[:, None]
            waveform = waveform[:, None]

        assert (
            gf_waveform.size == waveform.size
        ), "Ground truth and Griffin-Lim waveform shapes do not match"

        # Estimate LPCs.
        order = self.config.order
        hop_length = self.config.hop_length
        win_length = self.config.win_length
        predictions = []
        for band in range(waveform.shape[1]):
            band_predictions = []
            for idx in range(order, gf_waveform.size, hop_length):
                # Extract waveform frame for LPC computation.
                lpc_frame = gf_waveform[idx - order : idx - order + win_length, band]
                if lpc_frame.size < win_length:
                    lpc_frame = np.pad(lpc_frame, (0, win_length - lpc_frame.size))

                try:
                    frame_coeffs = librosa.lpc(lpc_frame, order=order)
                except FloatingPointError:
                    frame_coeffs = np.zeros((order,), dtype="float32")

                # Compute prediction using ground truth waveform and LPCs.
                lfilter_coeffs = np.hstack([[0], -1 * frame_coeffs[1:]])
                frame_pred = scipy.signal.lfilter(
                    lfilter_coeffs, [1], waveform[idx - order : idx + hop_length, band]
                )
                if frame_pred.size < hop_length + order:
                    frame_pred = np.pad(frame_pred, (0, order))
                band_predictions.append(frame_pred[order:])
            predictions.append(np.concatenate(band_predictions))

        if is_multiband:
            np_predictions = np.stack(predictions, axis=1)
        else:
            np_predictions = predictions[0]

        assert (
            np_predictions.size == waveform.size
        ), "Predictions don't match waveform shape"
        return np_predictions.astype(np.float32)
