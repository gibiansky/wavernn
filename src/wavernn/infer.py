"""
Inference entrypoint for WaveRNN.
"""
import os
import time

import click
import librosa  # type: ignore
import numpy as np
import soundfile  # type: ignore
import torch
from omegaconf import OmegaConf
from tqdm import tqdm  # type: ignore

from wavernn.dataset import AudioDataset
from wavernn.model import Config, Model
from wavernn.prune import PruneConfig, prune
from wavernn.train import CHECKPOINTS_DIR, CONFIG_PATH
from wavernn.util import die_if, load_extension_module

# Format string for tqdm progress bar used during audio synthesis with the
# 'infer' command. Provides slightly more meaningful units than the default.
INFERENCE_BAR_FMT: str = "{l_bar}{bar}| {n:.02f}/{total:.2f} [{rate_noinv_fmt}]"


@click.command("infer")
@click.option(
    "--path", required=True, type=click.Path(file_okay=False), help="Model directory"
)
@click.option(
    "--input",
    "input_file",
    required=True,
    type=click.Path(dir_okay=False, exists=True),
    help="Input WAV file",
)
@click.option(
    "--output",
    "output_file",
    required=True,
    type=click.Path(dir_okay=False),
    help="Output WAV file",
)
def infer(  # pylint: disable=missing-param-doc
    path: str, input_file: str, output_file: str
) -> None:
    """
    Run copy-synthesis inference with a WaveRNN.
    """
    config_path = os.path.join(path, CONFIG_PATH)
    checkpoint_path = os.path.join(path, CHECKPOINTS_DIR, "last.ckpt")

    die_if(not os.path.exists(config_path), f"Missing config path {config_path}")
    die_if(
        not os.path.exists(checkpoint_path),
        f"Missing checkpoint path {checkpoint_path}",
    )
    die_if(not input_file.endswith(".wav"), "--input argument must have .wav extension")
    die_if(
        not output_file.endswith(".wav"), "--output argument must have .wav extension"
    )

    # Create a model with the config.
    model_config: Config = OmegaConf.structured(Config)
    model_config.merge_with(OmegaConf.load(config_path))  # type: ignore
    model = Model.load_from_checkpoint(checkpoint_path, config=model_config)

    # Load the data using a dataset.
    dataset = AudioDataset(
        os.path.dirname(input_file), [os.path.basename(input_file)], model_config.data
    )
    clips = list(dataset.load_samples_from(input_file))

    # Compute expected audio duration.
    mel = model_config.data.mel
    sample_rate = mel.sample_rate
    total_samples: int = sum(int(clip.waveform.numel()) for clip in clips)
    total_secs = total_samples / sample_rate

    # Synthesize with a nice progress bar.
    synthesized_clips = []
    with tqdm(total=total_secs, bar_format=INFERENCE_BAR_FMT, unit="sec") as progress:
        for synthesized in model.infer(clip.spectrogram for clip in clips):
            synthesized_clips.append(synthesized)
            progress.update(synthesized.size / sample_rate)

    # If necessary, apply de-emphasis (inverse of pre-emphasis) to the signal.
    waveform = np.concatenate(synthesized_clips)
    if mel.pre_emphasis > 0:
        waveform = librosa.effects.deemphasis(waveform, coef=mel.pre_emphasis)
        waveform = np.clip(waveform, -0.9999, 0.9999)

    soundfile.write(output_file, waveform, sample_rate)


@click.command("benchmark")
@click.option(
    "--config", required=True, type=click.Path(dir_okay=False), help="Model config"
)
@click.option("--warmup-frames", default=100, help="How many frames to run for warmup")
@click.option(
    "--bench-frames", default=500, help="How many frames to run for benchmark"
)
def benchmark(  # pylint: disable=missing-param-doc
    config: str, warmup_frames: int, bench_frames: int
) -> None:
    """
    Benchmark inference of an untrained WaveRNN.
    """
    # Create a model with the config.
    model_config: Config = OmegaConf.structured(Config)
    model_config.merge_with(OmegaConf.load(config))  # type: ignore
    model = Model(config=model_config)

    # Set the model input range.
    model.conditioner.set_input_range(-2, 2)

    # Apply pruning (as if we had trained to completion).
    #
    # Note: Since the matrices are different sizes, the initial weights will
    # have different magnitudes. Since pruning is magnitude-based, this will
    # significantly affect which weights are pruned and skew benchmark results.
    # For example, the output matrix is usually smallest and thus has the
    # largest weight magnitudes (due to Xavier initialization), and thus will
    # have near zero sparsity.
    weights = model.weights()
    sparse_matrices = [
        weights.output_weight,
        weights.gru_weight_hh,
        weights.hidden_weight,
    ]
    prune(
        config=model_config.prune,
        parameters=sparse_matrices,
        step=model_config.prune.stop_iteration,
    )

    # Generate test input values.
    max_frames = max(warmup_frames, bench_frames)
    spectrogram = torch.randn(model_config.data.mel.n_mels, max_frames)

    # Run inference at least once before benchmarking to reduce startup noise.
    print("Warming up...")
    model.infer([spectrogram[:, :warmup_frames]])

    print("Benchmarking...")
    start_time = time.time()
    samples = next(model.infer([spectrogram[:, :bench_frames]], timing=True))
    elapsed = time.time() - start_time

    synthesized_seconds = samples.size / model_config.data.mel.sample_rate
    synthesis_rate = synthesized_seconds / elapsed
    realtime_factor = elapsed / synthesized_seconds
    print(
        f"Synthesized:       {bench_frames} frames = {synthesized_seconds:.2f} seconds"
    )
    print(f"Synthesis Rate:    {synthesis_rate:.2f} synthesized seconds per second")
    print(f"Real-Time Factor:  {realtime_factor:.2f}x realtime")


@click.command("gemv")
@click.option("--input-size", required=True, type=int, help="GEMV input size")
@click.option("--output-size", required=True, type=int, help="GEMV output size")
@click.option("--sparsity", required=True, type=float, help="Percent to prune")
@click.option("--warmup", default=10000, help="Multiplies to run for warmup")
@click.option("--bench", default=1000000, help="Multiplies to run for benchmark")
@click.option("--block-rows", default=8, help="Row block size")
@click.option("--block-cols", default=8, help="Column block size")
@torch.no_grad()
def gemv(  # pylint: disable=missing-param-doc
    input_size: int,
    output_size: int,
    warmup: int,
    bench: int,
    sparsity: float,
    block_rows: int,
    block_cols: int,
) -> None:
    """
    Benchmark and test sparse GEMV kernels.
    """
    # Create sparse matrix.
    matrix = torch.randn(output_size, input_size)
    vector = torch.randn(input_size)
    bias = torch.randn(output_size)
    prune(
        config=PruneConfig(
            final_sparsity=sparsity,
            start_iteration=0,
            stop_iteration=1,
            block=(block_rows, block_cols),
        ),
        parameters=[matrix],
        step=10,
    )

    # Compute with a dense gemv.
    expected = torch.addmv(bias, matrix, vector)

    # Run the optimized kernel op.
    load_extension_module()
    output = torch.ops.wavernn.sparse_gemv(bias, matrix, vector)

    if not torch.allclose(expected, output, atol=1e-5):
        print("Failed comparison test; op output differs from expected.")
        print("Max difference:", torch.abs(expected - output).max())

    elapsed_dense, elapsed_sparse = torch.ops.wavernn.sparse_gemv_benchmark(
        bias, matrix, vector, warmup, bench
    )

    dense_mean_ms = elapsed_dense.mean()
    dense_std_ms = elapsed_dense.std()
    print(f"Dense GEMV: {dense_mean_ms:.2f} ± {dense_std_ms:.1f} µs")

    sparse_mean_ms = elapsed_sparse.mean()
    sparse_std_ms = elapsed_sparse.std()
    print(f"Sparse GEMV: {sparse_mean_ms:.2f} ± {sparse_std_ms:.1f} µs")
    print(f"Sparsity Speedup: {dense_mean_ms / sparse_mean_ms:.2f}x")
