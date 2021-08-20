# WaveRNN

This repository contains a production-ready implementation of WaveRNN-based
neural vocoder for autoregressive waveform synthesis. This implementation,
unlike others, focuses on production applications, supporting custom datasets
and efficient inference on a variety of platforms.

## Background

To learn more about why we created this WaveRNN implementation, check out the announcement blog post:

* Announcement Post: A Production-Ready Open Source WaveRNN

To learn about what WaveRNN is or understand the nitty-gritty details of implementing it, check out our blog post series, WaveRNN Demystified:

* Part 1: What is WaveRNN?
* Part 2: Inference
* Part 3: Sparsity
* Part 4: Quantization Noise and Pre-Emphasis

## Installation

Prior to installation, we recommend setting up your environment with
[pyenv](https://github.com/pyenv/pyenv) with Python 3.9.6 and a [virtualenv](https://docs.python.org/3/library/venv.html). The
remainder of the installation instructions assume that you are using an
appropriate environment or virtualenv.

First, install MKL, which is needed to build the extension modules. You can
follow the installation instructions
[here](https://software.intel.com/content/www/us/en/develop/articles/installation-guide-for-intel-oneapi-toolkits.html).
On Ubuntu 20.04 or later, you can easily install from a package manager:
```
$ sudo apt install libmkl-dev
```

You can install this package from PyPI:

```
$ pip install wavernn
```

If you plan on developing in it or want the latest code, you can clone and install from Github:
```
$ git clone git@github.com:gibiansky/wavernn.git
$ cd wavernn && pip install --editable .
```

## Command-Line Usage

This package is used through the `wavernn` command-line interface. You can start by downloading a dataset for training. You can view the datasets supported out-of-the-box with `wavernn dataset list`:

```
$ wavernn dataset list
ljspeech
vctk
libritts
```

To download one of the datasets, use `wavernn dataset download` and provide a path to which to download to:

```
$ wavernn dataset download ljspeech --path ./ljspeech
```

Once a dataset is downloaded, you can start training with `wavernn train`, specifying a config file and where to save the model:

```
$ wavernn train --config config/wavernn.yaml --path runs/my-model --data ./ljspeech
```

This will start training. You will see a progress bar and information about the model. Tensorboard logs will be stored in the model directory and you can view them by starting Tensorboard and navigating to it in your browser:

```
$ tensorboard --logdir runs/my-model
```

Once you have a trained model (or at least one checkpoint has been saved), you can run copy-synthesis inference with `wavernn infer`:

```
$ wavernn infer \
    --path runs/my-model \
    --input ./ljspeech/LJSpeech-1.1/wavs/LJ001-0001.wav \
    --output resynthesized-LJ001-0001.wav
```

In addition to using the models for inference using the `infer` command, you
can export the models to be used with your own code using the `export` command:

```
$ wavernn export --path runs/my-model --output my-exported-model.jit
```

You can now use `my-exported-model.jit` along with the library API described
below to build custom applications (including full TTS engines) using your
exported WaveRNN.

## Library Usage

In addition to creating and using models using the `wavernn` command-line tool,
you can embed trained models for inference in your own Python or C++ applications.
To do so, export your model with the `export` command:

```
$ wavernn export --path /path/to/my/model --output /path/to/exported/model.jit
```

From Python, you can load and use this model as follows:

```python
# Import the package.
import numpy
import wavernn

# Load the exported WaveRNN.
model = wavernn.load("/path/to/exported/model.jit")

# Create a test spectrogram with 50 frames.
# (In practice, you need a real spectrogram, not random noise.)
n_mels = 80
n_timesteps = 50
spectrogram = numpy.random.random((n_mels, n_timesteps))

# Synthesize an audio clip.
waveform, state = model.synthesize(spectrogram, state=None)
```

## Citing

If you use this work in your research, please cite:

```
@misc{GibianskyWaveRNN,
  author = {Andrew Gibiansky},
  title = {Andrew's WaveRNN},
  year = {2021},
  howpublished = {\url{https://github.com/gibiansky/wavernn}}
}
```
