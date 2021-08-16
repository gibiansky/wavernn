# WaveRNN

This repository contains a production-ready implementation of WaveRNN-based
neural vocoder for autoregressive waveform synthesis. This implementation,
unlike others, focuses on production applications, supporting custom datasets
and efficient inference on a variety of platforms.

## Background

To learn more about why we created this WaveRNN implementation, check out the announcement blog post:

* Announcement Post: A Production-Ready Open Source WaveRNN

To learn about what WaveRNN is or understand the nitty-gritty details of implementing it, check out our blog post series, WaveRNN Demystified:

* Part 1: Intro
* Part 2: Inference
* Part 3: Sparsity

## Installation

You can install this package from PyPI:

```
pip install wavernn
```

If you plan on developing in it or want the latest, you can clone and install from Github:
```
git clone git@github.com:gibiansky/wavernn.git
cd wavernn && pip install --editable .
```

## Usage

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
