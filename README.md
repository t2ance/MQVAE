# Meta Quantization

This repository contains the official implementation of Meta Quantization, accompanying paper [Learning to Quantize for
Training Vector-Quantized Networks](https://icml.cc/virtual/2025/poster/43509).

## Steps to use our framework

### Install the pre-requisites

```
pip install -r requirements.txt
```

### Project structure

```
.
├── images # Additional results
├── models
│   ├── __init__.py
│   ├── discriminator.py # discriminator used in VQGAN
│   ├── inception.py # inception model
│   ├── lpips.py # perceptual loss
│   ├── models_vq.py # implementation of VQVAE / VQGAN
│   ├── vqgan_encoder_decoder.py # implementation of encoder of VQGAN
│   └── vqvae_encoder_decoder.py # implementation of encoder of VQVAE
├── arguments.py # commandline arguments
├── dataset.py #  dataset loader
├── ffhq-mqgan.json #  configuration file
├── requirements.txt # dependencies
├── training_vqgan.py # main script, where bilevel-optimization loop is implemented
├── utils.py
└── README.md
```

### Start the experiments

```
python training_vqgan.py --json_file ffhq-mqgan.json
```

You can add customized arguments by additional command line arguments in `ffhq-mqgan.json` file, which defines all hyperparameters.
Please refer to the detailed definition of available arguments in `arguments.py`.

### Multi-GPU training

DDP is supported in our current implementation.

```
torchrun --nproc_per_node=16 --master_port=29500 training_vqgan.py --json_file ffhq-mqgan.json;
```
