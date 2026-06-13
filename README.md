# MTSC Sulci Unsupervised Training

This public cleanup keeps the package focused on:

- unsupervised/pre-training workflows in `pre_training/pretraining.py`
- fine-tuning workflows in `fine_tuning/fine_tuning.py`
- classification workflows in `scripts/classification.py`

## Layout

```text
MTSC_sulci/
├── pre_training/      # unsupervised pre-training entry points
├── fine_tuning/       # fine-tuning entry points
├── scripts/           # runnable workflow scripts
└── utilities/         # data loading, transforms, models, trainers
```

The package root keeps compatibility aliases for the old flat imports. Prefer
the subpackage paths for new code, for example:

```python
from MTSC_sulci.pre_training.pretraining import pre
from MTSC_sulci.scripts.classification import classif
from MTSC_sulci.utilities.load_data import data_build
```

## Installation

Create and activate a Python environment, then install the package from this
repository:

```bash
python -m pip install .
```

For development, use editable mode:

```bash
python -m pip install -e .
```

The installer reads [setup.py](setup.py), installs the Python dependencies, and
registers two command-line tools:

```bash
mtsc-pre-training --help
mtsc-classification --help
```

GPU-enabled PyTorch installations can be platform-specific. If you need a
specific CUDA build, install `torch` and `torchvision` first using the command
recommended by the PyTorch website, then run:

```bash
python -m pip install -e .
```

The optional `xformers` acceleration dependency is not installed by default. To
request it:

```bash
python -m pip install -e ".[xformers]"
```

## Tests

The test suite includes lightweight synthetic 3D self-training smoke tests. It
creates small toy 3D sphere volumes and checks four objective families:
reconstruction, contrastive learning, diffusion-style denoising, and GAN-style
adversarial reconstruction.

Run it with the standard library test runner:

```bash
python -m unittest discover tests
```

For a quicker visual walkthrough, open:

```text
tests/synthetic_pretraining_demo.ipynb
```

Current smoke-test results on the toy dataset:

```text
reconstruction: initial_loss=0.043076 final_loss=0.020939 improvement=51.39%
contrastive: initial_loss=0.005044 final_loss=0.000495 improvement=90.18%
diffusion: initial_loss=0.043263 final_loss=0.022034 improvement=49.07%
gan: initial_reconstruction=0.042939 final_reconstruction=0.027283 reconstruction_improvement=36.46% discriminator_accuracy=100.00%
```

## Running Analyses

You can run the public workflows either with command-line parameters or with a
YAML/JSON config file. The example YAML lives at:

```text
configs/default.yaml
```

Pre-training with the config file:

```bash
python scripts/pre_training.py --config configs/default.yaml
```

or, after installation:

```bash
mtsc-pre-training --config configs/default.yaml
```

Pre-training with command-line overrides:

```bash
mtsc-pre-training \
  --method pre \
  --data-root1 /path/to/subjects/ \
  --data-root2 /path/to/labels/ \
  --path /path/to/output/ \
  --model-name pre_training_model_swift.pt \
  --batch-n 2 \
  --num-epochs 5
```

Available pre-training methods are `pre`, `gan`, `contrastive`, and
`diffusion`.

Classification with the config file:

```bash
python scripts/classification.py --config configs/default.yaml
```

or, after installation:

```bash
mtsc-classification --config configs/default.yaml
```

Classification with command-line overrides:

```bash
mtsc-classification \
  --data-root1 /path/to/subjects/ \
  --data-root2 /path/to/labels/ \
  --excel /path/to/labels.csv \
  --path /path/to/output/ \
  --model-name class_model.pt \
  --batch-n 2 \
  --num-epochs 5
```

Command-line arguments override values from the config file.

## Weights & Biases

Do not commit a personal W&B API key to this repository. To enable online W&B
logging, set the key outside the code:

```bash
export WANDB_API_KEY="your-key"
```

For local or public runs without W&B authentication, use:

```bash
export WANDB_MODE=offline
```

or:

```bash
export WANDB_MODE=disabled
```
