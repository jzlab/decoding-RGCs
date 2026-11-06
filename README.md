# LNP_LN_LN_Models

**Forked from:** [https://github.com/NilouGhazavi/LNP_LN_LN_Models](https://github.com/NilouGhazavi/LNP_LN_LN_Models)

PyTorch implementation of Linear-Nonlinear-Poisson (LNP) models for Retinal Ganglion Cells (RGCs), combined with a neural decoder to reconstruct video frames from simulated RGC firing patterns.

## Installation

Create and activate the required Conda environment:

```bash
conda env create -f environment.yml
conda activate lnp_torch
```

## Pipeline Setup and Execution

The pipeline consists of three stages: data generation, neural decoder training, and video reconstruction.

### 1. Generate RGC Activations
Generates simulated RGC activations from input videos and saves them as `.h5` files. Supports processing single videos or batch processing entire directories.

```bash
python autoencoder/sk_generate_activations.py \
    --video lnp_naturalmovies/ \
    --params shared/params.yaml \
    --output lnp_activations_bank \
    --random-crops 25
```

### 2. Train the Decoder
Trains a PyTorch Decoder mapping RGC firing rates back to original video crops. The dataset loader memory-maps the serialized `.h5` files to handle large multi-video structures efficiently.

```bash
python autoencoder/sk_train_decoder.py \
    --activations lnp_activations_bank \
    --params shared/params.yaml \
    --training-params autoencoder/params_training.yaml \
    --output best_decoder.pt \
    --epochs 50 \
    --device cuda
```

### 3. Reconstruct Target Videos
Chops a target video into spatial patches, computes responses, decodes them using the trained model weights, and writes the stitched patches into an output `.mp4`.

```bash
python autoencoder/sk_reconstruct_video.py \
    --video lnp_naturalmovies/Video_2.mp4 \
    --weights best_decoder.pt \
    --output-video Video_2_reconstructed.mp4 \
    --device cuda
```

## Configuration Files

- **`shared/params.yaml`**: Configures the spatial/temporal constraints and convolution parameters for the forward Retina LNP simulation.
- **`autoencoder/params_training.yaml`**: Defines decoder architecture properties and basic training hyperparameters.
## Repository layout

Each reconstruction approach is self-contained, with its own configs, notebooks
and model weights:

| Path | Contents |
|---|---|
| `shared/` | Code used by every pipeline: the RGC mosaic encoders (`sk_retina_encoders.py`), IO helpers (`sk_utils.py`) and the retina config `params.yaml`. |
| `autoencoder/` | LNP encoder to neural decoder reconstruction. |
| `diffusion/` | Diffusion-guided reconstruction, including the DPS backbone (`dps_backbone/`) and the spectral analysis package (`analysis/`). |
| `meanflow/` | Mean-flow reconstruction (in progress). |
| `from_fork/` | Upstream code inherited from the fork, unmodified. |
| `tests/` | pytest suite. |

Weights and cached activations live in `<package>/models/` and are gitignored;
only a `.gitkeep` is tracked so the directories survive a clone.

Entrypoints are plain scripts and can be run from any working directory:

```bash
python diffusion/diffusion_hyperparameter_search.py --help
python autoencoder/sk_train_decoder.py --help
```

Each one prepends the repository root to `sys.path`, so `shared/` resolves, and
its config defaults are anchored to the repository root rather than the CWD.

Cell-type combination sweeps are driven by `run_sweeps.sh`:

```bash
./run_sweeps.sh --stage base --backbone dps    # four major types
./run_sweeps.sh --stage sm   --backbone dps    # with ON_Smooth_Monostratified
```
