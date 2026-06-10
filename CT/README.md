# 3D CT Reconstruction

## Installation Guide

Please install the dependencies listed below using conda or pip.

### Dependency

This code has been tested with the following dependency packages:
```
python: 3.9.13
torch: 1.12.1
numpy: 1.24.3
skimage: 0.19.3
yaml: 6.0
opencv (cv2): 4.6.0
odl: 1.0.0.dev0
astra-toolbox
nni: 3.0
```

## Data

The pancreas 4D-CT dataset (`pancs_4dct_phase1.npz`, `pancs_4dct_phase6.npz`) can be obtained from the [NeRP](https://github.com/liyues/NeRP.git) repository. Place the `.npz` files under `data/ct_data/` before running experiments. Refer to the configs for data path settings.

## 3D CT Reconstruction Experiment

The 3D CT reconstruction experiments use a 3D cone-beam geometry and are designed to reconstruct volumetric images from sparsely sampled projections, leveraging prior information and hardware-aware quantization.

### Step 1: Prior Embedding

Represent the 3D prior image by an implicit neural network. The prior image (e.g., `pancs_4dct_phase1.npz`: phase-1 image of a 10-phase 4D pancreas CT dataset) is provided under the `data/ct_data` folder.

```
bash scripts/train.sh
```

### Step 2: Network Training

Reconstruct the 3D CT image from sparsely sampled projections using the prior embedding. The reconstruction target image (e.g., `pancs_4dct_phase6.npz`: phase-6 image of a 10-phase 4D pancreas CT dataset) is also provided under `data/ct_data`.

```
bash scripts/tune.sh
```

### Step 3: Inference

After training, the network weights are quantized using Hardware-aware Quantization **(HAQ)**, and the reconstructed 3D image is inferred and saved. This step simulates the deployment of the model on resistive memory hardware and evaluates the quantized model's performance.

```
bash scripts/test.sh
```

## Hardware-aware Hyperparameter Optimization (HAPO)

Hardware-aware Hyperparameter optimization is implemented in the Jupyter notebook `HPO_search.ipynb`. 

## Acknowledgements

This codebase is improved based on [NeRP](https://github.com/liyues/NeRP.git). We thank the original authors for their open-source contribution.
