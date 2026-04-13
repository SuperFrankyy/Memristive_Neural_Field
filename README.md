# Efficient and Accurate Neural Field Reconstruction Using Resistive Memory

This repository contains the minimal demo code for the paper **"Efficient and Accurate Neural Field Reconstruction Using Resistive Memory"** for software simulation.

**The simulation code includes:**
- Neural network training
- Hardware-aware network compression
- Hardware-aware hyperparameter optimization (HAPO)
- Hardware-aware quantization (HAQ) simulation

**When using real resistive memory hardware for inference, matrix multiplication is performed by hardware calls to a Xilinx FPGA in the customized system via `pynq.dma`.**

The codebase integrates three sub-projects, each corresponding to a specific task:

- **CT Reconstruction** (`CT/`)

---

## Project Structure

```
memristive_neural_field/
  ├── CT/         # CT reconstruction
  └── README.md   # (this file)
```

---

## Dependencies

- Python 3.9+ (Anaconda recommended)
- Each sub-project has its own `requirements.txt` with detailed dependencies.   
- Main dependencies include:
  - torch
  - numpy
  - matplotlib
  - imageio
  - imageio-ffmpeg
  - configargparse
  - tensorboard
  - tqdm
  - opencv-python
  - lpips
  - jupyter

**Please refer to each sub-project's README and requirements.txt for precise environment setup.**

Example installation for a sub-project (e.g., NVS):

```bash
cd NVS
pip install -r requirements.txt
```

---

## Quick Start

### 1. CT Reconstruction (`CT/`)
- See `CT/README.md` for details on data preparation, training, and evaluation.



<!-- ## Acknowledgement

This codebase is inspired by and builds upon the following open-source projects:

- [yenchenlin/nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch)
- [liyues/NeRP](https://github.com/liyues/NeRP)
- [albertpumarola/D-NeRF](https://github.com/albertpumarola/D-NeRF)

If you use this code for research, please cite these projects and our paper.  -->