# NeRF Training & Inference (NVS)

## Dependencies

Please use the following dependencies (install via conda or pip):

```
python                3.9.13
torch                 1.12.1
torchvision           0.13.1
numpy                 1.24.3
imageio               2.22.1
imageio-ffmpeg        0.4.7
matplotlib            3.5.1
configargparse        1.5.3
tensorboard           2.10.1
tqdm                  4.64.1
opencv-python         4.6.0
pytorch-msssim        0.2.1
lpips                 0.1.4
nni                   3.0
torch-pruning         1.3.2
```
## Data Preparation

Place the NeRF synthetic datasets under the `data/nerf_synthetic/` directory, for example:
```
data/nerf_synthetic/lego
```

You can download the datasets from Google Drive:
[link](https://drive.google.com/drive/folders/1cK3UDIJqKAAm7zyrxRYVFJ0BRMgrwhh4)

## Config Files

All training config files are in the `NVS/configs/` directory. Configs for 8 main datasets (chair, drums, ficus, hotdog, materials, mic, ship, lego) are provided.

## Training

To train all scenes in batch:
```
bash scripts/train.sh
```
Or train a single scene:
```
python run_nerf.py --config configs/lego.txt
```

## Hardware-aware Quantization

After training, you can simulate the hardware-aware quantization process of the model using `quantize_nerf.py`. The main hyperparameters are:

- `--bits`: Quantization bitwidth (default: 8)
- `--use_ptq`: Use PTQ quantization method (default: False, use HAQ by default)
- `--ratio`: Significance ratio for HAQ quantization (default: 1.5)
- `--dataset`: Dataset name (default: mic, should match your experiment)

Example usage:
```
python quantize_nerf.py --bits 8 --ratio 1.5 --dataset lego
```

## Hardware-aware Hyperparameter Optimization (HAPO)

Hardware-aware hyperparameter optimization is implemented in the Jupyter notebook `HPO_nerf.ipynb`.

You can use this notebook to search for optimal software and hardware configurations (e.g., netwidth, rank, bits, ratio, etc.) in a hardware-aware manner.

This will launch a series of experiments using `run_nerf_hpo.py` as the trial script.

## Inference

After quantization, you can perform inference or evaluation using the quantized model checkpoint (e.g., `400000.tar`).

You may need to modify your evaluation or rendering script to load the quantized checkpoint, for example:
```
python run_nerf.py --config configs/lego.txt --render_only --ckpt logs/prune_all_hardwaredata_quantbittest/lego_w128*8_r32_pruning0.9_train/400000.tar
```
Replace the `--ckpt` path with your quantized model path as needed.

## Results & Logs

Training logs and results are saved in the `logs/train/` directory by default.

## Acknowledgement

This project is built upon [NeRF](http://www.matthewtancik.com/nerf) and the [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch) implementation.
