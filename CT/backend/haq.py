import torch
import numpy as np

def haq_simu(weight, n=16, ratio=1.5, std=0.15, coarse=True):
    device = weight.device

    # normalize weight, keep on original device
    target_weight = weight / torch.max(torch.abs(weight))

    if coarse:
        ratio = torch.full((n,), ratio, device=device)

    print(ratio)

    # create resistance tensor on same device
    resistance = torch.zeros((target_weight.shape[0], target_weight.shape[1], n), device=device)

    # precompute cumulative product of ratios on device
    ratio_cumprods = torch.ones(n, device=device)
    for i in range(1, n):
        ratio_cumprods[i] = ratio_cumprods[i-1] * ratio[i-1]

    for x in range(n):
        print('bit', x+1)
        # subtract 0.5 from all resistance values
        new_resistance = resistance - 0.5

        # zero out bits after position x
        new_resistance[:, :, x:] = 0

        # create zero tensor with weight shape on same device
        decimal_weight = torch.zeros(target_weight.shape, device=device)

        # convert each resistance element to decimal
        for i in range(new_resistance.shape[0]):
            for j in range(new_resistance.shape[1]):
                if target_weight[i, j] == 0:  # zero weights need no processing
                    decimal_weight[i][j] = 0
                else:
                    for k in range(new_resistance.shape[2]):
                        decimal_weight[i][j] += new_resistance[i][j][k] / ratio_cumprods[k]

        weight_diff = decimal_weight - target_weight
        # compute relative error
        relative_error = weight_diff / target_weight
        # show proportion of relative error > 0.1
        print('relative error > 0.1:', torch.sum(relative_error > 0.1) / (relative_error.shape[0] * relative_error.shape[1]))

        for i in range(resistance.shape[0]):
            for j in range(resistance.shape[1]):
                if weight_diff[i, j] > 0:
                    resistance[i, j, x] = 0
                if weight_diff[i, j] < 0:
                    resistance[i, j, x] = 1

        # add Gaussian noise with std to non-zero resistance values
        mask = resistance[:, :, x] != 0
        noise = torch.randn(resistance[:, :, x][mask].shape[0], device=device) * std
        resistance[:, :, x][mask] += noise

    # restore original range
    decimal_weight = decimal_weight * torch.max(torch.abs(weight))

    return decimal_weight, weight_diff, relative_error, resistance

def ptq(weight, n=16):
    """
    Binary quantization of weights.
    Args:
        weight: input weight tensor (torch.Tensor)
        n: number of quantization bits (int), minimum 1
    Returns:
        decimal_weight: quantized decimal weights
        binary: binary representation
    """
    device = weight.device

    # perform all computations on original device
    max_val = torch.max(torch.abs(weight))
    normalized_weight = weight / max_val

    if n == 1:
        # 1-bit quantization: binarize to -1 and 1
        binary = torch.zeros((*weight.shape, 1), device=device, dtype=torch.float32)
        binary[..., 0] = (normalized_weight > 0).float()
        decimal_weight = (2 * binary[..., 0] - 1) * max_val
    else:
        # map [-1,1] to [0,1]
        scaled_weight = (normalized_weight + 1) / 2

        # quantize to 2^n levels
        quantized = torch.round(scaled_weight * (2**n - 1))

        # get binary representation, ensure float type
        binary = torch.zeros((*weight.shape, n), device=device, dtype=torch.float32)
        for i in range(n):
            binary[..., i] = (quantized % 2).float()
            quantized = torch.floor(quantized / 2)

        # dequantize to decimal
        decimal = torch.zeros_like(weight)
        powers = torch.tensor([2**i for i in range(n)], device=device, dtype=torch.float32)

        decimal = torch.einsum('...i,i->...', binary, powers)

        # map back to original range
        decimal = decimal / (2**n - 1)  # map to [0,1]
        decimal = decimal * 2 - 1       # map to [-1,1]
        decimal_weight = decimal * max_val  # map back to original range

    return decimal_weight, binary
