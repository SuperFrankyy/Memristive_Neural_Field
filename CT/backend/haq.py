import torch
import numpy as np

def haq_simu(weight, n=16, ratio=1.5, std=0.15, coarse=True):
    # 记住输入weight的设备
    device = weight.device
    
    # 将weight归一化，保持在原设备上
    target_weight = weight / torch.max(torch.abs(weight))
    
    # ratio默认为n长度，直接创建在正确的设备上
    if coarse:
        ratio = torch.full((n,), ratio, device=device)

    print(ratio)
    
    # 在同一设备上创建resistance tensor
    resistance = torch.zeros((target_weight.shape[0], target_weight.shape[1], n), device=device)

    # 预计算ratio的累积乘积，在GPU上
    ratio_cumprods = torch.ones(n, device=device)
    for i in range(1, n):
        ratio_cumprods[i] = ratio_cumprods[i-1] * ratio[i-1]

    for x in range(n):
        print('第', x+1, '比特')
        # 把resistance_1中所有值减去0.5
        new_resistance = resistance - 0.5

        # 把第x比特后面的bit都置为0
        new_resistance[:, :, x:] = 0

        # 生成一个weight_1形状的张量，每个元素的值为0，在同一设备上
        decimal_weight = torch.zeros(target_weight.shape, device=device)

        # 对resistance_1中的每个元素进行处理，首先转化为十进制数
        for i in range(new_resistance.shape[0]):
            for j in range(new_resistance.shape[1]):
                if target_weight[i, j] == 0: # 如果权重为0，则不需要进行处理
                    decimal_weight[i][j] = 0
                else:
                    for k in range(new_resistance.shape[2]):
                        # 使用预计算的ratio_cumprods，全部在GPU上进行
                        decimal_weight[i][j] += new_resistance[i][j][k] / ratio_cumprods[k]

        weight_diff = decimal_weight - target_weight
        # 计算relative error
        relative_error = weight_diff / target_weight
        # 展示relative大于0.1的比例
        print('relative error > 0.1:', torch.sum(relative_error > 0.1) / (relative_error.shape[0] * relative_error.shape[1]))
        
        for i in range(resistance.shape[0]):
            for j in range(resistance.shape[1]):
                if weight_diff[i, j] > 0:
                    resistance[i, j, x] = 0
                if weight_diff[i, j] < 0: 
                    resistance[i, j, x] = 1

        # 对resistance[:, :, x]中非0的值加上标准差为std的高斯噪声
        mask = resistance[:, :, x] != 0
        noise = torch.randn(resistance[:, :, x][mask].shape[0], device=device) * std
        resistance[:, :, x][mask] += noise
    
    # 恢复原始范围
    decimal_weight = decimal_weight * torch.max(torch.abs(weight))

    return decimal_weight, weight_diff, relative_error, resistance

def ptq(weight, n=16):
    """
    对权重进行二进制量化
    Args:
        weight: 输入权重 (torch.Tensor)
        n: 量化位数 (int)，最小为1
    Returns:
        decimal_weight: 量化后的十进制权重
        binary: 二进制表示
    """
    device = weight.device
    
    # 保持在原始设备上进行所有计算
    max_val = torch.max(torch.abs(weight))
    normalized_weight = weight / max_val
    
    if n == 1:
        # 1位量化特殊处理：直接二值化为-1和1
        binary = torch.zeros((*weight.shape, 1), device=device, dtype=torch.float32)
        binary[..., 0] = (normalized_weight > 0).float()
        decimal_weight = (2 * binary[..., 0] - 1) * max_val
    else:
        # 将[-1,1]映射到[0,1]
        scaled_weight = (normalized_weight + 1) / 2
        
        # 量化到2^n级别
        quantized = torch.round(scaled_weight * (2**n - 1))
        
        # 获取二进制表示，确保使用float类型
        binary = torch.zeros((*weight.shape, n), device=device, dtype=torch.float32)
        for i in range(n):
            binary[..., i] = (quantized % 2).float()  # 确保转换为float
            quantized = torch.floor(quantized / 2)
        
        # 反量化为十进制
        decimal = torch.zeros_like(weight)
        powers = torch.tensor([2**i for i in range(n)], device=device, dtype=torch.float32)  # 确保使用float类型
        
        # 使用einsum进行高效计算
        decimal = torch.einsum('...i,i->...', binary, powers)
        
        # 映射回原始范围
        decimal = decimal / (2**n - 1)  # 映射到[0,1]
        decimal = decimal * 2 - 1       # 映射到[-1,1]
        decimal_weight = decimal * max_val  # 映射回原始范围
    
    return decimal_weight, binary
