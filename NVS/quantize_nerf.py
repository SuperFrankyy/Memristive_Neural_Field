import torch
import numpy as np
from backend.haq import haq_simu, ptq  # Import two quantization methods
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Quantization for NeRF model')
    parser.add_argument('--bits', type=int, default=8, help='Number of quantization bits')
    parser.add_argument('--use_ptq', action='store_true', help='Whether to use PTQ quantization method')
    parser.add_argument('--ratio', type=float, default=1.5, help='Ratio for HAQ quantization')
    parser.add_argument('--dataset', type=str, default='mic', help='Dataset name')
    return parser.parse_args()

def quantize_model(model, bits=8, ratio=1.5, use_ptq=False):
    """
    Quantize the model using haq_simu or ptq method
    
    Args:
        model: The model to be quantized (OrderedDict)
        bits: Number of quantization bits
        ratio: Quantization ratio (used only in haq_simu)
        use_ptq: Whether to use PTQ quantization method
    
    Returns:
        quantized_weights: Dictionary of quantized weights
    """
    quantized_weights = {}
    
    # Directly iterate over OrderedDict
    for key, param in model.items():
        if 'weight' in key and isinstance(param, torch.Tensor):
            # Clone weights to avoid modifying the original weights directly
            weight = param.clone()
            
            try:
                if use_ptq:
                    # Quantize using PTQ
                    decimal_weight, binary = ptq(weight, n=bits)
                    quantized_weights[key] = decimal_weight
                    print(f'{key} PTQ quantization done')
                    # Release memory
                    del binary
                else:
                    # Quantize using haq_simu
                    decimal_weight, weight_diff, relative_error, resistance = haq_simu(
                        weight, n=bits, ratio=ratio, std=0.15, coarse=True
                    )
                    quantized_weights[key] = decimal_weight
                    
                    # Calculate quantization error
                    error_ratio = torch.sum(torch.abs(relative_error) > 0.1).item() / relative_error.numel()
                    print(f'{key} HAQ quantization done, ratio of relative error > 0.1: {error_ratio:.4f}')
                    
                    # Release memory
                    del weight_diff, relative_error, resistance
                
                # Clean GPU cache after processing each weight
                if key != list(model.keys())[-1]:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Error occurred during quantization of {key}: {str(e)}")
                continue
    
    return quantized_weights

def apply_quantized_weights(model, quantized_weights):
    """
    Apply quantized weights to the model
    
    Args:
        model: Original model (OrderedDict)
        quantized_weights: Dictionary of quantized weights
    """
    for key in quantized_weights.keys():
        if key in model:
            model[key] = quantized_weights[key]
    
    return model

def main():
    args = parse_args()
    print(f"Using parameters: bits={args.bits}, use_ptq={args.use_ptq}, ratio={args.ratio}, dataset={args.dataset}")
    
    # Set path according to dataset name
    base_path = f"logs/prune_all_hardwaredata_quantbittest/{args.dataset}_w128*8_r32_pruning0.9_train/"
    
    # Load model
    checkpoint = torch.load(base_path + '200000.tar')
    
    # Get model state dict
    model_fn = checkpoint['network_fn_state_dict']
    model_fine = checkpoint['network_fine_state_dict']
    
    try:
        # Quantize
        print("\nStart quantizing coarse network...")
        quantized_weights_fn = quantize_model(
            model_fn,
            bits=args.bits,
            ratio=args.ratio,
            use_ptq=args.use_ptq
        )
        
        print("\nStart quantizing fine network...")
        quantized_weights_fine = quantize_model(
            model_fine,
            bits=args.bits,
            ratio=args.ratio,
            use_ptq=args.use_ptq
        )
        
        # Apply quantized weights
        model_fn = apply_quantized_weights(model_fn, quantized_weights_fn)
        model_fine = apply_quantized_weights(model_fine, quantized_weights_fine)
        
        print("\nModel quantization completed")
        
        # Save quantized model
        new_checkpoint = {
            'network_fn_state_dict': model_fn,
            'network_fine_state_dict': model_fine,
            'global_step': checkpoint.get('global_step', 0)
        }
        torch.save(new_checkpoint, base_path + '400000.tar')
        
    except Exception as e:
        print(f"\nError occurred during quantization: {str(e)}")

if __name__ == "__main__":
    main() 