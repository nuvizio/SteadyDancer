import torch
import torch.nn as nn
import bitsandbytes as bnb
import logging

def replace_linear_with_bnb(module, quantization_type='8bit'):
    """
    Recursively replace nn.Linear with bnb.nn.Linear8bitLt or bnb.nn.Linear4bit.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            has_bias = child.bias is not None
            
            if quantization_type == '8bit':
                bnb_layer = bnb.nn.Linear8bitLt(
                    child.in_features,
                    child.out_features,
                    bias=has_bias,
                    has_fp16_weights=False,
                    threshold=6.0,
                )
                bnb_layer.weight = bnb.nn.Int8Params(
                    child.weight.data,
                    requires_grad=False,
                    has_fp16_weights=False
                )
            else:
                # 4-bit
                bnb_layer = bnb.nn.Linear4bit(
                    child.in_features, 
                    child.out_features, 
                    bias=has_bias,
                    compute_dtype=torch.bfloat16,
                    compress_statistics=True,
                    quant_type='nf4'
                )
                bnb_layer.weight = bnb.nn.Params4bit(
                    child.weight.data,
                    requires_grad=False,
                    compress_statistics=True,
                    quant_type='nf4'
                )
            
            if has_bias:
                bnb_layer.bias = nn.Parameter(child.bias.data)
                
            setattr(module, name, bnb_layer)
        else:
            replace_linear_with_bnb(child, quantization_type)

def quantize_model(model, quantization_type='8bit'):
    logging.info(f"Quantizing model to {quantization_type} using bitsandbytes...")
    replace_linear_with_bnb(model, quantization_type=quantization_type)
    logging.info("Quantization complete.")
    return model
