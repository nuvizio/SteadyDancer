import torch
import torch.nn as nn
import bitsandbytes as bnb
import logging

def replace_linear_with_bnb(module, quantization_config=None):
    """
    Recursively replace nn.Linear with bnb.nn.Linear4bit.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            # Check if we should quantize this layer
            # Usually we don't quantize the output head or embeddings if they are sensitive,
            # but for 14B model, we probably need to quantize most linear layers.
            # WanModel uses Linear for q, k, v, o, ffn.
            
            # Create new bnb layer
            has_bias = child.bias is not None
            bnb_layer = bnb.nn.Linear4bit(
                child.in_features, 
                child.out_features, 
                bias=has_bias,
                compute_dtype=torch.bfloat16,
                compress_statistics=True,
                quant_type='nf4'
            )
            
            # Copy weights
            # We need to create Params4bit from the original weight
            bnb_layer.weight = bnb.nn.Params4bit(
                child.weight.data,
                requires_grad=False,
                compress_statistics=True,
                quant_type='nf4'
            )
            
            if has_bias:
                bnb_layer.bias = nn.Parameter(child.bias.data)
                
            # Replace
            setattr(module, name, bnb_layer)
        else:
            replace_linear_with_bnb(child, quantization_config)

def quantize_model(model):
    logging.info("Quantizing model to 4-bit (NF4) using bitsandbytes...")
    replace_linear_with_bnb(model)
    logging.info("Quantization complete.")
    return model
