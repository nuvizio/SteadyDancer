import torch
import gguf
import numpy as np
import os
import logging
from tqdm import tqdm

def load_wan_gguf(model, gguf_path, device='cpu'):
    """
    Loads a GGUF model into a PyTorch model (WanModel).
    Dequantizes Q8_0 weights to Float16/BFloat16 on the fly.
    """
    logging.info(f"Loading GGUF model from {gguf_path}...")
    reader = gguf.GGUFReader(gguf_path)
    
    state_dict = {}
    
    # Map GGUF keys to PyTorch keys
    # GGUF keys usually replace '.' with something else or match exactly?
    # We need to check the keys.
    
    # Iterate over tensors
    for tensor in tqdm(reader.tensors, desc="Loading GGUF tensors"):
        name = tensor.name
        # GGUF naming convention might differ. 
        # e.g. "blk.0.attn_q.weight" vs "blocks.0.self_attn.q.weight"
        # We might need a mapping function.
        # For now, let's assume direct mapping or simple replacement.
        
        # Read data
        data = tensor.data # This is a memmap or byte array
        
        # Dequantize if needed
        if tensor.tensor_type == gguf.GGMLQuantizationType.Q8_0:
            # Q8_0: 32 weights per block.
            # Block structure: d (f16), qs (32 x i8)
            # Total 34 bytes.
            
            # Convert to numpy
            # We need to interpret bytes.
            # This is tricky without a C extension.
            # But we can try numpy structured array.
            
            n_blocks = tensor.n_elements // 32
            
            # Create a view
            # We need to handle the data buffer.
            # tensor.data is a numpy memmap usually?
            # gguf reader returns numpy array for simple types, but for quants?
            # It returns the raw bytes usually.
            
            raw_data = np.frombuffer(data, dtype=np.uint8)
            
            # Reshape to blocks
            # Each block is 34 bytes.
            blocks = raw_data.reshape(n_blocks, 34)
            
            # Extract delta (first 2 bytes) -> float16
            deltas = blocks[:, :2].view(np.float16).flatten()
            
            # Extract qs (next 32 bytes) -> int8
            qs = blocks[:, 2:].view(np.int8)
            
            # Dequantize: w = d * q
            # Broadcast delta: (N, 1) * (N, 32)
            weights = deltas[:, None] * qs
            
            # Flatten
            weights = weights.flatten()
            
            # Convert to torch
            torch_tensor = torch.from_numpy(weights).to(torch.bfloat16)
            
        elif tensor.tensor_type == gguf.GGMLQuantizationType.F16:
            torch_tensor = torch.from_numpy(np.frombuffer(data, dtype=np.float16)).to(torch.bfloat16)
        elif tensor.tensor_type == gguf.GGMLQuantizationType.F32:
            torch_tensor = torch.from_numpy(np.frombuffer(data, dtype=np.float32)).to(torch.bfloat16)
        else:
            logging.warning(f"Unsupported tensor type {tensor.tensor_type} for {name}")
            continue
            
        # Reshape to original shape (GGUF stores in reverse order usually)
        # tensor.shape is what GGUF reports.
        # For Linear weights: GGUF [in, out], PyTorch [out, in].
        # For Conv3d?
        
        # Let's check the shape from GGUF
        gguf_shape = tuple(tensor.shape)
        
        if torch_tensor.numel() != np.prod(gguf_shape):
             torch_tensor = torch_tensor[:np.prod(gguf_shape)]
        
        torch_tensor = torch_tensor.view(*gguf_shape)
        
        # Handle Transposition based on name/shape
        if name.endswith(".weight") and len(gguf_shape) == 2:
            # Linear layer weight: Transpose
            torch_tensor = torch_tensor.t()
        elif "modulation" in name and len(gguf_shape) == 3:
            # Modulation: GGUF [dim, 6, 1] -> PyTorch [1, 6, dim]
            # We assume it's reversed.
            # permute(2, 1, 0)
            torch_tensor = torch_tensor.permute(2, 1, 0)
        elif len(gguf_shape) > 2 and "weight" in name:
             # Conv3d weights?
             # Wan uses Conv3d for patch_embedding.
             # PyTorch Conv3d: [out, in, k, k, k]
             # GGUF might be [k, k, k, in, out] (reversed)
             # We should reverse the dimensions.
             dims = list(range(len(gguf_shape)))
             torch_tensor = torch_tensor.permute(dims[::-1])
             
        state_dict[name] = torch_tensor

    # Load into model
    # We use strict=False because names might not match perfectly
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    if len(missing) > 0:
        logging.warning(f"Missing keys in GGUF load: {missing[:5]} ...")
    if len(unexpected) > 0:
        logging.warning(f"Unexpected keys in GGUF load: {unexpected[:5]} ...")
        
    logging.info("GGUF model loaded.")
    return model
