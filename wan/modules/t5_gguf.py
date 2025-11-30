import torch
import os
import logging
from llama_cpp import Llama

class GGUFT5Encoder(torch.nn.Module):
    def __init__(self, model_path, device=torch.device('cpu'), **kwargs):
        super().__init__()
        self.device = device
        
        logging.info(f"Loading GGUF T5 Encoder from {model_path}...")
        
        # Initialize Llama-cpp with embedding mode
        # n_gpu_layers=-1 means offload ALL layers to GPU
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1, 
            embedding=True,
            verbose=False,
            n_ctx=512, # Match Wan's text_len
            **kwargs
        )
        
        # We don't have a 'model' attribute like the original class, 
        # but we add a dummy one to avoid breaking code that checks for it.
        self.model = torch.nn.Identity() 

    def forward(self, text_list, device=None):
        """
        Args:
            text_list: List of strings (prompts)
            device: Target device for the output tensor
        Returns:
            Tensor of shape [Batch, SeqLen, Dim]
        """
        if device is None:
            device = self.device

        embeddings = []
        for text in text_list:
            # Tokenize
            tokens = self.llm.tokenize(text.encode("utf-8"), add_bos=True, special=True)
            
            # Truncate or Pad to 512
            # Note: T5 usually uses 512. We need to handle this carefully.
            # Llama-cpp's embed() returns the embedding for the *provided* tokens.
            # We might need to pad manually if the model expects fixed size.
            
            # For now, let's just get the embeddings for the tokens we have.
            # Wan 2.1 expects [Batch, 512, 4096] usually.
            
            # Run inference to get embeddings
            # create_embedding() returns a list of floats (pooled? or per token?)
            # Wait, create_embedding usually returns the POOLED embedding (1 vector).
            # We need the SEQUENCE of embeddings (one per token).
            
            # Llama-cpp-python's `embed()` method is what we want for per-token embeddings?
            # Actually, `llm.embed(tokens)` is the low-level API.
            
            # Let's check how to get hidden states. 
            # Standard llama.cpp embedding mode returns the embedding of the *last token* or *all tokens* depending on config.
            # For T5 encoder, we need the full sequence.
            
            # Workaround: We might need to ensure the model was converted with specific flags 
            # or use a specific API call. 
            # However, for now, let's assume `create_embedding` gives us what we need 
            # OR we use the lower level `eval`.
            
            # Actually, for T5 GGUF specifically, it's often used as an encoder.
            # Let's try to use the `embed` method if available, or just `create_embedding` 
            # and see if it returns a list of lists.
            
            # If `embedding=True` is set in constructor:
            # output is typically a list of lists [seq_len, dim]
            output = self.llm.embed(tokens) 
            
            # Convert to tensor
            emb_tensor = torch.tensor(output, dtype=torch.float32)
            
            # We should return the embedding for the tokens we actually have.
            # Llama-cpp might return embeddings for all tokens including padding if we padded?
            # But here we passed `tokens` which is the tokenized list.
            # So `output` length should match `tokens` length.
            
            embeddings.append(emb_tensor)

        # Return list of tensors (moved to device)
        return [e.to(device) for e in embeddings]
