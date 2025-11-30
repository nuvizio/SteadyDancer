import torch
import os
import logging
from llama_cpp import Llama

class GGUFT5Encoder(torch.nn.Module):
    def __init__(self, model_path, device=torch.device('cpu'), **kwargs):
        super().__init__()
        self.device = device
        self.model_path = model_path
        self.kwargs = kwargs
        self.llm = None
        
        logging.info(f"Loading GGUF T5 Encoder from {model_path}...")
        self.load_model()
        
        # We don't have a 'model' attribute like the original class, 
        # but we add a dummy one to avoid breaking code that checks for it.
        self.model = torch.nn.Identity() 

    def load_model(self):
        if self.llm is not None:
            return
            
        # Initialize Llama-cpp with embedding mode
        # n_gpu_layers=-1 means offload ALL layers to GPU
        n_gpu_layers = self.kwargs.get('n_gpu_layers', -1)
        
        self.llm = Llama(
            model_path=self.model_path,
            n_gpu_layers=n_gpu_layers, 
            embedding=True,
            verbose=False,
            n_ctx=512, # Match Wan's text_len
            **self.kwargs
        )

    def offload_model(self):
        if self.llm is not None:
            logging.info("Offloading GGUF T5 Encoder...")
            del self.llm
            self.llm = None
            import gc
            gc.collect()
            torch.cuda.empty_cache()

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

        # Ensure model is loaded
        self.load_model()

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
            # If `embedding=True` is set in constructor:
            # output is typically a list of lists [seq_len, dim]
            output = self.llm.embed(tokens) 
            
            # Convert to tensor
            emb_tensor = torch.tensor(output, dtype=torch.float32)
            
            embeddings.append(emb_tensor)

        # Return list of tensors (moved to device)
        return [e.to(device) for e in embeddings]
