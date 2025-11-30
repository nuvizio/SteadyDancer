import gguf
import sys

def inspect_gguf(path):
    print(f"Inspecting {path}...")
    reader = gguf.GGUFReader(path)
    print(f"Found {len(reader.tensors)} tensors.")
    for i, tensor in enumerate(reader.tensors):
        if i < 20:
            print(f"{tensor.name} | Shape: {tensor.shape} | Type: {tensor.tensor_type}")
    
if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect_gguf(sys.argv[1])
    else:
        print("Usage: python inspect_gguf.py <path_to_gguf>")
