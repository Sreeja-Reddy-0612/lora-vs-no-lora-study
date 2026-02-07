import os
import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def adapter_size_mb(adapter_path):
    total = 0
    for root, _, files in os.walk(adapter_path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return round(total / (1024 * 1024), 2)

def memory_stats(model):
    if torch.cuda.is_available() and model.device.type == "cuda":
        return f"{round(torch.cuda.max_memory_allocated() / (1024**2), 2)} MB (GPU)"
    return "CPU only (no GPU memory)"

def model_device(model):
    return model.device.type.upper()
