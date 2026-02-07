import torch




def get_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_memory_usage(model):
    device = next(model.parameters()).device
    return "CPU (no GPU memory)" if device.type == "cpu" else "GPU"
