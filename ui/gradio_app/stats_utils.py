import torch


def count_total_params(model):
    return sum(p.numel() for p in model.parameters())


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_memory_usage(model):
    if torch.cuda.is_available() and next(model.parameters()).is_cuda:
        return f"{round(torch.cuda.memory_allocated() / 1e6, 2)} MB (GPU)"
    return "CPU only (no GPU memory)"
