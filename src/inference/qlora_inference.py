import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel


def load_qlora_model(base_model_name, adapter_path):
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="cpu",
        torch_dtype=torch.float32,
    )

    model = PeftModel.from_pretrained(
        model,
        adapter_path,
        is_trainable=False,
    )

    model.eval()
    return model
