import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


BASE_MODEL = "gpt2-medium"
LORA_PATH = "C:\\Users\\APPLE\\Desktop\\lora-vs-no-lora-study\\artifacts\\adapters\\lora"
QLORA_PATH = "C:\\Users\\APPLE\\Desktop\\lora-vs-no-lora-study\\artifacts\\adapters\\qlora"


def load_base_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    model.eval()
    return tokenizer, model


def try_load_adapter(base_model, adapter_path):
    if not os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        print(f"⚠️ Adapter not found at {adapter_path}, skipping.")
        return None

    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        is_trainable=False,
    )
    model.eval()
    return model


def load_models():
    tokenizer, base_model = load_base_model()

    lora_model = try_load_adapter(base_model, LORA_PATH)
    qlora_model = try_load_adapter(base_model, QLORA_PATH)

    return tokenizer, base_model, lora_model, qlora_model
