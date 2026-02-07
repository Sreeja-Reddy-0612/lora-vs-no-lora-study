from datasets import Dataset
import json
from pathlib import Path


def load_instruction_json(path: str) -> Dataset:
    """
    Load instruction-style data from a JSON file.
    Expected format:
    [
      {"instruction": "...", "response": "..."},
      ...
    ]
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if len(raw) == 0:
        raise ValueError("Dataset is empty. Fine-tuning intentionally blocked.")

    formatted = [
        {
            "text": f"Instruction: {ex['instruction']}\nResponse: {ex['response']}"
        }
        for ex in raw
    ]

    return Dataset.from_list(formatted)


def load_inline_examples() -> Dataset:
    """
    Minimal inline dataset (used in your LoRA / QLoRA phases)
    """
    data = [
        {"text": "Instruction: Say hello\nResponse: Hello!"},
        {
            "text": "Instruction: What is AI?\nResponse: AI stands for Artificial Intelligence."
        },
        {
            "text": "Instruction: Capital of France?\nResponse: Paris."
        },
    ]
    return Dataset.from_list(data)
