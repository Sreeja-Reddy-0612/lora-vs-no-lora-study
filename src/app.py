"""
Inference-only entry point (NO TRAINING)
"""

from utils.tokenizer import load_tokenizer
from inference.qlora_inference import load_qlora_model
from eval.generation_eval import generate_response


def main():
    model_name = "gpt2-medium"
    adapter_path = "../artifacts/adapters/qlora"

    print("🔹 Loading tokenizer...")
    tokenizer = load_tokenizer(model_name)

    print("🔹 Loading QLoRA adapter...")
    model = load_qlora_model(model_name, adapter_path)

    print("🔹 Running inference...")
    response = generate_response(
        model,
        tokenizer,
        "Capital of France?"
    )

    print("\n✅ Model response:")
    print(response)


if __name__ == "__main__":
    main()
