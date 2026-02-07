from data.instruction_loader import load_inline_examples
from utils.tokenizer import load_tokenizer, tokenize_dataset
from training.qlora_finetune import train_qlora


def main():
    model_name = "gpt2-medium"

    print("🔹 Loading dataset...")
    dataset = load_inline_examples()

    print("🔹 Loading tokenizer...")
    tokenizer = load_tokenizer(model_name)

    print("🔹 Tokenizing dataset...")
    tokenized_ds = tokenize_dataset(dataset, tokenizer)

    print("🔹 Training QLoRA adapter...")
    train_qlora(
        model_name=model_name,
        tokenizer=tokenizer,
        tokenized_dataset=tokenized_ds,
        epochs=1,
    )


if __name__ == "__main__":
    main()
