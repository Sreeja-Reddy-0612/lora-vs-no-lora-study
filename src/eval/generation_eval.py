import torch


def generate_response(model, tokenizer, instruction, max_new_tokens=30):
    prompt = f"Instruction: {instruction}\nResponse:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)
