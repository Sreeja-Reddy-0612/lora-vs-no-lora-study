import time
import gradio as gr
import torch

def build_comparison_ui(tokenizer, models):

    def run_all(prompt):
        outputs = []

        for name, model in models.items():
            formatted = f"Instruction: {prompt}\nResponse:"
            inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

            start = time.time()
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=60,
                    repetition_penalty=1.2,
                    pad_token_id=tokenizer.eos_token_id
                )

            elapsed = round(time.time() - start, 3)

            text = tokenizer.decode(out[0], skip_special_tokens=True)
            if "Response:" in text:
                text = text.split("Response:", 1)[1].strip()

            outputs.append(
                f"⏱ {elapsed}s | 🧠 Params: {sum(p.numel() for p in model.parameters())} | 💾 CPU\n{text}"
            )

        return outputs

    gr.Markdown("## 🔍 Side-by-Side Comparison")

    prompt = gr.Textbox(
        label="Prompt",
        value="Explain AI in 2 short bullet points."
    )

    btn = gr.Button("Compare")

    outputs = [
        gr.Textbox(label=name, lines=5)
        for name in models.keys()
    ]

    btn.click(run_all, inputs=prompt, outputs=outputs)
