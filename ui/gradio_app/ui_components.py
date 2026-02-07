import gradio as gr


def build_ui(run_fn):
    with gr.Blocks() as demo:
        gr.Markdown("# 🔬 LoRA vs QLoRA Comparison")

        prompt = gr.Textbox(label="Prompt", value="Capital of France?")

        btn = gr.Button("Run Comparison")

        base_out = gr.Textbox(label="Base Model")
        lora_out = gr.Textbox(label="LoRA")
        qlora_out = gr.Textbox(label="QLoRA")

        btn.click(
            run_fn,
            inputs=prompt,
            outputs=[base_out, lora_out, qlora_out]
        )

    return demo
