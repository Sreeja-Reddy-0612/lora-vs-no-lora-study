import gradio as gr
from model_loader import load_models
from transformers import pipeline


tokenizer, base_model, lora_model, qlora_model = load_models()

pipelines = {
    "Base Model": pipeline("text-generation", model=base_model, tokenizer=tokenizer),
}

if lora_model:
    pipelines["LoRA"] = pipeline("text-generation", model=lora_model, tokenizer=tokenizer)

if qlora_model:
    pipelines["QLoRA"] = pipeline("text-generation", model=qlora_model, tokenizer=tokenizer)


def generate(prompt, model_choice):
    generator = pipelines[model_choice]
    output = generator(prompt, max_new_tokens=50, do_sample=False)
    return output[0]["generated_text"]


with gr.Blocks() as demo:
    gr.Markdown("## 🔬 LoRA vs QLoRA Comparison UI")

    prompt = gr.Textbox(label="Input Prompt", value="Capital of France?")
    model_choice = gr.Radio(
        choices=list(pipelines.keys()),
        value="Base Model",
        label="Model"
    )
    output = gr.Textbox(label="Output")

    run = gr.Button("Run")
    run.click(generate, inputs=[prompt, model_choice], outputs=output)

demo.launch()
