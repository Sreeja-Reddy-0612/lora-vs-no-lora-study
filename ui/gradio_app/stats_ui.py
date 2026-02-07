import os
import gradio as gr
import torch

def get_adapter_size(path):
    if not path or not os.path.exists(path):
        return "N/A"
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return f"{round(total / (1024*1024), 2)} MB"


def build_stats_page(base_model, lora_model, qlora_model, lora_path, qlora_path):

    def collect():
        rows = []

        def add(name, model, adapter_path=None):
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            rows.append([
                name,
                f"{total:,}",
                f"{trainable:,}",
                get_adapter_size(adapter_path),
                model.device.type.upper(),
                "CPU only (no GPU memory)"
            ])

        add("Base Model", base_model)
        if lora_model:
            add("LoRA", lora_model, lora_path)
        if qlora_model:
            add("QLoRA", qlora_model, qlora_path)

        return rows

    gr.Markdown("## 📊 Model Statistics")

    table = gr.Dataframe(
        headers=[
            "Model",
            "Total Params",
            "Trainable Params",
            "Adapter Size",
            "Device",
            "Memory"
        ],
        value=collect(),   # ✅ SET INITIAL VALUE
        interactive=False
    )

    refresh = gr.Button("Refresh Stats")
    refresh.click(fn=collect, outputs=table)
