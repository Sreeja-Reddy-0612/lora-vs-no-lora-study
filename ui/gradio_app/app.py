import time
import gradio as gr
import torch

from model_loader import load_models
from comparison_page import build_comparison_ui
from stats_ui import build_stats_page

# ---------------- LOAD MODELS ----------------
tokenizer, base_model, lora_model, qlora_model = load_models()

LORA_PATH = "artifacts/adapters/lora"
QLORA_PATH = "artifacts/adapters/qlora"

MODELS = {"Base Model": base_model}
if lora_model:
    MODELS["LoRA"] = lora_model
if qlora_model:
    MODELS["QLoRA"] = qlora_model


# ---------------- INFERENCE (UNCHANGED CORE) ----------------
def generate(prompt, model_choice):
    model = MODELS[model_choice]

    formatted = f"Instruction: {prompt}\nResponse:"
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    start = time.time()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.7,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    elapsed = round(time.time() - start, 3)

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    if "Response:" in decoded:
        decoded = decoded.split("Response:", 1)[1].strip()

    return f"⏱ {elapsed}s\n{decoded}"


# ---------------- DARK ORANGE THEME (FIXED) ----------------
custom_css = """
/* Global */
body {
  background:#0b0b0b;
  color:white;
}

/* Headings */
h1, h2, h3 {
  color:#ff8c42 !important;
}

/* Tabs */
.gr-tab-item {
  color:white !important;
}
.gr-tab-item.selected {
  color:#ff8c42 !important;
  border-bottom:2px solid #ff8c42 !important;
}

/* Inputs */
textarea, input {
  background:#151515 !important;
  color:white !important;
  border:1px solid #2a2a2a !important;
}

/* Buttons (neutral) */
button {
  background:#1e1e1e !important;
  color:white !important;
  border:1px solid #2a2a2a !important;
}
button:hover {
  background:#262626 !important;
}

/* ✅ RADIO BUTTON FIX (THIS IS THE KEY) */
.gr-radio input[type="radio"] {
  accent-color:#666666;   /* unselected */
}

.gr-radio input[type="radio"]:checked {
  accent-color:#ff8c42;   /* selected ORANGE */
}

/* Optional: label highlight when selected */
.gr-radio label:has(input:checked) {
  color:#ff8c42 !important;
  font-weight:600;
}
"""


# ---------------- APP ----------------
with gr.Blocks() as app:
    gr.Markdown("# 🧪 LoRA vs QLoRA Engineering Study")

    with gr.Tabs():
        # -------- Inference --------
        with gr.Tab("Inference"):
            prompt = gr.Textbox(
                label="Prompt",
                value=(
                    "Answer in exactly 2 bullet points, each under 6 words.\n"
                    "Question: What is AI?"
                )
            )

            model_choice = gr.Radio(
                choices=list(MODELS.keys()),
                value="Base Model",
                label="Model"
            )

            output = gr.Textbox(label="Output", lines=6)
            run = gr.Button("Run")
            run.click(generate, [prompt, model_choice], output)

        # -------- Comparison --------
        with gr.Tab("Comparison"):
            build_comparison_ui(tokenizer, MODELS)

        # -------- Stats --------
        with gr.Tab("Stats"):
            build_stats_page(
                base_model,
                lora_model,
                qlora_model,
                LORA_PATH,
                QLORA_PATH
            )

app.launch(css=custom_css)
