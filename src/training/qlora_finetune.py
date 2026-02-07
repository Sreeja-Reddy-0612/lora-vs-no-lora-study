import os
import torch
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model


def train_qlora(
    model_name,
    tokenizer,
    tokenized_dataset,
    epochs=1,
):
    print("🔹 Initializing QLoRA (CPU-safe mode)...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": "cpu"},
        llm_int8_enable_fp32_cpu_offload=True,
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["c_attn"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    args = TrainingArguments(
        output_dir="./qlora_tmp",
        per_device_train_batch_size=1,
        num_train_epochs=epochs,
        fp16=False,
        bf16=False,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    # 🔥 THIS IS THE MOST IMPORTANT PART
    save_dir = os.path.abspath("../artifacts/adapters/qlora")
    os.makedirs(save_dir, exist_ok=True)

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    print(f"\n✅ QLoRA adapter saved at: {save_dir}")

    return model
