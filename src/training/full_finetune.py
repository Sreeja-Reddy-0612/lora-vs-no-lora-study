import torch
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)


def train_full_finetune(
    model_name,
    tokenized_dataset,
    output_dir="./full_ft_out",
    epochs=3,
):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=None, mlm=False
    )

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        num_train_epochs=epochs,
        fp16=True,
        logging_steps=1,
        report_to="none",
        save_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    trainer.train()
