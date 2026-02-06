## Result

Full fine-tuning fails before training due to an empty instruction dataset.

This demonstrates how brittle full fine-tuning pipelines are: training cannot
even begin without a perfectly prepared dataset.

This failure is intentional and motivates Phase 3 (LoRA fine-tuning).
