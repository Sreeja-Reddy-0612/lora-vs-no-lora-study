## Phase 2: Full Fine-Tuning Failure

- Attempted full fine-tuning of gpt2-medium
- GPU: Tesla T4 (15GB)
- Training failed with CUDA OOM
- Even batch size = 2 is not viable
- Confirms full fine-tuning is impractical on limited hardware

Conclusion:
Full fine-tuning does not scale. Parameter-efficient methods are required.
