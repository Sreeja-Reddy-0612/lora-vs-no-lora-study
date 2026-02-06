prompt = "Instruction: Capital of France?\nResponse:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=20)

print(tokenizer.decode(output[0], skip_special_tokens=True))
Instruction: Capital of France?
Response: Paris.

---

## Training Observations

- Training completes without OOM
- Loss fluctuates (expected due to tiny dataset)
- Adapter weights update correctly
- Base model remains frozen and quantized

This confirms:
- Correct quantized loading
- Proper LoRA attachment
- Trainer + data collator compatibility

---

## Results

| Metric | Outcome |
|------|--------|
| GPU OOM | ❌ No |
| Training Stability | ✅ Stable |
| Trainable Params | ~0.22% |
| Memory Efficiency | ✅ High |
| Colab Compatibility | ✅ Yes |

---

## Key Takeaways

1. **Full fine-tuning is not scalable** for most practitioners
2. **LoRA improves feasibility**, but still loads full base weights
3. **QLoRA is the most practical fine-tuning strategy** today
4. QLoRA enables:
   - Low-cost experimentation
   - Rapid iteration
   - Production-grade fine-tuning pipelines

---

## Why This Matters (Industry Perspective)

QLoRA is used in:
- Instruction tuning
- Domain adaptation
- Alignment research
- Cost-sensitive production systems

This phase demonstrates **real-world fine-tuning engineering**, not toy examples.

---

## Next Phases (Planned)

- Phase 5: QLoRA vs LoRA comparison
- Phase 6: Adapter merging & inference
- Phase 7: Evaluation metrics
- Phase 8: Paper-style results & ablations

---

## Status
✅ Completed successfully  
✅ Memory-safe  
✅ Reproducible in Colab  

