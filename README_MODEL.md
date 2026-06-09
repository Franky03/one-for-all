---
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:Qwen/Qwen2.5-0.5B-Instruct
- lora
- transformers
- knowledge-distillation
- cka
license: mit
---

# Deku — One for All Student

Qwen2.5-0.5B-Instruct fine-tuned via **gated CKA geometry distillation** from 5 heterogeneous teacher LLMs. The student learns to absorb the representation geometry of multiple teachers simultaneously through a learned routing gate.

## Teachers

| Model | Strength |
|---|---|
| Qwen2.5-1.5B-Instruct | code, structured reasoning |
| SmolLM2-1.7B-Instruct | curated quality |
| Phi-3.5-mini-instruct | instruction following, CoT |
| gemma-2-2b-it | long context |
| MiniCPM-2B-sft-bf16 | multilingual, efficiency |

## Method

**Path B — geometry-only, tokenizer-agnostic distillation.**

Each teacher has a different tokenizer and hidden dimension, making token-level KL divergence ill-defined across the ensemble. Instead, the student learns to align its hidden-state geometry with each teacher via **CKA (Centered Kernel Alignment)**, weighted by a learned gating network that routes each input to the most relevant teacher.

The objective is:

```
L = λ1·L_task + λ2·L_KL(Qwen1.5B) + λ3·L_geo(gate)
```

- `L_task` — next-token cross-entropy on the training mix
- `L_KL` — KL divergence from Qwen2.5-1.5B (same tokenizer, zero friction)
- `L_geo` — gated CKA loss: `1 - mean_i gate_i · CKA(H_student, Pi_i · H_teacher_i)`

Lambdas follow a three-phase curriculum: task-only warmup → KL ramp-in → geometry ramp-in.

## Training

- **Base:** Qwen/Qwen2.5-0.5B-Instruct
- **Adapter:** LoRA r=64, α=128 on all attention + MLP projections
- **Data:** OpenHermes-2.5 (70%) + GSM8K (20%) + ARC-Challenge (10%)
- **Steps:** 5 000 · batch 8 · seq 512
- **Hardware:** A100-80GB via Modal
- **Precision:** bfloat16

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = PeftModel.from_pretrained(base, "build-small-hackathon/deku")
tok = AutoTokenizer.from_pretrained("build-small-hackathon/deku")

inputs = tok("Explain what a hash map is.", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=200)
print(tok.decode(out[0], skip_special_tokens=True))
```

## Demo

Live soul space + probe interface: [build-small-hackathon/one-for-all](https://huggingface.co/spaces/build-small-hackathon/one-for-all)

---
PEFT 0.19.1
