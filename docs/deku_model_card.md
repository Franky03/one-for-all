---
language:
- en
- pt
- zh
license: apache-2.0
base_model: Qwen/Qwen2.5-0.5B-Instruct
tags:
- knowledge-distillation
- lora
- peft
- qwen2
- geometry-distillation
- cka
pipeline_tag: text-generation
---

# Deku — One for All Student

Deku is a **Qwen2.5-0.5B-Instruct** fine-tuned with LoRA via gated CKA geometry distillation (Path B). It absorbs representation structure from 6 heterogeneous teacher LLMs simultaneously, without access to teacher logits or shared tokenizers.

## Model Details

- **Base model:** [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- **Fine-tuning method:** LoRA (PEFT) + GatingNetwork (linear head over student hidden states)
- **Distillation strategy:** Geometry-only (CKA) — Path B, tokenizer-agnostic
- **Language(s):** English, Portuguese, Chinese (inherited from base)
- **License:** Apache 2.0
- **Developed by:** build-small-hackathon team

## Teachers

| Teacher | Parameters | Hidden dim |
|---------|-----------|------------|
| Qwen2.5-1.5B-Instruct | 1.5B | 1536 |
| SmolLM2-1.7B-Instruct | 1.7B | 2048 |
| Phi-3.5-mini-instruct | 3.8B | 3072 |
| gemma-2-2b-it | 2.7B | 2304 |
| MiniCPM-2B-sft-bf16 | 2.7B | 2304 |
| Nemotron-Mini-4B-Instruct | 4B | 3072 |

## Distillation Approach

**Path B — geometry-only, tokenizer-agnostic:**

1. Each teacher processes its own tokenization of the same text. No shared vocabulary required.
2. Sequence representations are pooled via masked mean (attention mask weighted) to a single vector per model.
3. Linear projection heads map each teacher's hidden space into the student's hidden space (d=896).
4. A **GatingNetwork** (linear layer over student pooled state → softmax over 6 teachers) learns which teacher's geometry to prioritize per input.
5. Loss = task cross-entropy + λ·CKA geometry loss (student vs. gated teacher mixture).

The CKA geometry loss aligns the *relational structure* of representations (which samples are similar to which) rather than raw activation values, making it robust to dimension mismatch and tokenizer differences.

## Usage

```python
from transformers import AutoTokenizer
from peft import PeftModel, AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained(
    "build-small-hackathon/deku",
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("build-small-hackathon/deku")

inputs = tokenizer("Explain gradient descent in one sentence.", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Additional Artifacts

- **GatingNetwork weights:** `gating.pt` — `torch.load("gating.pt")`, `state_dict` for a `nn.Linear(896, 6)`
- **Projection weights:** `projections.pt` — list of 6 `nn.Linear` state dicts (teacher_i → student space)
- **GGUF (llama.cpp):** [build-small-hackathon/deku-gguf](https://huggingface.co/build-small-hackathon/deku-gguf) — merged f16 + q8_0 builds, plus `gating.npz` for a torch-free gate
- **Visualization data:** [build-small-hackathon/ofa-viz-data](https://huggingface.co/datasets/build-small-hackathon/ofa-viz-data) — raw projected embeddings for 3D UMAP soul space
- **Interactive Space:** [build-small-hackathon/one-for-all](https://huggingface.co/spaces/build-small-hackathon/one-for-all)

## Training

- **Steps:** 5000
- **Batch size:** 8
- **Optimizer:** AdamW, lr=2e-4
- **Hardware:** Modal A100-80GB
- **Data:** weighted mix of [teknium/OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) (0.70), [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) (0.20), [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) (0.10)
