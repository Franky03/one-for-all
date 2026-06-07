"""
ofa/data/dataloader.py
----------------------
Builds the training data stream. Yields batches as dicts:
    {"input_ids": LongTensor (B, T), "labels": ..., "task_label": str}

Supports weighted mixing of multiple HuggingFace datasets via cfg.dataset_mix.
Currently: OpenHermes-2.5 (70%) + GSM8K (20%) + ARC-Challenge (10%).

The task_label rides along so the trainer can build task-conditional soul
fingerprints (which teacher the student routes to per task type) — this is what
turns the gating telemetry into the paper's Figure 2.

Heavy imports (datasets, torch) are lazy. A `mock_dataloader` provides a
dependency-free stream so the trainer/loss plumbing is testable without GPU
or network.
"""

from __future__ import annotations

from typing import Iterator, Callable

from ..config import OFAConfig


# crude task tagging from instruction text -> category, for fingerprint breakdown
_TASK_KEYWORDS = {
    "math": ["question:", "calculate", "how many", "how much", "solve", "equation",
             "total", "ratio", "percent", "average", "probability"],
    "code": ["def ", "function", "code", "python", "algorithm", "bug"],
    "reasoning": ["why", "explain", "reason", "step by step", "prove", "because",
                  "choices:", "answer:"],
    "factual": ["what is", "who", "when", "where", "fact"],
    "multilingual": ["translate", "portugu", "español", "français", "中文"],
}

# datasets that require a specific config name in load_dataset(id, config, ...)
_DS_CONFIGS: dict[str, str] = {
    "openai/gsm8k":    "main",
    "allenai/ai2_arc": "ARC-Challenge",
}


def _tag_task(text: str) -> str:
    low = text.lower()
    for task, kws in _TASK_KEYWORDS.items():
        if any(k in low for k in kws):
            return task
    return "general"


def make_dataloader(cfg: OFAConfig, tokenizer) -> Callable[[], Iterator[dict]]:
    """Weighted-mix dataloader over cfg.dataset_mix (streamed). Returns a generator fn."""
    import torch
    from datasets import load_dataset, interleave_datasets

    dataset_list, probs = [], []
    for ds_id, weight in cfg.dataset_mix.items():
        extra = (_DS_CONFIGS[ds_id],) if ds_id in _DS_CONFIGS else ()
        ds = load_dataset(ds_id, *extra, split="train", streaming=True)
        dataset_list.append(ds)
        probs.append(weight)

    # normalise probabilities (in case they don't sum to exactly 1.0)
    total = sum(probs)
    probs = [p / total for p in probs]

    combined = (interleave_datasets(dataset_list, probabilities=probs, seed=42)
                if len(dataset_list) > 1 else dataset_list[0])

    def gen() -> Iterator[dict]:
        batch_texts, batch_tasks = [], []
        for ex in combined:
            text = _extract_text(ex)
            if not text:
                continue
            batch_texts.append(text)
            batch_tasks.append(_tag_task(text))
            if len(batch_texts) == cfg.batch_size:
                enc = tokenizer(batch_texts, return_tensors="pt",
                                padding=True, truncation=True,
                                max_length=cfg.max_seq_len)
                input_ids = enc.input_ids.cuda()
                attention_mask = enc.attention_mask.cuda()
                labels = input_ids.clone()
                labels[enc.attention_mask == 0] = -100
                yield {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                    # raw text rides along so each teacher can re-tokenize with
                    # its OWN tokenizer (tokenizer-agnostic geometry distillation)
                    "texts": list(batch_texts),
                    # majority task in the batch (simple, robust)
                    "task_label": max(set(batch_tasks), key=batch_tasks.count),
                }
                batch_texts, batch_tasks = [], []

    return gen


def _extract_text(ex: dict) -> str:
    """Extract plain text from OpenHermes, GSM8K, or ARC examples."""
    # OpenHermes-2.5 — conversations list
    if "conversations" in ex:
        convs = ex["conversations"]
        if not convs:
            return ""
        parts = [c.get("value", "") for c in convs]
        return "\n".join(parts)
    # GSM8K — question + answer (no choices key)
    if "question" in ex and "answer" in ex and "choices" not in ex:
        return f"Question: {ex['question']}\nAnswer: {ex['answer']}"
    # ARC — question + multiple-choice + answer key
    if "question" in ex and "choices" in ex:
        choices = ex["choices"]
        if isinstance(choices, dict):
            texts  = choices.get("text", [])
            labels = choices.get("label", [str(i + 1) for i in range(len(texts))])
            choices_str = "\n".join(f"{l}: {t}" for l, t in zip(labels, texts))
        else:
            choices_str = "\n".join(str(c) for c in choices)
        answer = ex.get("answerKey", "")
        return f"Question: {ex['question']}\nChoices:\n{choices_str}\nAnswer: {answer}"
    return ex.get("text", "")


def mock_dataloader(cfg: OFAConfig, vocab_size: int = 1000,
                    seq_len: int = 16, n_batches: int = 8) -> Callable[[], Iterator[dict]]:
    """
    Dependency-light stream of random token batches for testing the trainer/loss
    plumbing without datasets/network. Requires torch (the trainer needs it
    anyway), but no model downloads.
    """
    import torch
    tasks = ["math", "code", "reasoning", "factual", "general"]

    def gen() -> Iterator[dict]:
        for b in range(n_batches):
            ids = torch.randint(0, vocab_size, (cfg.batch_size, seq_len))
            yield {
                "input_ids": ids,
                "attention_mask": torch.ones_like(ids),
                "labels": ids.clone(),
                "texts": [f"mock example {b}-{j}" for j in range(cfg.batch_size)],
                "task_label": tasks[b % len(tasks)],
            }

    return gen
