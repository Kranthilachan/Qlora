"""
Dataset loader and formatter for Qwen2.5 ChatML format.
"""
import json
from typing import Dict, List, Tuple
from datasets import Dataset
from transformers import PreTrainedTokenizer

SYSTEM_PROMPT = (
    "You are OpsPilot, an expert DevOps and Linux Systems Engineer. "
    "When asked for assistance, you always respond strictly in valid JSON format with the following keys:\n"
    "- 'command': The exact, optimal terminal/bash command to run.\n"
    "- 'explanation': A concise technical explanation of what each flag and command does.\n"
    "- 'safety_level': One of 'SAFE', 'MEDIUM', or 'HIGH'.\n"
    "- 'caution': Important risk factor or warning before running the command."
)


def format_chatml_prompt(instruction: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    """Formats an instruction into Qwen ChatML prompt format up to assistant start."""
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def format_full_example(instruction: str, response: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    """Formats full training conversation including assistant output."""
    prompt = format_chatml_prompt(instruction, system_prompt)
    return f"{prompt}{response}<|im_end|>"


def load_and_prepare_dataset(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 256,
    train_split_ratio: float = 0.9,
    seed: int = 42
) -> Tuple[Dataset, Dataset]:
    """Loads .jsonl dataset, applies ChatML formatting, and tokenizes for causal LM training."""
    raw_samples: List[Dict[str, str]] = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_samples.append(json.loads(line))

    # Tokenize full texts
    def tokenize_fn(batch):
        full_texts = [
            format_full_example(inst, resp)
            for inst, resp in zip(batch["instruction"], batch["response"])
        ]
        tokenized = tokenizer(
            full_texts,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )
        # For causal LM, labels are identical to input_ids
        tokenized["labels"] = [ids[:] for ids in tokenized["input_ids"]]
        return tokenized

    hf_dataset = Dataset.from_list(raw_samples)
    tokenized_dataset = hf_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=hf_dataset.column_names,
        desc="Tokenizing dataset",
    )

    # Train / Val split
    split_dataset = tokenized_dataset.train_test_split(test_size=1.0 - train_split_ratio, seed=seed)
    return split_dataset["train"], split_dataset["test"]
