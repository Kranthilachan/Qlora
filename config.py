"""
Central Configuration for Qwen2.5-0.5B LoRA / QLoRA Lab
"""
import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    # Base model identifier (HuggingFace Hub)
    model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    # Fallback to local path if downloaded
    local_model_dir: str = "./models/qwen2.5-0.5b-base"
    # Target device for training ('cpu' for local, 'cuda' if GPU is available)
    device: str = "cpu"
    # Precision
    torch_dtype: str = "float32"  # 'float32' or 'bfloat16' for CPU


@dataclass
class LoraConfigData:
    # LoRA rank (intrinsic rank of adaptation matrices)
    r: int = 8
    # Scaling factor (alpha / r determines weight scaling)
    lora_alpha: int = 16
    # Dropout probability for LoRA layers
    lora_dropout: float = 0.05
    # Modules to apply LoRA to (Attention projection layers in Qwen2.5)
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    # Task type for PEFT
    task_type: str = "CAUSAL_LM"
    bias: str = "none"


@dataclass
class TrainingConfig:
    output_dir: str = "./outputs/qwen_lora_adapter"
    merged_output_dir: str = "./outputs/qwen_merged_model"
    data_path: str = "./data/devops_dataset.jsonl"
    
    # Batch size and gradient accumulation
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4  # Effective batch size = 4
    
    # Optimization
    learning_rate: float = 3e-4
    num_train_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Token length
    max_seq_length: int = 256
    
    # Logging & Saving
    logging_steps: int = 5
    save_strategy: str = "epoch"
    seed: int = 42


@dataclass
class QLoraConfigData:
    """Settings used when running QLoRA on a GPU (e.g. Google Colab / Kaggle)"""
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"  # Normal Float 4
    bnb_4bit_use_double_quant: bool = True  # Nested quantization for extra memory savings
