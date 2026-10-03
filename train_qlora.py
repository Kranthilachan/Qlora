"""
Train Qwen2.5 using QLoRA (4-bit NormalFloat Quantization + LoRA).
Designed for GPU environments (Google Colab / Kaggle / NVIDIA GPU).
Includes device check and fallback explanation if run on CPU.
"""
import os
import sys
import time
import torch
from rich.console import Console
from rich.panel import Panel

from config import ModelConfig, LoraConfigData, TrainingConfig, QLoraConfigData
from data.dataset_loader import load_and_prepare_dataset

console = Console()


def train_qlora():
    model_cfg = ModelConfig()
    lora_cfg = LoraConfigData()
    train_cfg = TrainingConfig()
    qlora_cfg = QLoraConfigData()

    console.print(Panel.fit(
        "[bold cyan]Qwen2.5 QLoRA (4-Bit NF4) Training Pipeline[/bold cyan]\n"
        f"Quantization: [green]4-bit NormalFloat (NF4) with Double Quantization[/green]\n"
        f"Compute Dtype: [green]{qlora_cfg.bnb_4bit_compute_dtype}[/green]\n"
        f"Base Model: [yellow]{model_cfg.model_name_or_path}[/yellow]",
        border_style="cyan"
    ))

    # Check CUDA availability
    if not torch.cuda.is_available():
        console.print(Panel(
            "[bold yellow]⚠️ No CUDA GPU detected on current machine![/bold yellow]\n\n"
            "QLoRA requires CUDA tensor kernels (`bitsandbytes` NF4 quantization) for backpropagation.\n"
            "For training on your local CPU:\n"
            "   👉 Run: [bold green]python train_lora.py[/bold green] (Fully optimized for CPU!)\n\n"
            "To run this exact 4-bit QLoRA script on a free GPU:\n"
            "   👉 Open the included [bold cyan]notebooks/qlora_colab.ipynb[/bold cyan] in Google Colab (free T4 GPU).",
            title="GPU Required for QLoRA",
            border_style="yellow"
        ))
        return

    # If CUDA is available, proceed with full QLoRA pipeline:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
        Trainer,
        DataCollatorForSeq2Seq,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

    compute_dtype = getattr(torch, qlora_cfg.bnb_4bit_compute_dtype)

    # 1. 4-bit BitsAndBytes Configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=qlora_cfg.load_in_4bit,
        bnb_4bit_quant_type=qlora_cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=qlora_cfg.bnb_4bit_use_double_quant,
    )

    console.print("[dim]Loading tokenizer...[/dim]")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load 4-bit quantized base model
    console.print("[dim]Loading model in 4-bit NF4 precision...[/dim]")
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # 3. Prepare for k-bit training (freezes quantized weights & casts layer norm to FP32)
    model = prepare_model_for_kbit_training(model)

    # 4. LoRA Adapter configuration
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        target_modules=lora_cfg.target_modules,
        bias=lora_cfg.bias,
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 5. Dataset & Collator
    train_dataset, eval_dataset = load_and_prepare_dataset(
        data_path=train_cfg.data_path,
        tokenizer=tokenizer,
        max_length=train_cfg.max_seq_length,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)

    # 6. Training Arguments
    qlora_output_dir = "./outputs/qwen_qlora_adapter"
    training_args = TrainingArguments(
        output_dir=qlora_output_dir,
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        learning_rate=train_cfg.learning_rate,
        num_train_epochs=train_cfg.num_train_epochs,
        warmup_ratio=train_cfg.warmup_ratio,
        logging_steps=train_cfg.logging_steps,
        save_strategy=train_cfg.save_strategy,
        fp16=False,
        bf16=True,
        report_to="none",
        seed=train_cfg.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    console.print("\n[bold green]🚀 Commencing QLoRA training on GPU...[/bold green]")
    trainer.train()

    model.save_pretrained(qlora_output_dir)
    tokenizer.save_pretrained(qlora_output_dir)
    console.print(f"[bold green]✅ QLoRA adapter saved to {qlora_output_dir}![/bold green]")


if __name__ == "__main__":
    train_qlora()
