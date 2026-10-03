"""
Train Qwen2.5-0.5B using LoRA on CPU.
This script demonstrates parameter-efficient fine-tuning (PEFT) on standard CPU hardware.
"""
import os
import sys
import time
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from config import ModelConfig, LoraConfigData, TrainingConfig
from data.dataset_loader import load_and_prepare_dataset

console = Console()


def print_trainable_parameters(model):
    """Prints the number of trainable parameters versus frozen parameters."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    percentage = 100 * trainable_params / all_param
    
    table = Table(title="🔬 Parameter Efficiency (LoRA)", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Count / Value", style="green")
    
    table.add_row("Total Parameters", f"{all_param:,}")
    table.add_row("Trainable Parameters (LoRA)", f"{trainable_params:,}")
    table.add_row("Frozen Parameters", f"{all_param - trainable_params:,}")
    table.add_row("Trainable %", f"{percentage:.4f}%")
    table.add_row("Memory Savings", f"~{100 - percentage:.2f}% weights frozen")
    
    console.print(table)
    return trainable_params, all_param, percentage


def main():
    model_cfg = ModelConfig()
    lora_cfg_data = LoraConfigData()
    train_cfg = TrainingConfig()

    console.print(Panel.fit(
        "[bold cyan]Qwen2.5-0.5B LoRA Fine-Tuning Lab (CPU)[/bold cyan]\n"
        f"Base Model: [yellow]{model_cfg.model_name_or_path}[/yellow]\n"
        f"Target Device: [green]{model_cfg.device.upper()}[/green]\n"
        f"LoRA Rank (r): [green]{lora_cfg_data.r}[/green] | Alpha: [green]{lora_cfg_data.lora_alpha}[/green]",
        border_style="cyan"
    ))

    # 1. Load Tokenizer
    console.print("[dim]Loading tokenizer...[/dim]")
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.model_name_or_path,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Model in Float32 for CPU stability
    console.print("[dim]Loading base model in Float32 for CPU execution...[/dim]")
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.model_name_or_path,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # 3. Configure LoRA (PEFT)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg_data.r,
        lora_alpha=lora_cfg_data.lora_alpha,
        lora_dropout=lora_cfg_data.lora_dropout,
        target_modules=lora_cfg_data.target_modules,
        bias=lora_cfg_data.bias,
    )

    # 4. Wrap with PEFT
    model = get_peft_model(model, peft_config)
    print_trainable_parameters(model)

    # 5. Load and Tokenize Dataset
    console.print(f"[dim]Loading dataset from {train_cfg.data_path}...[/dim]")
    train_dataset, eval_dataset = load_and_prepare_dataset(
        data_path=train_cfg.data_path,
        tokenizer=tokenizer,
        max_length=train_cfg.max_seq_length,
        train_split_ratio=0.9,
    )
    console.print(f"[green]Dataset split: {len(train_dataset)} train samples, {len(eval_dataset)} eval samples.[/green]")

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt"
    )

    # 6. Training Arguments
    os.makedirs(train_cfg.output_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=train_cfg.output_dir,
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        learning_rate=train_cfg.learning_rate,
        num_train_epochs=train_cfg.num_train_epochs,
        warmup_steps=2,
        weight_decay=train_cfg.weight_decay,
        max_grad_norm=train_cfg.max_grad_norm,
        logging_steps=train_cfg.logging_steps,
        save_strategy=train_cfg.save_strategy,
        eval_strategy="no",
        save_total_limit=1,
        use_cpu=True,
        report_to="none",
        logging_first_step=True,
        seed=train_cfg.seed,
    )

    # 7. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # 8. Start Training
    console.print("\n[bold green]🚀 Commencing LoRA training on CPU...[/bold green]")
    start_time = time.time()
    train_result = trainer.train()
    elapsed = time.time() - start_time

    console.print(f"\n[bold green]✅ Training completed in {elapsed:.1f} seconds ({elapsed/60:.2f} mins)![/bold green]")
    console.print(f"Final Training Loss: [yellow]{train_result.training_loss:.4f}[/yellow]")

    # 9. Save LoRA Adapter & Tokenizer
    console.print(f"[dim]Saving LoRA adapter to {train_cfg.output_dir}...[/dim]")
    trainer.model.save_pretrained(train_cfg.output_dir)
    tokenizer.save_pretrained(train_cfg.output_dir)
    
    adapter_size_mb = sum(
        os.path.getsize(os.path.join(train_cfg.output_dir, f))
        for f in os.listdir(train_cfg.output_dir)
        if os.path.isfile(os.path.join(train_cfg.output_dir, f))
    ) / (1024 * 1024)

    console.print(Panel(
        f"[bold green]LoRA Adapter Successfully Saved![/bold green]\n"
        f"Location: [yellow]{train_cfg.output_dir}[/yellow]\n"
        f"Adapter On-Disk Size: [bold cyan]{adapter_size_mb:.2f} MB[/bold cyan]\n"
        f"(Compare this to ~1000 MB base model!)",
        title="Training Complete",
        border_style="green"
    ))


if __name__ == "__main__":
    main()
