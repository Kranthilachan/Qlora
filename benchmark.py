"""
Benchmark Script: LoRA vs QLoRA Architecture & Resource Analysis.
Displays parameter counts, memory footprints, and architectural trade-offs.
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from config import ModelConfig, LoraConfigData

console = Console()


def analyze_model():
    model_cfg = ModelConfig()
    lora_cfg = LoraConfigData()

    console.print(Panel.fit(
        "[bold cyan]🔍 Architecture & Memory Benchmark: LoRA vs. QLoRA[/bold cyan]\n"
        f"Model: [yellow]{model_cfg.model_name_or_path}[/yellow]",
        border_style="cyan"
    ))

    # Load Base Model
    console.print("[dim]Analyzing model layers and tensor parameters...[/dim]")
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.model_name_or_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    
    total_params = sum(p.numel() for p in model.parameters())

    # Configure PEFT LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        target_modules=lora_cfg.target_modules,
        bias=lora_cfg.bias,
    )
    lora_model = get_peft_model(model, peft_config)

    trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    trainable_pct = (trainable_params / total_params) * 100

    # Memory calculations
    # 1 param in FP32 = 4 bytes
    # 1 param in FP16/BF16 = 2 bytes
    # 1 param in INT4 / NF4 = 0.5 bytes (4 bits)
    mem_fp32_mb = (total_params * 4) / (1024 * 1024)
    mem_fp16_mb = (total_params * 2) / (1024 * 1024)
    mem_int4_mb = (total_params * 0.5) / (1024 * 1024)
    adapter_mem_mb = (trainable_params * 4) / (1024 * 1024)

    # 1. Parameter Breakdown Table
    param_table = Table(title="📊 Parameter Distribution", show_header=True, header_style="bold magenta")
    param_table.add_column("Component", style="cyan")
    param_table.add_column("Parameters", justify="right", style="green")
    param_table.add_column("% of Total", justify="right", style="yellow")

    param_table.add_row("Base Model (Frozen)", f"{frozen_params:,}", f"{100 - trainable_pct:.3f}%")
    param_table.add_row("LoRA Adapters (Trainable)", f"{trainable_params:,}", f"{trainable_pct:.3f}%")
    param_table.add_row("Total Model", f"{total_params:,}", "100.000%")
    console.print(param_table)

    # 2. Precision & Memory Footprint Comparison Table
    mem_table = Table(title="💾 Precision & Memory Comparison", show_header=True, header_style="bold blue")
    mem_table.add_column("Technique", style="cyan")
    mem_table.add_column("Weight Precision", style="magenta")
    mem_table.add_column("Base Weights (RAM/VRAM)", justify="right", style="green")
    mem_table.add_column("Trainable Gradient State", justify="right", style="yellow")
    mem_table.add_column("Hardware Suitability", style="white")

    mem_table.add_row(
        "Full Fine-Tuning",
        "FP32 / FP16",
        f"~{mem_fp16_mb:.1f} MB",
        f"~{mem_fp16_mb * 4:.1f} MB (All params)",
        "High-end GPUs only"
    )
    mem_table.add_row(
        "Standard LoRA",
        "FP32 (CPU) / FP16",
        f"~{mem_fp32_mb:.1f} MB (FP32)",
        f"~{adapter_mem_mb * 3:.2f} MB (Only adapters)",
        "CPU & Standard GPUs (Fast)"
    )
    mem_table.add_row(
        "QLoRA (4-bit NF4)",
        "4-bit NF4 + LoRA",
        f"~{mem_int4_mb:.1f} MB (4-bit)",
        f"~{adapter_mem_mb * 3:.2f} MB (Only adapters)",
        "GPUs (VRAM Saver / Free Colab T4)"
    )
    console.print(mem_table)

    console.print(Panel(
        f"[bold green]Key Takeaway:[/bold green]\n"
        f"By using LoRA, you only train [bold cyan]{trainable_params:,} parameters[/bold cyan] ({trainable_pct:.2f}% of the model).\n"
        f"The saved adapter file is only [bold cyan]~{adapter_mem_mb:.2f} MB[/bold cyan], allowing instant distribution and dynamic swapping!",
        border_style="green"
    ))


if __name__ == "__main__":
    analyze_model()
