"""
Export and Merge LoRA Adapter for Ollama Integration.
1. Merges LoRA delta weights permanently into base model (merge_and_unload).
2. Generates an Ollama Modelfile configured for Qwen ChatML.
3. Provides 1-command import to run with `ollama run opspilot`.
"""
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from rich.console import Console
from rich.panel import Panel

from config import ModelConfig, TrainingConfig
from data.dataset_loader import SYSTEM_PROMPT

console = Console()


def merge_and_export():
    model_cfg = ModelConfig()
    train_cfg = TrainingConfig()

    if not os.path.exists(train_cfg.output_dir):
        console.print(f"[bold red]LoRA adapter directory {train_cfg.output_dir} not found![/bold red]")
        console.print("Please train the adapter first with: [yellow]python train_lora.py[/yellow]")
        return

    console.print(Panel.fit(
        "[bold cyan]Merging LoRA Adapter into Qwen Base Model[/bold cyan]\n"
        f"Adapter Path: [yellow]{train_cfg.output_dir}[/yellow]\n"
        f"Merged Output: [green]{train_cfg.merged_output_dir}[/green]",
        border_style="cyan"
    ))

    # 1. Load Base Model and Tokenizer
    console.print("[dim]Loading base model and tokenizer...[/dim]")
    tokenizer = AutoTokenizer.from_pretrained(train_cfg.output_dir, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_cfg.model_name_or_path,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # 2. Attach PEFT Adapter
    console.print("[dim]Attaching LoRA adapter weights...[/dim]")
    lora_model = PeftModel.from_pretrained(base_model, train_cfg.output_dir)

    # 3. Merge Weights Permanently
    console.print("[bold green]Merging weights (merge_and_unload)...[/bold green]")
    merged_model = lora_model.merge_and_unload()

    # 4. Save Merged Model
    os.makedirs(train_cfg.merged_output_dir, exist_ok=True)
    console.print(f"[dim]Saving merged weights and tokenizer to {train_cfg.merged_output_dir}...[/dim]")
    merged_model.save_pretrained(train_cfg.merged_output_dir)
    tokenizer.save_pretrained(train_cfg.merged_output_dir)

    # 5. Create Ollama Modelfile
    modelfile_path = os.path.join(train_cfg.merged_output_dir, "Modelfile")
    modelfile_content = f"""FROM .
TEMPLATE \"\"\"{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
{{{{ end }}}}\"\"\"
SYSTEM \"\"\"{SYSTEM_PROMPT}\"\"\"
PARAMETER temperature 0.1
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
"""
    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(modelfile_content)

    console.print(Panel(
        f"[bold green]Successfully merged and exported![/bold green]\n\n"
        f"Model files saved to: [yellow]{train_cfg.merged_output_dir}[/yellow]\n"
        f"Ollama Modelfile generated: [yellow]{modelfile_path}[/yellow]\n\n"
        f"[bold cyan]To run directly in your local Ollama:[/bold cyan]\n"
        f"1. cd {os.path.abspath(train_cfg.merged_output_dir)}\n"
        f"2. ollama create opspilot -f Modelfile\n"
        f"3. ollama run opspilot\n",
        title="Ollama Ready",
        border_style="green"
    ))


if __name__ == "__main__":
    merge_and_export()
