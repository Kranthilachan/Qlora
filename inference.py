"""
Inference & Head-to-Head Comparison Script.
Compares Base Qwen2.5-0.5B vs. LoRA Fine-Tuned Qwen2.5-0.5B side-by-side.
"""
import os
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from rich.console import Console
from rich.panel import Panel
from rich.columns import Columns

from config import ModelConfig, TrainingConfig
from data.dataset_loader import format_chatml_prompt, SYSTEM_PROMPT

console = Console()


def generate_response(model, tokenizer, prompt_text: str, max_new_tokens: int = 150) -> str:
    """Generates text from the model given a ChatML formatted prompt."""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy for deterministic, reproducible comparison
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Slice only generated new tokens
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return response


def run_comparison(test_query: str):
    model_cfg = ModelConfig()
    train_cfg = TrainingConfig()

    console.print(Panel(
        f"[bold yellow]Test Query:[/bold yellow] [bold white]{test_query}[/bold white]",
        title="🔍 Comparison Query",
        border_style="cyan"
    ))

    # 1. Load Tokenizer & Base Model
    console.print("[dim]Loading base Qwen2.5-0.5B model...[/dim]")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_cfg.model_name_or_path,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    base_model.eval()

    chatml_prompt = format_chatml_prompt(test_query, SYSTEM_PROMPT)

    # 2. Base Model Generation
    console.print("[dim]Generating output with Base Model (Zero-Shot)...[/dim]")
    t0 = time.time()
    base_output = generate_response(base_model, tokenizer, chatml_prompt)
    base_time = time.time() - t0

    # 3. Check if LoRA Adapter exists
    if not os.path.exists(train_cfg.output_dir):
        console.print(Panel(
            base_output,
            title="Base Model Output (LoRA adapter not found. Run train_lora.py first!)",
            border_style="yellow"
        ))
        return

    # 4. Load LoRA Adapter onto Base Model
    console.print(f"[dim]Attaching LoRA Adapter from {train_cfg.output_dir}...[/dim]")
    lora_model = PeftModel.from_pretrained(base_model, train_cfg.output_dir)
    lora_model.eval()

    # 5. LoRA Model Generation
    console.print("[dim]Generating output with LoRA Fine-Tuned Model...[/dim]")
    t1 = time.time()
    lora_output = generate_response(lora_model, tokenizer, chatml_prompt)
    lora_time = time.time() - t1

    # 6. Display Comparison Side by Side
    panel_base = Panel(
        base_output,
        title=f"🛑 Base Qwen2.5-0.5B ({base_time:.2f}s)",
        subtitle="Unfine-tuned (Raw / Free-form)",
        border_style="red",
        width=55,
    )
    panel_lora = Panel(
        lora_output,
        title=f"🚀 LoRA Fine-Tuned Qwen ({lora_time:.2f}s)",
        subtitle="Adapted to OpsPilot Strict JSON",
        border_style="green",
        width=55,
    )

    console.print(Columns([panel_base, panel_lora]))


def interactive_mode():
    train_cfg = TrainingConfig()
    model_cfg = ModelConfig()

    if not os.path.exists(train_cfg.output_dir):
        console.print("[bold red]LoRA adapter not found![/bold red] Please run [yellow]python train_lora.py[/yellow] first.")
        return

    console.print("[bold green]Loading fine-tuned model for interactive chat...[/bold green]")
    tokenizer = AutoTokenizer.from_pretrained(train_cfg.output_dir, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_cfg.model_name_or_path,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    lora_model = PeftModel.from_pretrained(base_model, train_cfg.output_dir)
    lora_model.eval()

    console.print("\n[bold cyan]OpsPilot Interactive Shell (type 'exit' to quit)[/bold cyan]")
    while True:
        try:
            query = input("\n[OpsPilot Query] > ").strip()
            if not query or query.lower() in ("exit", "quit", "q"):
                break
            prompt = format_chatml_prompt(query, SYSTEM_PROMPT)
            response = generate_response(lora_model, tokenizer, prompt)
            console.print(Panel(response, title="OpsPilot Response (JSON)", border_style="green"))
        except (KeyboardInterrupt, EOFError):
            break


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--chat":
        interactive_mode()
    else:
        sample_query = (
            sys.argv[1] if len(sys.argv) > 1 
            else "Find all processes running on port 3000 and terminate them immediately."
        )
        run_comparison(sample_query)
