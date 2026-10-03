# 🧠 Qwen2.5 LoRA & QLoRA Lab (CPU & Cloud GPU)

A self-contained project demonstrating parameter-efficient fine-tuning (PEFT) on **Qwen2.5-0.5B-Instruct**.

- **LoRA on CPU**: Trained locally on standard Intel Core i5 CPU in ~3 minutes using PyTorch & Hugging Face PEFT.
- **QLoRA on GPU**: Ready-to-run 4-bit NormalFloat (`NF4`) pipeline with `bitsandbytes` (for Google Colab or NVIDIA GPUs).
- **Ollama Integration**: Built-in script to merge adapters and run the model natively with `ollama run opspilot`.

---

## 📁 Project Structure

```
/home/kranthiveer/Documents/Qlora/
├── config.py                 # Central hyperparameter & device configurations
├── benchmark.py              # Architecture & memory footprint analyzer
├── train_lora.py             # CPU-optimized LoRA training script
├── train_qlora.py            # 4-bit NF4 QLoRA script (GPU / Colab)
├── inference.py              # Side-by-side comparison & interactive CLI
├── export_to_ollama.py       # Adapter merger & Ollama Modelfile generator
├── data/
│   ├── devops_dataset.jsonl  # Curated Linux/DevOps dataset
│   ├── create_dataset.py     # Dataset generator
│   └── dataset_loader.py     # Qwen ChatML formatter & tokenizer
├── notebooks/
│   └── qlora_colab.ipynb     # 1-click Google Colab notebook for 4-bit QLoRA
├── outputs/
│   ├── qwen_lora_adapter/    # Trained LoRA weights (~15 MB)
│   └── qwen_merged_model/    # Merged full model + Ollama Modelfile
└── requirements.txt          # Frozen environment dependencies
```

---

## ⚡ Quickstart Commands

All commands run inside the virtual environment (`.venv`):

### 1. View Architecture & Memory Benchmark
Compares standard LoRA vs QLoRA 4-bit memory footprints and trainable parameters:
```bash
.venv/bin/python benchmark.py
```

### 2. Fine-Tune with LoRA on CPU
Trains the model on CPU in ~3 minutes:
```bash
.venv/bin/python train_lora.py
```
*Trained on 494M parameter base model with only 1.08M trainable adapter parameters (0.22% of weights).*

### 3. Run Head-to-Head Comparison (Base vs. LoRA)
```bash
.venv/bin/python inference.py "Find all processes running on port 3000 and terminate them immediately."
```

### 4. Interactive Chat Mode
Chat with your fine-tuned assistant directly in your terminal:
```bash
.venv/bin/python inference.py --chat
```

### 5. Export and Run in Ollama
Merge the adapter weights and generate an Ollama model:
```bash
.venv/bin/python export_to_ollama.py
```
Then import and run in Ollama:
```bash
cd outputs/qwen_merged_model
ollama create opspilot -f Modelfile
ollama run opspilot
```

### 6. Run QLoRA (4-Bit NF4) on Cloud GPU
To train the larger **Qwen2.5-7B** or test 4-bit NormalFloat quantization on a free GPU:
Open `notebooks/qlora_colab.ipynb` in [Google Colab](https://colab.research.google.com/) with a free T4 GPU runtime.
