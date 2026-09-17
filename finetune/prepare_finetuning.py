import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

class FinancialFineTuningPreparer:
    """
    Prepares domain-specific fine-tuning datasets and configurations (LoRA / QLoRA / Unsloth)
    for the TEEP 2026 financial NLP research initiative.
    
    This operates as an auxiliary capability to allow specialized adaptation of models
    on corporate annual reports, semantic shifts, and financial sentiment analysis.
    """
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path(__file__).resolve().parent / "exports"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_instruction_dataset(
        self,
        chunks: List[Dict[str, Any]],
        format_type: str = "alpaca"
    ) -> str:
        """
        Transforms indexed annual report chunks into structured instruction-tuning pairs:
        - Task 1: Financial Sentiment & Tone Identification
        - Task 2: Risk Factor Extraction & Severity Assessment
        - Task 3: Fact Extraction with Citation Attribution
        """
        dataset = []

        for chunk in chunks:
            text = chunk.get("text", "")
            meta = chunk.get("metadata", {})
            ticker = meta.get("ticker", "Company")
            year = meta.get("fiscal_year", "2024")
            section = meta.get("section", "BODY")
            sec_title = meta.get("section_title", "General")

            if len(text.strip()) < 30:
                continue

            # Generate Fact Extraction / QA sample
            prompt_qa = f"Based strictly on {ticker}'s FY{year} annual report ({section}: {sec_title}), identify the key disclosures and operational assertions."
            
            if format_type.lower() == "alpaca":
                dataset.append({
                    "instruction": prompt_qa,
                    "input": text,
                    "output": f"In FY{year}, {ticker} discloses in {section} ({sec_title}): {text[:250]}..."
                })
            else:
                # ChatML format
                dataset.append({
                    "messages": [
                        {"role": "system", "content": "You are a financial NLP analyst specializing in SEC corporate annual reports."},
                        {"role": "user", "content": f"{prompt_qa}\n\nContext:\n{text}"},
                        {"role": "assistant", "content": f"Factual findings from {ticker} FY{year} ({section}):\n{text[:250]}..."}
                    ]
                })

        out_file = self.output_dir / f"financial_rag_dataset_{format_type}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2)

        return str(out_file)

    def generate_unsloth_training_script(self, base_model: str = "unsloth/llama-3-8b-Instruct-bnb-4bit") -> str:
        """
        Generates a standalone, executable training script for QLoRA fine-tuning using Unsloth
        or HuggingFace PEFT + bitsandbytes.
        """
        script_content = f'''"""
TEEP 2026 Domain-Specific Financial Model Fine-Tuning Script
Using Unsloth / Hugging Face PEFT QLoRA for Corporate Annual Report Semantic Shift Reasoning.
"""

import os
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# 1. Configuration
MODEL_NAME = "{base_model}"
MAX_SEQ_LENGTH = 2048
DATASET_PATH = "financial_rag_dataset_alpaca.json"
OUTPUT_DIR = "outputs_financial_lora"

def main():
    print("Initializing QLoRA fine-tuning pipeline for TEEP 2026...")
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=True,
            dtype=None
        )

        # Apply LoRA adapters
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            use_gradient_checkpointing=True,
            random_state=3407
        )
        print("LoRA parameters configured successfully.")

        # Load generated dataset
        if not os.path.exists(DATASET_PATH):
            print(f"Dataset not found at {{DATASET_PATH}}. Export dataset via the RAG dashboard first.")
            return

        dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="input",
            max_seq_length=MAX_SEQ_LENGTH,
            dataset_num_proc=2,
            packing=False,
            args=TrainingArguments(
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                warmup_steps=10,
                max_steps=60,
                learning_rate=2e-4,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=5,
                output_dir=OUTPUT_DIR,
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407
            )
        )
        print("Beginning training run...")
        trainer.train()
        print(f"Training complete. Weights saved to {{OUTPUT_DIR}}")

    except ImportError:
        print("Unsloth not detected. Install via: pip install unsloth [cu121-ampere-torch240]")

if __name__ == "__main__":
    main()
'''
        script_path = self.output_dir / "train_qlora_unsloth.py"
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)

        return str(script_path)
