import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig


DEFAULT_TARGET_MODULES: Dict[str, List[str]] = {
    # Llama-family (Llama, Mistral, Mixtral, Qwen2, Gemma2 mostly share these names)
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mixtral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "qwen2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gemma": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gemma2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    # Others
    "falcon": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    "gpt2": ["c_attn", "c_proj", "c_fc"],
    "phi": ["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"],
}


def guess_target_modules(model_type: str) -> List[str]:
    model_type = (model_type or "").lower()
    for key, modules in DEFAULT_TARGET_MODULES.items():
        if key in model_type:
            return modules
    # Fallback to common set
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def make_bnb_config(load_in_4bit: bool, compute_dtype: torch.dtype) -> Optional[BitsAndBytesConfig]:
    if not load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def build_text(example: Dict[str, str]) -> str:
    # Simple instruction-following format. The model learns to output compact JSON.
    prompt = example["prompt"].strip()
    response = example["response"].strip()
    return f"{prompt}\n\nAnswer: {response}"


def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for FD calculators")
    parser.add_argument("--base_model", required=True, help="Base model name or local path")
    parser.add_argument("--output_dir", required=True, help="Directory to save the adapter")
    parser.add_argument("--train_path", default="/workspace/data/fd_train.jsonl")
    parser.add_argument("--eval_path", default="/workspace/data/fd_eval.jsonl")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_train_epochs", type=float, default=2)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", default="cosine")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--no_4bit", action="store_true", help="Force disable 4-bit load (overrides --load_in_4bit)")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prefers_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if (args.bf16 or prefers_bf16) else torch.float16

    use_4bit = False
    if not args.no_4bit:
        use_4bit = args.load_in_4bit and torch.cuda.is_available()

    bnb_config = make_bnb_config(use_4bit, compute_dtype)

    print(f"Loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        torch_dtype=compute_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_type = getattr(model.config, "model_type", "")
    target_modules = guess_target_modules(model_type)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
        bias="none",
    )

    # Load datasets
    train_ds = load_dataset("json", data_files=args.train_path, split="train")
    eval_ds = load_dataset("json", data_files=args.eval_path, split="train")

    def formatting_func(examples):
        texts = []
        for p, r in zip(examples["prompt"], examples["response"]):
            texts.append(f"{p}\n\nAnswer: {r}")
        return {"text": texts}

    # Map to 'text' field so SFTTrainer can use dataset_text_field
    train_ds = train_ds.map(formatting_func, batched=True, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(formatting_func, batched=True, remove_columns=eval_ds.column_names)

    training_config = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=(compute_dtype == torch.bfloat16),
        fp16=(compute_dtype == torch.float16),
        seed=args.seed,
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        packing=False,
        report_to=["none"],
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
    )

    trainer.train()

    print("Saving adapter and tokenizer...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
