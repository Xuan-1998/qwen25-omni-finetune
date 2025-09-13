#!/usr/bin/env python3
"""
Simple Qwen 2.5 Omni Fine-tuning Script (without flash-attention)
"""

import os
import json
import torch
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType

def load_data(dataset_path):
    """Load training data."""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def preprocess_data(data, tokenizer):
    """Preprocess data for training."""
    processed_data = []
    
    for item in data:
        # Create conversation format
        conversation = [
            {"role": "user", "content": item['input']},
            {"role": "assistant", "content": item['output']}
        ]
        
        # Apply chat template
        text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False
        )
        
        processed_data.append({"text": text})
    
    return Dataset.from_list(processed_data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset_path", type=str, default="./data/train_data.json")
    parser.add_argument("--output_dir", type=str, default="./models/qwen25_simple_sft")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    
    args = parser.parse_args()
    
    print("🚀 Starting simple Qwen 2.5 Omni fine-tuning...")
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {args.output_dir}")
    
    # Load tokenizer
    print("📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print("📥 Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Apply LoRA
    print("🔧 Applying LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load and preprocess data
    print("📊 Loading data...")
    data = load_data(args.dataset_path)
    print(f"Loaded {len(data)} samples")
    
    processed_data = preprocess_data(data, tokenizer)
    
    # Data collator
    def collate_fn(examples):
        # Tokenize
        texts = [example["text"] for example in examples]
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=1024,
            return_tensors="pt"
        )
        
        # For causal LM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        logging_steps=1,
        save_steps=10,
        save_total_limit=2,
        remove_unused_columns=False,
        fp16=True,
        dataloader_num_workers=0,  # Avoid multiprocessing issues
        report_to=None,  # Disable wandb/tensorboard
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_data,
        data_collator=collate_fn,
        tokenizer=tokenizer,
    )
    
    # Start training
    print("🏋️ Starting training...")
    trainer.train()
    
    # Save model
    print("💾 Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    print("✅ Training completed!")

if __name__ == "__main__":
    main()
