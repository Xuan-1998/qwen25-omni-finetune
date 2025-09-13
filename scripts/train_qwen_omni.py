#!/usr/bin/env python3
"""
Qwen 2.5 Omni Fine-tuning Script
Supports both full fine-tuning and LoRA/QLoRA methods
"""

import os
import json
import argparse
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    HfArgumentParser,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import transformers

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """Arguments pertaining to model configuration."""
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code when loading model"}
    )
    use_lora: bool = field(
        default=True,
        metadata={"help": "Whether to use LoRA for efficient fine-tuning"}
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "LoRA rank"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha parameter"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA dropout"}
    )
    target_modules: str = field(
        default="q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Target modules for LoRA, comma-separated"}
    )

@dataclass
class DataArguments:
    """Arguments pertaining to data configuration."""
    dataset_path: str = field(
        metadata={"help": "Path to training dataset"}
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length"}
    )
    preprocessing_num_workers: int = field(
        default=4,
        metadata={"help": "Number of workers for preprocessing"}
    )

@dataclass
class TrainingArguments(TrainingArguments):
    """Training arguments."""
    output_dir: str = field(default="./results")
    logging_dir: str = field(default="./logs")
    per_device_train_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=8)
    learning_rate: float = field(default=1e-5)
    num_train_epochs: int = field(default=3)
    save_steps: int = field(default=500)
    logging_steps: int = field(default=10)
    warmup_steps: int = field(default=100)
    fp16: bool = field(default=True)
    remove_unused_columns: bool = field(default=False)
    dataloader_num_workers: int = field(default=4)

def load_and_preprocess_data(dataset_path: str, tokenizer, max_seq_length: int):
    """Load and preprocess the training dataset."""
    logger.info(f"Loading dataset from {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    def preprocess_function(examples):
        """Preprocess function for the dataset."""
        inputs = examples['input']
        targets = examples['output']
        
        # Create conversation format for Qwen 2.5 Omni
        conversations = []
        for inp, tgt in zip(inputs, targets):
            conversation = [
                {"role": "user", "content": inp},
                {"role": "assistant", "content": tgt}
            ]
            conversations.append(conversation)
        
        # Tokenize
        model_inputs = tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt"
        )
        
        # Create labels (same as input_ids for causal LM)
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        
        return model_inputs
    
    # Create dataset
    dataset = Dataset.from_dict({
        'input': [item['input'] for item in data],
        'output': [item['output'] for item in data]
    })
    
    # Apply preprocessing
    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Preprocessing dataset"
    )
    
    return processed_dataset

def setup_model_and_tokenizer(model_args: ModelArguments):
    """Setup model and tokenizer."""
    logger.info(f"Loading model and tokenizer from {model_args.model_name_or_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="right"
    )
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Apply LoRA if requested
    if model_args.use_lora:
        logger.info("Applying LoRA configuration")
        target_modules = model_args.target_modules.split(',')
        
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, tokenizer

def main():
    """Main training function."""
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_args)
    
    # Load and preprocess data
    train_dataset = load_and_preprocess_data(
        data_args.dataset_path,
        tokenizer,
        data_args.max_seq_length
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Save model
    logger.info(f"Saving model to {training_args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
