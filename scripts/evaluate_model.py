#!/usr/bin/env python3
"""
Evaluation script for fine-tuned Qwen 2.5 Omni model
"""

import os
import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_path: str, base_model_name: str = None, use_lora: bool = True):
    """Load the fine-tuned model and tokenizer."""
    
    if base_model_name is None:
        base_model_name = "Qwen/Qwen2.5-Omni-7B"
    
    logger.info(f"Loading base model: {base_model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path if not use_lora else base_model_name,
        trust_remote_code=True
    )
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load LoRA weights if using LoRA
    if use_lora and os.path.exists(model_path):
        logger.info(f"Loading LoRA weights from {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
    
    return model, tokenizer

def evaluate_model(model, tokenizer, test_data_path: str, max_new_tokens: int = 512):
    """Evaluate the model on test data."""
    
    logger.info(f"Loading test data from {test_data_path}")
    
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    results = []
    
    for i, item in enumerate(test_data):
        logger.info(f"Processing item {i+1}/{len(test_data)}")
        
        input_text = item['input']
        expected_output = item['output']
        
        # Tokenize input
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": input_text}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        
        result = {
            "input": input_text,
            "expected": expected_output,
            "generated": response,
            "length": len(response)
        }
        
        results.append(result)
        
        # Print result
        print(f"\n--- Test Case {i+1} ---")
        print(f"Input: {input_text}")
        print(f"Expected: {expected_output}")
        print(f"Generated: {response}")
        print("-" * 50)
    
    return results

def save_results(results: list, output_path: str):
    """Save evaluation results."""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results saved to {output_path}")
    
    # Print summary
    total_tests = len(results)
    avg_length = sum(r['length'] for r in results) / total_tests
    
    print(f"\n📊 Evaluation Summary:")
    print(f"Total tests: {total_tests}")
    print(f"Average response length: {avg_length:.1f} characters")
    print(f"Results saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Qwen 2.5 Omni model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-Omni-7B", help="Base model name")
    parser.add_argument("--test_data", type=str, default="./data/test_data.json", help="Path to test data")
    parser.add_argument("--output", type=str, default="./evaluation_results.json", help="Output file for results")
    parser.add_argument("--use_lora", action="store_true", default=True, help="Whether using LoRA")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum new tokens to generate")
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        args.model_path, 
        args.base_model, 
        args.use_lora
    )
    
    # Evaluate model
    results = evaluate_model(
        model, 
        tokenizer, 
        args.test_data, 
        args.max_new_tokens
    )
    
    # Save results
    save_results(results, args.output)

if __name__ == "__main__":
    main()
