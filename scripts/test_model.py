#!/usr/bin/env python3
"""
Test the fine-tuned Qwen 2.5 model
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def load_model(model_path, base_model_name):
    """Load the fine-tuned model."""
    print(f"📥 Loading base model: {base_model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load LoRA weights
    print(f"📥 Loading LoRA weights from {model_path}")
    model = PeftModel.from_pretrained(model, model_path)
    
    return model, tokenizer

def test_model(model, tokenizer, test_questions):
    """Test the model with sample questions."""
    print("🧪 Testing the fine-tuned model...")
    print("=" * 60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n📝 Test {i}: {question}")
        print("-" * 40)
        
        # Create conversation format
        conversation = [{"role": "user", "content": question}]
        
        # Apply chat template
        input_text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = tokenizer(input_text, return_tensors="pt")
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        
        print(f"🤖 Answer: {response}")
        print("-" * 40)

def main():
    # Model paths
    model_path = "./models/qwen25_simple_sft"
    base_model_name = "Qwen/Qwen2.5-7B-Instruct"
    
    # Load model
    model, tokenizer = load_model(model_path, base_model_name)
    
    # Test questions
    test_questions = [
        "请描述这张图片中的内容。",
        "解释一下什么是人工智能？",
        "写一个Python函数来计算斐波那契数列。",
        "请翻译这句话：'Hello, how are you today?'",
        "分析一下机器学习中的过拟合问题。"
    ]
    
    # Test the model
    test_model(model, tokenizer, test_questions)
    
    print("\n✅ Model testing completed!")

if __name__ == "__main__":
    main()
