#!/usr/bin/env python3
"""
Create a sample dataset for Qwen 2.5 Omni fine-tuning
"""

import json
import os

def create_sample_dataset():
    """Create a sample multi-modal dataset for Qwen 2.5 Omni."""
    
    # Sample data for multi-modal fine-tuning
    sample_data = [
        {
            "input": "请描述这张图片中的内容。",
            "output": "这是一张美丽的风景图片，展示了蓝天白云下的绿色山脉和清澈的湖泊。图片构图优美，色彩丰富，给人一种宁静祥和的感觉。",
            "image_path": "images/sample1.jpg"  # Optional: for multi-modal training
        },
        {
            "input": "解释一下什么是人工智能？",
            "output": "人工智能（AI）是指让机器模拟人类智能的技术。它包括机器学习、深度学习、自然语言处理等多个领域。AI可以帮助我们解决复杂问题，提高工作效率，改善生活质量。",
            "image_path": None
        },
        {
            "input": "请分析这段音频的内容。",
            "output": "这段音频包含了一段清晰的对话，语速适中，发音标准。主要内容涉及技术讨论，说话者表达了专业的观点和建议。",
            "audio_path": "audio/sample1.wav"  # Optional: for audio processing
        },
        {
            "input": "总结一下这个视频的主要观点。",
            "output": "这个视频主要讨论了当前技术发展的趋势，强调了创新和合作的重要性。演讲者提出了几个关键建议：持续学习、拥抱变化、建立团队合作。整个演讲结构清晰，逻辑性强。",
            "video_path": "videos/sample1.mp4"  # Optional: for video processing
        },
        {
            "input": "帮我写一个Python函数来计算斐波那契数列。",
            "output": "```python\ndef fibonacci(n):\n    \"\"\"\n    计算斐波那契数列的第n项\n    \n    Args:\n        n (int): 要计算的项数\n    \n    Returns:\n        int: 斐波那契数列的第n项\n    \"\"\"\n    if n <= 1:\n        return n\n    \n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    \n    return b\n\n# 示例用法\nprint(fibonacci(10))  # 输出: 55\n```",
            "image_path": None
        },
        {
            "input": "请翻译这句话：'Hello, how are you today?'",
            "output": "这句话的中文翻译是：'你好，你今天怎么样？'",
            "image_path": None
        },
        {
            "input": "解释一下机器学习中的过拟合问题。",
            "output": "过拟合（Overfitting）是机器学习中的一个常见问题，指模型在训练数据上表现很好，但在新的、未见过的数据上表现较差的现象。\n\n主要原因包括：\n1. 模型过于复杂，参数过多\n2. 训练数据不足\n3. 训练时间过长\n\n解决方法：\n1. 增加训练数据\n2. 使用正则化技术\n3. 减少模型复杂度\n4. 使用交叉验证\n5. 早停（Early Stopping）",
            "image_path": None
        },
        {
            "input": "请分析这张图片中的数学公式。",
            "output": "这张图片展示了一个复杂的数学公式，看起来是一个积分表达式。公式涉及三角函数和指数函数，可能用于描述某种物理现象或数学关系。具体的推导过程需要更多的上下文信息。",
            "image_path": "images/math_formula.jpg"
        }
    ]
    
    # Create data directory if it doesn't exist
    os.makedirs("./data", exist_ok=True)
    
    # Save dataset
    output_path = "./data/train_data.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Sample dataset created at {output_path}")
    print(f"📊 Dataset contains {len(sample_data)} samples")
    print("📝 Dataset format includes:")
    print("  - Text-only conversations")
    print("  - Multi-modal inputs (image, audio, video)")
    print("  - Code generation examples")
    print("  - Translation tasks")
    print("  - Technical explanations")
    
    # Create additional dataset files
    create_validation_dataset()
    create_test_dataset()
    
    return output_path

def create_validation_dataset():
    """Create validation dataset."""
    validation_data = [
        {
            "input": "请解释什么是深度学习？",
            "output": "深度学习是机器学习的一个子领域，它使用多层神经网络来模拟人脑的工作方式。深度学习能够自动学习数据中的复杂模式和特征，在图像识别、语音处理、自然语言理解等领域取得了突破性进展。",
            "image_path": None
        },
        {
            "input": "写一个简单的排序算法。",
            "output": "```python\ndef bubble_sort(arr):\n    \"\"\"\n    冒泡排序算法\n    \"\"\"\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n - i - 1):\n            if arr[j] > arr[j + 1]:\n                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n    return arr\n\n# 测试\nnumbers = [64, 34, 25, 12, 22, 11, 90]\nsorted_numbers = bubble_sort(numbers)\nprint(sorted_numbers)  # [11, 12, 22, 25, 34, 64, 90]\n```",
            "image_path": None
        }
    ]
    
    with open("./data/validation_data.json", 'w', encoding='utf-8') as f:
        json.dump(validation_data, f, ensure_ascii=False, indent=2)
    
    print("✅ Validation dataset created")

def create_test_dataset():
    """Create test dataset."""
    test_data = [
        {
            "input": "请分析这个代码的时间复杂度。",
            "output": "这个代码的时间复杂度是O(n²)。外层循环执行n次，内层循环在最坏情况下也执行n次，因此总体时间复杂度为O(n²)。",
            "image_path": None
        },
        {
            "input": "解释一下什么是区块链技术？",
            "output": "区块链是一种分布式账本技术，它通过密码学方法将数据块按时间顺序连接起来，形成一个不可篡改的链条。每个区块包含前一个区块的哈希值，确保了数据的安全性和完整性。区块链技术广泛应用于数字货币、智能合约、供应链管理等领域。",
            "image_path": None
        }
    ]
    
    with open("./data/test_data.json", 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    print("✅ Test dataset created")

if __name__ == "__main__":
    create_sample_dataset()
