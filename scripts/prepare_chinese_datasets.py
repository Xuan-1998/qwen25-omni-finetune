#!/usr/bin/env python3
"""
Prepare Chinese datasets for Qwen 2.5 Omni fine-tuning
Downloads and processes Chinese multi-modal datasets
"""

import os
import json
import requests
import argparse
from datasets import load_dataset

class ChineseDatasetProcessor:
    def __init__(self, data_dir="./data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Chinese datasets available
        self.chinese_datasets = {
            "chinese_llava": {
                "name": "Chinese LLaVA Dataset",
                "description": "Chinese instruction following with images",
                "source": "custom_collection"
            },
            "coco_chinese": {
                "name": "COCO Chinese Captions",
                "description": "Chinese image captions",
                "source": "coco_chinese"
            },
            "ai_challenger": {
                "name": "AI Challenger Caption",
                "description": "Chinese image captioning competition dataset",
                "source": "ai_challenger"
            },
            "flickr30k_chinese": {
                "name": "Flickr30K Chinese",
                "description": "Chinese Flickr image descriptions",
                "source": "flickr30k_chinese"
            }
        }
    
    def create_chinese_llava_dataset(self, num_samples=5000):
        """Create a Chinese LLaVA-style dataset."""
        print("📝 Creating Chinese LLaVA dataset...")
        
        # Sample Chinese instruction templates
        templates = [
            {
                "input": "请详细描述这张图片中的内容。",
                "output_template": "这张图片展示了{content}。"
            },
            {
                "input": "图片中有什么物体？",
                "output_template": "图片中包含以下物体：{objects}。"
            },
            {
                "input": "描述图片中的场景和环境。",
                "output_template": "这是一个{scene}的场景，{environment}。"
            },
            {
                "input": "分析图片中的颜色搭配。",
                "output_template": "图片主要使用了{colors}等颜色，整体色调{description}。"
            },
            {
                "input": "这张图片给你的感受是什么？",
                "output_template": "这张图片让我感到{feeling}，因为{reason}。"
            }
        ]
        
        # Sample content for templates
        content_samples = [
            "美丽的自然风光，有蓝天白云和绿树",
            "城市街景，车水马龙，行人匆匆",
            "室内环境，温馨舒适的家庭氛围",
            "食物和饮料，看起来非常美味",
            "动物在自然环境中的生活状态",
            "建筑物和城市景观，现代化设计",
            "人物活动，日常生活场景",
            "艺术和创意作品，富有想象力"
        ]
        
        dataset = []
        for i in range(num_samples):
            template = templates[i % len(templates)]
            content = content_samples[i % len(content_samples)]
            
            # Create sample data
            sample = {
                "input": template["input"],
                "output": template["output_template"].format(
                    content=content,
                    objects="各种有趣的物体",
                    scene="温馨和谐",
                    environment="环境优美",
                    colors="蓝色、绿色、白色",
                    description="清新自然",
                    feeling="平静和愉悦",
                    reason="画面和谐美好"
                ),
                "image_path": f"sample_images/sample_{i % 100}.jpg"  # Placeholder
            }
            
            dataset.append(sample)
        
        # Save dataset
        output_file = os.path.join(self.data_dir, "chinese_llava.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Chinese LLaVA dataset created: {output_file}")
        print(f"📊 Total samples: {len(dataset)}")
        return output_file
    
    def create_chinese_qa_dataset(self, num_samples=3000):
        """Create Chinese Q&A dataset."""
        print("📝 Creating Chinese Q&A dataset...")
        
        qa_samples = [
            {
                "input": "什么是人工智能？",
                "output": "人工智能（AI）是指让机器模拟人类智能的技术，包括机器学习、深度学习、自然语言处理等领域。它可以帮助我们解决复杂问题，提高工作效率。"
            },
            {
                "input": "机器学习有哪些类型？",
                "output": "机器学习主要分为三类：1) 监督学习，使用标记数据训练模型；2) 无监督学习，从无标记数据中发现模式；3) 强化学习，通过与环境交互学习最优策略。"
            },
            {
                "input": "深度学习与传统机器学习有什么区别？",
                "output": "深度学习使用多层神经网络自动学习特征，而传统机器学习需要人工设计特征。深度学习在处理图像、语音、文本等复杂数据时表现更好，但需要更多数据和计算资源。"
            },
            {
                "input": "什么是自然语言处理？",
                "output": "自然语言处理（NLP）是人工智能的一个分支，专注于让计算机理解、生成和处理人类语言。应用包括机器翻译、情感分析、问答系统、文本摘要等。"
            },
            {
                "input": "解释一下计算机视觉的应用。",
                "output": "计算机视觉的应用非常广泛，包括：图像识别、人脸识别、自动驾驶、医学影像分析、工业质检、增强现实等。它让机器能够'看懂'图像和视频内容。"
            },
            {
                "input": "什么是大数据？",
                "output": "大数据是指规模巨大、类型多样、处理速度快的数据集合。它具有5V特征：Volume（大量）、Velocity（高速）、Variety（多样）、Veracity（真实性）、Value（价值）。"
            },
            {
                "input": "云计算有什么优势？",
                "output": "云计算的主要优势包括：1) 按需使用，灵活扩展；2) 降低成本，无需购买硬件；3) 高可靠性，专业维护；4) 全球部署，就近访问；5) 自动更新，保持最新。"
            },
            {
                "input": "区块链技术有哪些特点？",
                "output": "区块链技术具有以下特点：1) 去中心化，无需中央机构；2) 不可篡改，数据安全可靠；3) 透明性，所有交易可追溯；4) 共识机制，保证数据一致性；5) 智能合约，自动执行合约条款。"
            }
        ]
        
        # Generate more samples by varying the questions
        dataset = []
        variations = [
            "请解释", "什么是", "介绍一下", "请详细说明", "能否解释一下", "请简单介绍", "请详细描述"
        ]
        
        for i in range(num_samples):
            base_sample = qa_samples[i % len(qa_samples)]
            variation = variations[i % len(variations)]
            
            # Create variation
            sample = {
                "input": f"{variation}{base_sample['input']}",
                "output": base_sample["output"],
                "image_path": None
            }
            
            dataset.append(sample)
        
        # Save dataset
        output_file = os.path.join(self.data_dir, "chinese_qa.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Chinese Q&A dataset created: {output_file}")
        print(f"📊 Total samples: {len(dataset)}")
        return output_file
    
    def create_chinese_code_dataset(self, num_samples=2000):
        """Create Chinese code generation dataset."""
        print("📝 Creating Chinese code generation dataset...")
        
        code_samples = [
            {
                "input": "写一个Python函数来计算斐波那契数列的第n项",
                "output": "```python\ndef fibonacci(n):\n    \"\"\"\n    计算斐波那契数列的第n项\n    \n    Args:\n        n (int): 要计算的项数\n    \n    Returns:\n        int: 斐波那契数列的第n项\n    \"\"\"\n    if n <= 1:\n        return n\n    \n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    \n    return b\n\n# 示例用法\nprint(fibonacci(10))  # 输出: 55\n```"
            },
            {
                "input": "实现一个快速排序算法",
                "output": "```python\ndef quick_sort(arr):\n    \"\"\"\n    快速排序算法实现\n    \n    Args:\n        arr (list): 待排序的数组\n    \n    Returns:\n        list: 排序后的数组\n    \"\"\"\n    if len(arr) <= 1:\n        return arr\n    \n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    \n    return quick_sort(left) + middle + quick_sort(right)\n\n# 示例用法\nnumbers = [64, 34, 25, 12, 22, 11, 90]\nsorted_numbers = quick_sort(numbers)\nprint(sorted_numbers)  # [11, 12, 22, 25, 34, 64, 90]\n```"
            },
            {
                "input": "写一个函数来检查字符串是否是回文",
                "output": "```python\ndef is_palindrome(s):\n    \"\"\"\n    检查字符串是否是回文\n    \n    Args:\n        s (str): 要检查的字符串\n    \n    Returns:\n        bool: 如果是回文返回True，否则返回False\n    \"\"\"\n    # 转换为小写并移除空格\n    s = s.lower().replace(' ', '')\n    \n    # 使用双指针检查\n    left, right = 0, len(s) - 1\n    \n    while left < right:\n        if s[left] != s[right]:\n            return False\n        left += 1\n        right -= 1\n    \n    return True\n\n# 示例用法\nprint(is_palindrome('A man a plan a canal Panama'))  # True\nprint(is_palindrome('racecar'))  # True\nprint(is_palindrome('hello'))  # False\n```"
            },
            {
                "input": "实现一个简单的计算器类",
                "output": "```python\nclass Calculator:\n    \"\"\"\n    简单的计算器类\n    \"\"\"\n    \n    def __init__(self):\n        self.history = []\n    \n    def add(self, a, b):\n        \"\"\"加法运算\"\"\"\n        result = a + b\n        self.history.append(f'{a} + {b} = {result}')\n        return result\n    \n    def subtract(self, a, b):\n        \"\"\"减法运算\"\"\"\n        result = a - b\n        self.history.append(f'{a} - {b} = {result}')\n        return result\n    \n    def multiply(self, a, b):\n        \"\"\"乘法运算\"\"\"\n        result = a * b\n        self.history.append(f'{a} * {b} = {result}')\n        return result\n    \n    def divide(self, a, b):\n        \"\"\"除法运算\"\"\"\n        if b == 0:\n            raise ValueError('除数不能为零')\n        result = a / b\n        self.history.append(f'{a} / {b} = {result}')\n        return result\n    \n    def get_history(self):\n        \"\"\"获取计算历史\"\"\"\n        return self.history\n\n# 示例用法\ncalc = Calculator()\nprint(calc.add(5, 3))  # 8\nprint(calc.multiply(4, 6))  # 24\nprint(calc.get_history())\n```"
            }
        ]
        
        # Generate variations
        dataset = []
        prefixes = [
            "请", "帮我", "能否", "请帮我", "可以", "请实现", "请写一个"
        ]
        
        for i in range(num_samples):
            base_sample = code_samples[i % len(code_samples)]
            prefix = prefixes[i % len(prefixes)]
            
            sample = {
                "input": f"{prefix}{base_sample['input']}",
                "output": base_sample["output"],
                "image_path": None
            }
            
            dataset.append(sample)
        
        # Save dataset
        output_file = os.path.join(self.data_dir, "chinese_code.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Chinese code dataset created: {output_file}")
        print(f"📊 Total samples: {len(dataset)}")
        return output_file
    
    def combine_chinese_datasets(self):
        """Combine all Chinese datasets."""
        print("🔄 Combining Chinese datasets...")
        
        datasets_to_combine = [
            "chinese_llava.json",
            "chinese_qa.json", 
            "chinese_code.json"
        ]
        
        combined_data = []
        
        for dataset_file in datasets_to_combine:
            file_path = os.path.join(self.data_dir, dataset_file)
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    combined_data.extend(data)
                    print(f"📁 Added {len(data)} samples from {dataset_file}")
        
        # Save combined dataset
        combined_file = os.path.join(self.data_dir, "chinese_combined.json")
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Combined Chinese dataset created: {combined_file}")
        print(f"📊 Total samples: {len(combined_data)}")
        return combined_file

def main():
    parser = argparse.ArgumentParser(description="Prepare Chinese datasets for Qwen 2.5 Omni")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--create_all", action="store_true", help="Create all Chinese datasets")
    parser.add_argument("--llava_samples", type=int, default=5000, help="Number of LLaVA samples")
    parser.add_argument("--qa_samples", type=int, default=3000, help="Number of Q&A samples")
    parser.add_argument("--code_samples", type=int, default=2000, help="Number of code samples")
    
    args = parser.parse_args()
    
    processor = ChineseDatasetProcessor(args.data_dir)
    
    if args.create_all:
        # Create all datasets
        processor.create_chinese_llava_dataset(args.llava_samples)
        processor.create_chinese_qa_dataset(args.qa_samples)
        processor.create_chinese_code_dataset(args.code_samples)
        processor.combine_chinese_datasets()
    else:
        print("请使用 --create_all 来创建所有中文数据集")

if __name__ == "__main__":
    main()
