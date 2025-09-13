#!/usr/bin/env python3
"""
Download public datasets for Qwen 2.5 Omni fine-tuning
Includes various multi-modal datasets from Hugging Face and other sources
"""

import os
import json
import argparse
import requests
from datasets import load_dataset, Dataset
from huggingface_hub import hf_hub_download
import pandas as pd

class DatasetDownloader:
    def __init__(self, data_dir="./data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Available datasets
        self.available_datasets = {
            "llava_instruct": {
                "name": "LLaVA-Instruct",
                "description": "Multi-modal instruction following dataset",
                "hf_name": "liuhaotian/LLaVA-Instruct-150K",
                "type": "multimodal"
            },
            "coco_captions": {
                "name": "COCO Captions",
                "description": "Image captioning dataset",
                "hf_name": "HuggingFaceM4/COCO",
                "type": "image_captioning"
            },
            "vqa_v2": {
                "name": "VQA v2",
                "description": "Visual Question Answering dataset",
                "hf_name": "HuggingFaceM4/VQAv2",
                "type": "vqa"
            },
            "scienceqa": {
                "name": "ScienceQA",
                "description": "Science question answering with images",
                "hf_name": "derek-thomas/ScienceQA",
                "type": "science_qa"
            },
            "mathvista": {
                "name": "MathVista",
                "description": "Mathematical reasoning with visual elements",
                "hf_name": "AI4Math/MathVista",
                "type": "math_visual"
            },
            "chartqa": {
                "name": "ChartQA",
                "description": "Question answering on charts and graphs",
                "hf_name": "huggingface/ChartQA",
                "type": "chart_qa"
            },
            "docvqa": {
                "name": "DocVQA",
                "description": "Document-based visual question answering",
                "hf_name": "hf-internal-testing/example-docvqa",
                "type": "doc_qa"
            },
            "textvqa": {
                "name": "TextVQA",
                "description": "Text-based visual question answering",
                "hf_name": "HuggingFaceM4/TextVQA",
                "type": "text_vqa"
            },
            "ai2d": {
                "name": "AI2D",
                "description": "Diagram understanding dataset",
                "hf_name": "HuggingFaceM4/AI2D",
                "type": "diagram"
            },
            "nocaps": {
                "name": "NoCaps",
                "description": "Novel object captioning",
                "hf_name": "HuggingFaceM4/NoCaps",
                "type": "captioning"
            }
        }
    
    def list_available_datasets(self):
        """List all available datasets."""
        print("📚 Available Datasets:")
        print("=" * 60)
        for key, info in self.available_datasets.items():
            print(f"🔑 Key: {key}")
            print(f"   Name: {info['name']}")
            print(f"   Description: {info['description']}")
            print(f"   Type: {info['type']}")
            print(f"   HF Name: {info['hf_name']}")
            print("-" * 60)
    
    def download_dataset(self, dataset_key, split="train", max_samples=None):
        """Download a specific dataset."""
        if dataset_key not in self.available_datasets:
            print(f"❌ Dataset '{dataset_key}' not found!")
            self.list_available_datasets()
            return False
        
        dataset_info = self.available_datasets[dataset_key]
        print(f"📥 Downloading {dataset_info['name']}...")
        
        try:
            # Load dataset from Hugging Face
            dataset = load_dataset(dataset_info['hf_name'], split=split)
            
            # Limit samples if specified
            if max_samples and len(dataset) > max_samples:
                dataset = dataset.select(range(max_samples))
                print(f"📊 Limited to {max_samples} samples")
            
            # Convert to Qwen format
            converted_data = self.convert_to_qwen_format(dataset, dataset_info['type'])
            
            # Save dataset
            output_file = os.path.join(self.data_dir, f"{dataset_key}_{split}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(converted_data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Dataset saved to {output_file}")
            print(f"📊 Total samples: {len(converted_data)}")
            return True
            
        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
            return False
    
    def convert_to_qwen_format(self, dataset, dataset_type):
        """Convert dataset to Qwen 2.5 Omni format."""
        converted_data = []
        
        for i, item in enumerate(dataset):
            if dataset_type == "multimodal":
                # LLaVA-Instruct format
                converted_item = {
                    "input": item.get("instruction", item.get("question", "")),
                    "output": item.get("output", item.get("answer", "")),
                    "image_path": item.get("image", None)
                }
            
            elif dataset_type == "image_captioning":
                # COCO Captions format
                converted_item = {
                    "input": "请描述这张图片的内容。",
                    "output": item.get("caption", ""),
                    "image_path": item.get("image", None)
                }
            
            elif dataset_type == "vqa":
                # VQA format
                converted_item = {
                    "input": item.get("question", ""),
                    "output": item.get("answer", ""),
                    "image_path": item.get("image", None)
                }
            
            elif dataset_type == "science_qa":
                # ScienceQA format
                converted_item = {
                    "input": f"问题：{item.get('question', '')}",
                    "output": item.get("answer", ""),
                    "image_path": item.get("image", None)
                }
            
            elif dataset_type == "math_visual":
                # MathVista format
                converted_item = {
                    "input": f"请解决这个数学问题：{item.get('question', '')}",
                    "output": item.get("answer", ""),
                    "image_path": item.get("image", None)
                }
            
            elif dataset_type == "chart_qa":
                # ChartQA format
                converted_item = {
                    "input": f"根据图表回答问题：{item.get('question', '')}",
                    "output": item.get("answer", ""),
                    "image_path": item.get("image", None)
                }
            
            else:
                # Default format
                converted_item = {
                    "input": item.get("question", item.get("instruction", "")),
                    "output": item.get("answer", item.get("output", "")),
                    "image_path": item.get("image", None)
                }
            
            # Filter out empty inputs/outputs
            if converted_item["input"] and converted_item["output"]:
                converted_data.append(converted_item)
        
        return converted_data
    
    def download_multiple_datasets(self, dataset_keys, split="train", max_samples_per_dataset=1000):
        """Download multiple datasets."""
        print(f"📥 Downloading {len(dataset_keys)} datasets...")
        
        all_data = []
        for dataset_key in dataset_keys:
            if self.download_dataset(dataset_key, split, max_samples_per_dataset):
                # Load the downloaded data
                output_file = os.path.join(self.data_dir, f"{dataset_key}_{split}.json")
                with open(output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_data.extend(data)
        
        # Save combined dataset
        combined_file = os.path.join(self.data_dir, f"combined_{split}.json")
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Combined dataset saved to {combined_file}")
        print(f"📊 Total samples: {len(all_data)}")
    
    def create_splits(self, dataset_file, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        """Split dataset into train/validation/test sets."""
        print(f"📊 Creating dataset splits from {dataset_file}")
        
        with open(dataset_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Shuffle data
        import random
        random.shuffle(data)
        
        total_samples = len(data)
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)
        
        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]
        
        # Save splits
        base_name = os.path.splitext(os.path.basename(dataset_file))[0]
        
        train_file = os.path.join(self.data_dir, f"{base_name}_train.json")
        val_file = os.path.join(self.data_dir, f"{base_name}_val.json")
        test_file = os.path.join(self.data_dir, f"{base_name}_test.json")
        
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        
        with open(val_file, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)
        
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Dataset splits created:")
        print(f"   Train: {len(train_data)} samples -> {train_file}")
        print(f"   Val: {len(val_data)} samples -> {val_file}")
        print(f"   Test: {len(test_data)} samples -> {test_file}")

def main():
    parser = argparse.ArgumentParser(description="Download datasets for Qwen 2.5 Omni fine-tuning")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--dataset", type=str, help="Download specific dataset")
    parser.add_argument("--datasets", nargs="+", help="Download multiple datasets")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to download")
    parser.add_argument("--max_samples", type=int, help="Maximum number of samples to download")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--split_dataset", type=str, help="Split existing dataset into train/val/test")
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(args.data_dir)
    
    if args.list:
        downloader.list_available_datasets()
    
    elif args.dataset:
        downloader.download_dataset(args.dataset, args.split, args.max_samples)
    
    elif args.datasets:
        downloader.download_multiple_datasets(args.datasets, args.split, args.max_samples or 1000)
    
    elif args.split_dataset:
        downloader.create_splits(args.split_dataset)
    
    else:
        print("❌ Please specify an action. Use --help for options.")
        downloader.list_available_datasets()

if __name__ == "__main__":
    main()
