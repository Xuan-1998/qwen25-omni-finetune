#!/bin/bash

# Quick dataset download script for Qwen 2.5 Omni fine-tuning
# Downloads popular multi-modal datasets automatically

set -e

echo "🚀 Quick Dataset Download for Qwen 2.5 Omni"
echo "=========================================="

# Create data directory
mkdir -p ./data

echo "📥 Downloading popular multi-modal datasets..."

# Download LLaVA-Instruct (most popular for instruction tuning)
echo "1️⃣ Downloading LLaVA-Instruct dataset..."
python3 scripts/download_datasets.py --dataset llava_instruct --max_samples 5000

# Download COCO Captions for image captioning
echo "2️⃣ Downloading COCO Captions dataset..."
python3 scripts/download_datasets.py --dataset coco_captions --max_samples 3000

# Download VQA v2 for visual question answering
echo "3️⃣ Downloading VQA v2 dataset..."
python3 scripts/download_datasets.py --dataset vqa_v2 --max_samples 2000

# Download ScienceQA for science reasoning
echo "4️⃣ Downloading ScienceQA dataset..."
python3 scripts/download_datasets.py --dataset scienceqa --max_samples 1000

# Download ChartQA for chart understanding
echo "5️⃣ Downloading ChartQA dataset..."
python3 scripts/download_datasets.py --dataset chartqa --max_samples 500

# Combine all datasets
echo "🔄 Combining all datasets..."
python3 scripts/download_datasets.py --datasets llava_instruct coco_captions vqa_v2 scienceqa chartqa --max_samples 2000

# Create train/val/test splits
echo "📊 Creating dataset splits..."
python3 scripts/download_datasets.py --split_dataset ./data/combined_train.json

echo "✅ Dataset download completed!"
echo ""
echo "📁 Available datasets:"
echo "  - ./data/llava_instruct_train.json (LLaVA-Instruct)"
echo "  - ./data/coco_captions_train.json (COCO Captions)"
echo "  - ./data/vqa_v2_train.json (VQA v2)"
echo "  - ./data/scienceqa_train.json (ScienceQA)"
echo "  - ./data/chartqa_train.json (ChartQA)"
echo "  - ./data/combined_train.json (Combined dataset)"
echo ""
echo "📊 Train/Val/Test splits:"
echo "  - ./data/combined_train_train.json"
echo "  - ./data/combined_train_val.json"
echo "  - ./data/combined_train_test.json"
echo ""
echo "🎯 Ready for fine-tuning! Run:"
echo "  ./quick_start.sh"
