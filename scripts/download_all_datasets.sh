#!/bin/bash

# Complete dataset download script for Qwen 2.5 Omni
# Downloads both English and Chinese datasets

set -e

echo "🌍 Complete Dataset Download for Qwen 2.5 Omni"
echo "============================================="

# Create data directory
mkdir -p ./data

echo "📥 Phase 1: Downloading English datasets..."

# Download popular English multi-modal datasets
echo "1️⃣ LLaVA-Instruct dataset..."
python3 scripts/download_datasets.py --dataset llava_instruct --max_samples 3000

echo "2️⃣ COCO Captions dataset..."
python3 scripts/download_datasets.py --dataset coco_captions --max_samples 2000

echo "3️⃣ VQA v2 dataset..."
python3 scripts/download_datasets.py --dataset vqa_v2 --max_samples 1500

echo "4️⃣ ScienceQA dataset..."
python3 scripts/download_datasets.py --dataset scienceqa --max_samples 1000

echo "5️⃣ ChartQA dataset..."
python3 scripts/download_datasets.py --dataset chartqa --max_samples 500

echo "📥 Phase 2: Creating Chinese datasets..."

# Create Chinese datasets
echo "6️⃣ Chinese LLaVA dataset..."
python3 scripts/prepare_chinese_datasets.py --create_all --llava_samples 3000 --qa_samples 2000 --code_samples 1500

echo "🔄 Phase 3: Combining and organizing datasets..."

# Combine English datasets
echo "7️⃣ Combining English datasets..."
python3 scripts/download_datasets.py --datasets llava_instruct coco_captions vqa_v2 scienceqa chartqa --max_samples 1000

# Create final combined dataset (English + Chinese)
echo "8️⃣ Creating final combined dataset..."
python3 -c "
import json
import os

# Load English dataset
with open('./data/combined_train.json', 'r', encoding='utf-8') as f:
    english_data = json.load(f)

# Load Chinese dataset  
with open('./data/chinese_combined.json', 'r', encoding='utf-8') as f:
    chinese_data = json.load(f)

# Combine datasets
final_data = english_data + chinese_data

# Save final dataset
with open('./data/final_combined_dataset.json', 'w', encoding='utf-8') as f:
    json.dump(final_data, f, ensure_ascii=False, indent=2)

print(f'✅ Final dataset created with {len(final_data)} samples')
print(f'   English samples: {len(english_data)}')
print(f'   Chinese samples: {len(chinese_data)}')
"

# Create train/val/test splits
echo "9️⃣ Creating dataset splits..."
python3 scripts/download_datasets.py --split_dataset ./data/final_combined_dataset.json

echo "✅ All datasets downloaded and processed!"
echo ""
echo "📁 Available datasets:"
echo "======================"
echo ""
echo "🌍 English Datasets:"
echo "  - ./data/llava_instruct_train.json (LLaVA-Instruct)"
echo "  - ./data/coco_captions_train.json (COCO Captions)"
echo "  - ./data/vqa_v2_train.json (VQA v2)"
echo "  - ./data/scienceqa_train.json (ScienceQA)"
echo "  - ./data/chartqa_train.json (ChartQA)"
echo ""
echo "🇨🇳 Chinese Datasets:"
echo "  - ./data/chinese_llava.json (Chinese LLaVA)"
echo "  - ./data/chinese_qa.json (Chinese Q&A)"
echo "  - ./data/chinese_code.json (Chinese Code)"
echo "  - ./data/chinese_combined.json (Combined Chinese)"
echo ""
echo "🌐 Combined Datasets:"
echo "  - ./data/combined_train.json (English only)"
echo "  - ./data/final_combined_dataset.json (English + Chinese)"
echo ""
echo "📊 Train/Val/Test Splits:"
echo "  - ./data/final_combined_dataset_train.json"
echo "  - ./data/final_combined_dataset_val.json"
echo "  - ./data/final_combined_dataset_test.json"
echo ""
echo "🎯 Ready for fine-tuning! You can now run:"
echo "  ./quick_start.sh"
echo ""
echo "📈 Dataset Statistics:"
echo "======================"
python3 -c "
import json
import os

files = [
    './data/llava_instruct_train.json',
    './data/coco_captions_train.json', 
    './data/vqa_v2_train.json',
    './data/scienceqa_train.json',
    './data/chartqa_train.json',
    './data/chinese_llava.json',
    './data/chinese_qa.json',
    './data/chinese_code.json',
    './data/final_combined_dataset.json'
]

total_samples = 0
for file in files:
    if os.path.exists(file):
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            samples = len(data)
            total_samples += samples
            print(f'{os.path.basename(file):<35} {samples:>6} samples')

print(f'{"="*45}')
print(f'{"Total samples":<35} {total_samples:>6}')
"
