# 📚 Qwen 2.5 Omni Dataset Usage Guide

This guide explains how to use public datasets for fine-tuning Qwen 2.5 Omni models.

## 🎯 Supported Dataset Types

### 1. Multimodal Instruction Following Datasets
- **LLaVA-Instruct**: Most popular multimodal instruction dataset
- **Chinese LLaVA**: Chinese multimodal instruction dataset

### 2. Image Captioning Datasets
- **COCO Captions**: English image descriptions
- **COCO Chinese**: Chinese image descriptions
- **Flickr30K**: Image description data

### 3. Visual Question Answering Datasets
- **VQA v2**: Visual question answering benchmark dataset
- **TextVQA**: Text-based visual question answering
- **DocVQA**: Document-based visual question answering

### 4. Science and Math Datasets
- **ScienceQA**: Science question answering
- **MathVista**: Mathematical visual reasoning
- **AI2D**: Diagram understanding

### 5. Code Generation Datasets
- **Chinese Code**: Chinese code generation tasks
- **Algorithm Problems**: Algorithm problem solving

## 🚀 Quick Start

### Method 1: Using Quick Start Script
```bash
cd /ocean/projects/cis250057p/hhe4/finetune
chmod +x quick_start.sh
./quick_start.sh
```

Select the corresponding dataset download options:
- Option 5: Download English datasets
- Option 6: Create Chinese datasets  
- Option 7: Download all datasets (English + Chinese)

### Method 2: Using Command Line Scripts

#### Download English Datasets
```bash
chmod +x scripts/download_quick_datasets.sh
./scripts/download_quick_datasets.sh
```

#### Create Chinese Datasets
```bash
python3 scripts/prepare_chinese_datasets.py --create_all
```

#### Download All Datasets
```bash
chmod +x scripts/download_all_datasets.sh
./scripts/download_all_datasets.sh
```

### Method 3: Using Python Scripts

#### Download Specific Datasets
```bash
# Download LLaVA-Instruct dataset
python3 scripts/download_datasets.py --dataset llava_instruct --max_samples 5000

# Download COCO Captions dataset
python3 scripts/download_datasets.py --dataset coco_captions --max_samples 3000

# Download multiple datasets
python3 scripts/download_datasets.py --datasets llava_instruct coco_captions vqa_v2 --max_samples 2000
```

#### View Available Datasets
```bash
python3 scripts/download_datasets.py --list
```

## 📊 Dataset Details

### English Datasets

| Dataset | Type | Samples | Description |
|---------|------|---------|-------------|
| LLaVA-Instruct | Multimodal Instructions | 150K | Image understanding and instruction following |
| COCO Captions | Image Captioning | 118K | Image to text description |
| VQA v2 | Visual Question Answering | 443K | Image-based question answering |
| ScienceQA | Science QA | 21K | Science question answering |
| ChartQA | Chart QA | 9K | Chart understanding |

### Chinese Datasets

| Dataset | Type | Samples | Description |
|---------|------|---------|-------------|
| Chinese LLaVA | Multimodal Instructions | 5K | Chinese image understanding instructions |
| Chinese Q&A | Q&A | 3K | Chinese technical Q&A |
| Chinese Code | Code Generation | 2K | Chinese code generation tasks |

## 🔧 Dataset Format

All datasets are converted to a unified JSON format:

```json
[
  {
    "input": "User question or instruction",
    "output": "Expected model output",
    "image_path": "Optional image path",
    "audio_path": "Optional audio path",
    "video_path": "Optional video path"
  }
]
```

### Example Data

#### Multimodal Instructions
```json
{
  "input": "Please describe the content of this image.",
  "output": "This is a beautiful landscape image showing green mountains and clear lakes under blue sky and white clouds.",
  "image_path": "images/sample.jpg"
}
```

#### Code Generation
```json
{
  "input": "Write a Python function to calculate the Fibonacci sequence.",
  "output": "```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n```"
}
```

#### Technical Q&A
```json
{
  "input": "What is artificial intelligence?",
  "output": "Artificial Intelligence (AI) refers to technology that enables machines to simulate human intelligence, including machine learning, deep learning, natural language processing, and other fields."
}
```

## 📁 File Structure

After downloading, the dataset file structure is as follows:

```
data/
├── llava_instruct_train.json      # LLaVA-Instruct dataset
├── coco_captions_train.json       # COCO Captions dataset
├── vqa_v2_train.json              # VQA v2 dataset
├── scienceqa_train.json           # ScienceQA dataset
├── chartqa_train.json             # ChartQA dataset
├── chinese_llava.json             # Chinese LLaVA dataset
├── chinese_qa.json                # Chinese Q&A dataset
├── chinese_code.json              # Chinese code dataset
├── chinese_combined.json          # Chinese dataset combination
├── combined_train.json            # English dataset combination
├── final_combined_dataset.json    # Final combined dataset
├── final_combined_dataset_train.json  # Training set
├── final_combined_dataset_val.json    # Validation set
└── final_combined_dataset_test.json   # Test set
```

## 🎛️ Custom Configuration

### Adjust Dataset Size
```bash
# Limit maximum samples per dataset
python3 scripts/download_datasets.py --dataset llava_instruct --max_samples 1000

# Customize Chinese dataset size
python3 scripts/prepare_chinese_datasets.py --create_all --llava_samples 3000 --qa_samples 2000 --code_samples 1000
```

### Create Custom Dataset Splits
```bash
# Split existing dataset into train/val/test sets
python3 scripts/download_datasets.py --split_dataset ./data/my_dataset.json
```

## 🔍 Dataset Quality Check

### View Dataset Statistics
```bash
python3 -c "
import json
with open('./data/final_combined_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    print(f'Total samples: {len(data)}')
    print(f'Average input length: {sum(len(item[\"input\"]) for item in data) / len(data):.1f}')
    print(f'Average output length: {sum(len(item[\"output\"]) for item in data) / len(data):.1f}')
"
```

### View Dataset Samples
```bash
python3 -c "
import json
with open('./data/final_combined_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    for i, item in enumerate(data[:3]):
        print(f'Sample {i+1}:')
        print(f'Input: {item[\"input\"]}')
        print(f'Output: {item[\"output\"]}')
        print('-' * 50)
"
```

## 🚀 Start Training

After dataset preparation is complete, you can start fine-tuning:

```bash
# Train with combined dataset
python3 scripts/train_qwen_omni.py \
    --model_name_or_path "Qwen/Qwen2.5-Omni-7B" \
    --dataset_path "./data/final_combined_dataset.json" \
    --output_dir "./models/qwen25_omni_sft" \
    --use_lora \
    --num_train_epochs 3
```

## 🔧 Troubleshooting

### Common Issues

1. **Dataset Download Failure**
   - Check network connection
   - Verify Hugging Face access permissions
   - Try reducing `max_samples` parameter

2. **Out of Memory**
   - Reduce dataset size
   - Use smaller `max_samples` values
   - Process datasets in batches

3. **Dataset Format Errors**
   - Check JSON format is correct
   - Ensure required fields exist
   - Verify encoding format is UTF-8

### Get Help
```bash
# View dataset download help
python3 scripts/download_datasets.py --help

# View Chinese dataset creation help
python3 scripts/prepare_chinese_datasets.py --help
```

## 📈 Performance Recommendations

1. **Dataset Selection**:
   - For general fine-tuning, recommend using `final_combined_dataset.json`
   - For specific tasks, choose corresponding specialized datasets

2. **Training Strategy**:
   - Small datasets: Use more epochs
   - Large datasets: Use fewer epochs, more samples

3. **Hardware Optimization**:
   - Adjust batch size based on GPU memory
   - Use gradient accumulation to reduce memory usage

## 📚 Reference Resources

- [Hugging Face Datasets](https://huggingface.co/datasets)
- [LLaVA-Instruct Dataset](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K)
- [COCO Dataset](https://cocodataset.org/)
- [VQA Dataset](https://visualqa.org/)
- [Qwen 2.5-Omni Official Documentation](https://qwenlm.github.io/zh/blog/qwen2.5-omni/)