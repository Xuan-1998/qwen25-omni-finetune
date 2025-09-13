# 🚀 Deployment Guide

## GitHub Repository Creation Steps

### 1. Create New Repository on GitHub

1. Visit [GitHub](https://github.com)
2. Click the "+" button in the top right corner, select "New repository"
3. Repository name: `qwen25-omni-finetune`
4. Description: `🚀 Qwen 2.5 Omni Fine-tuning Toolkit - Complete toolkit for fine-tuning Qwen 2.5 Omni multimodal models`
5. Set to Public
6. Do NOT initialize README, .gitignore, or LICENSE (we already have them)
7. Click "Create repository"

### 2. Push Code to GitHub

```bash
# Ensure you're in the project directory
cd /ocean/projects/cis250057p/hhe4/finetune

# Add remote repository
git remote add origin https://github.com/Xuan-1998/qwen25-omni-finetune.git

# Push to GitHub
git push -u origin main
```

### 3. Authentication Issues

If you encounter authentication issues during push, you can use one of the following methods:

#### Method 1: Using Personal Access Token

```bash
# Authenticate with token
git remote set-url origin https://YOUR_TOKEN@github.com/Xuan-1998/qwen25-omni-finetune.git
git push -u origin main
```

#### Method 2: Using SSH

```bash
# Switch to SSH URL
git remote set-url origin git@github.com:Xuan-1998/qwen25-omni-finetune.git
git push -u origin main
```

### 4. Verify Deployment

After successful deployment, you should be able to access:
- Repository URL: https://github.com/Xuan-1998/qwen25-omni-finetune
- README file should display correctly
- All script files should be visible

## 📁 Project Files Description

Uploaded files include:

### Core Files
- `README.md` - Complete project documentation and usage guide
- `DATASETS_GUIDE.md` - Dataset usage guide
- `DEPLOYMENT.md` - Deployment guide (this file)
- `LICENSE` - MIT License
- `.gitignore` - Git ignore file configuration

### Script Files
- `quick_start.sh` - Quick start script
- `setup_environment.sh` - Environment setup script
- `setup_align_anything.sh` - align-anything setup
- `setup_ms_swift.sh` - ms-swift setup

### Configuration Files
- `configs/qwen_omni_config.yaml` - Training configuration file

### Training Scripts
- `scripts/train_qwen_omni.py` - Main training script
- `scripts/simple_train.py` - Simplified training script
- `scripts/qwen_omni_sft.sh` - Training launch script
- `scripts/evaluate_model.py` - Model evaluation script
- `scripts/test_model.py` - Model testing script
- `scripts/monitor_training.py` - Training monitoring script

### Dataset Scripts
- `scripts/download_datasets.py` - Dataset download script
- `scripts/download_quick_datasets.sh` - Quick English dataset download
- `scripts/download_all_datasets.sh` - Download all datasets
- `scripts/prepare_chinese_datasets.py` - Chinese dataset creation
- `scripts/create_sample_dataset.py` - Sample dataset creation

### Cluster Support
- `scripts/run_with_slurm.sh` - SLURM cluster execution script

## 🎯 Usage Instructions

Users can use this project in the following ways:

1. **Clone Repository**
   ```bash
   git clone https://github.com/Xuan-1998/qwen25-omni-finetune.git
   cd qwen25-omni-finetune
   ```

2. **Quick Start**
   ```bash
   chmod +x quick_start.sh
   ./quick_start.sh
   ```

3. **Set Environment Variables**
   ```bash
   export HF_HOME=/path/to/your/hf_cache/
   export TRANSFORMERS_CACHE=/path/to/your/hf_cache/transformers
   export HF_DATASETS_CACHE=/path/to/your/hf_cache/datasets
   ```

4. **Start Training**
   ```bash
   python3 scripts/simple_train.py --model_name "Qwen/Qwen2.5-7B-Instruct"
   ```

## 📊 Project Highlights

- ✅ **Complete fine-tuning toolkit**
- ✅ **Multiple fine-tuning method support**
- ✅ **Rich public datasets**
- ✅ **Chinese dataset generation**
- ✅ **H100 GPU tested and working**
- ✅ **SLURM cluster support**
- ✅ **Detailed documentation and examples**

## 🔗 Related Links

- [Qwen 2.5 Omni Official Repository](https://github.com/QwenLM/Qwen2.5-Omni)
- [align-anything Framework](https://github.com/PKU-Alignment/align-anything)
- [ms-swift Framework](https://github.com/modelscope/ms-swift)
- [Hugging Face Qwen Models](https://huggingface.co/Qwen)

---

🎉 Congratulations! Your Qwen 2.5 Omni fine-tuning toolkit has been successfully deployed to GitHub!