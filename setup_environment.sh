#!/bin/bash

# Qwen 2.5 Omni Fine-tuning Environment Setup
# Based on the information from GitHub issues and discussions

set -e

echo "🚀 Setting up Qwen 2.5 Omni Fine-tuning Environment..."

# Check CUDA availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA GPU not detected. Please ensure CUDA is installed."
    exit 1
fi

echo "✅ CUDA detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits

# Create conda environment if it doesn't exist
ENV_NAME="qwen25_omni_ft"
if ! conda env list | grep -q $ENV_NAME; then
    echo "📦 Creating conda environment: $ENV_NAME"
    conda create -n $ENV_NAME python=3.10 -y
fi

echo "🔄 Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# Install CUDA toolkit
echo "🔧 Installing CUDA toolkit..."
conda install nvidia/label/cuda-12.2.0::cuda -y
export CUDA_HOME=$CONDA_PREFIX

# Install PyTorch with CUDA support
echo "🔥 Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
echo "📚 Installing core dependencies..."
pip install transformers accelerate datasets peft bitsandbytes

# Install flash-attention for better performance
echo "⚡ Installing flash-attention..."
pip install -U flash-attn --no-build-isolation

# Install additional dependencies for multi-modal training
echo "🎨 Installing multi-modal dependencies..."
pip install opencv-python pillow librosa soundfile moviepy

# Install training frameworks
echo "🏗️ Installing training frameworks..."
pip install deepspeed wandb tensorboard

# Install specific transformers version for Qwen 2.5 Omni
echo "🔄 Installing specific transformers version..."
pip uninstall transformers -y
pip install git+https://github.com/huggingface/transformers@3a1ead0aabed473eafe527915eea8c197d424356

echo "✅ Environment setup completed!"
echo ""
echo "📋 Next steps:"
echo "1. Run: conda activate $ENV_NAME"
echo "2. Choose your preferred fine-tuning method:"
echo "   - align-anything: ./setup_align_anything.sh"
echo "   - ms-swift: ./setup_ms_swift.sh"
echo ""
echo "🎯 Available models:"
echo "   - Qwen/Qwen2.5-Omni-7B (recommended for most users)"
echo "   - Qwen/Qwen2.5-Omni-72B (requires high-end hardware)"
