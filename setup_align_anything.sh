#!/bin/bash

# Setup align-anything for Qwen 2.5 Omni fine-tuning
# Based on: https://github.com/QwenLM/Qwen2.5-Omni/issues/12

set -e

echo "🎯 Setting up align-anything for Qwen 2.5 Omni..."

cd align_anything

# Clone align-anything repository
if [ ! -d "align-anything" ]; then
    echo "📥 Cloning align-anything repository..."
    git clone https://github.com/PKU-Alignment/align-anything.git
fi

cd align-anything

# Install align-anything
echo "🔧 Installing align-anything..."
pip install -e .[train]

# Install specific dependencies for Qwen 2.5 Omni
echo "🔄 Installing Qwen 2.5 Omni specific dependencies..."
pip uninstall transformers -y
pip install git+https://github.com/huggingface/transformers@3a1ead0aabed473eafe527915eea8c197d424356
pip install -U flash-attn --no-build-isolation

echo "✅ align-anything setup completed!"
echo "📁 Location: $(pwd)"
echo "🚀 Ready to run fine-tuning with: bash scripts/qwen_omni_sft.sh"
