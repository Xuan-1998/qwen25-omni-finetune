#!/bin/bash

# Setup ms-swift for Qwen 2.5 Omni fine-tuning
# Based on: https://github.com/modelscope/ms-swift/pull/3613

set -e

echo "🎯 Setting up ms-swift for Qwen 2.5 Omni..."

cd ms_swift

# Clone ms-swift repository
if [ ! -d "ms-swift" ]; then
    echo "📥 Cloning ms-swift repository..."
    git clone https://github.com/modelscope/ms-swift.git
fi

cd ms-swift

# Install ms-swift
echo "🔧 Installing ms-swift..."
pip install -e .

# Install additional dependencies
echo "📚 Installing additional dependencies..."
pip install transformers datasets accelerate peft
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "✅ ms-swift setup completed!"
echo "📁 Location: $(pwd)"
echo "🚀 Ready to run fine-tuning with ms-swift framework"
