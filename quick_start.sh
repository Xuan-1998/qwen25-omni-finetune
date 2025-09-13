#!/bin/bash

# Quick Start Script for Qwen 2.5 Omni Fine-tuning
# This script provides a streamlined way to start fine-tuning

set -e

echo "🚀 Qwen 2.5 Omni Fine-tuning Quick Start"
echo "========================================"

# Check if we're in the right directory
if [ ! -f "setup_environment.sh" ]; then
    echo "❌ Please run this script from the finetune directory"
    exit 1
fi

# Function to display menu
show_menu() {
    echo ""
    echo "📋 Choose your fine-tuning method:"
    echo "1. Setup Environment (First time only)"
    echo "2. Use align-anything method"
    echo "3. Use ms-swift method"
    echo "4. Create sample dataset"
    echo "5. Download public datasets (English)"
    echo "6. Download Chinese datasets"
    echo "7. Download all datasets (English + Chinese)"
    echo "8. Start training (align-anything)"
    echo "9. Evaluate model"
    echo "10. Show system info"
    echo "11. Exit"
    echo ""
}

# Function to check GPU
check_gpu() {
    echo "🔍 Checking GPU availability..."
    if command -v nvidia-smi &> /dev/null; then
        echo "✅ NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
    else
        echo "❌ No NVIDIA GPU detected"
        return 1
    fi
}

# Function to show system info
show_system_info() {
    echo "💻 System Information:"
    echo "======================"
    
    # OS Info
    echo "OS: $(uname -s) $(uname -r)"
    
    # Python Info
    if command -v python3 &> /dev/null; then
        echo "Python: $(python3 --version)"
    fi
    
    # CUDA Info
    if command -v nvidia-smi &> /dev/null; then
        echo "CUDA Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)"
    fi
    
    # Memory Info
    echo "System Memory: $(free -h | grep '^Mem:' | awk '{print $2}')"
    
    # Disk Space
    echo "Available Disk: $(df -h . | tail -1 | awk '{print $4}')"
    
    # Check GPU
    check_gpu
}

# Function to create sample dataset
create_dataset() {
    echo "📊 Creating sample dataset..."
    if [ -f "scripts/create_sample_dataset.py" ]; then
        python3 scripts/create_sample_dataset.py
        echo "✅ Sample dataset created successfully"
    else
        echo "❌ Dataset creation script not found"
    fi
}

# Function to start training
start_training() {
    echo "🏋️ Starting training..."
    
    # Check if dataset exists
    if [ ! -f "data/train_data.json" ]; then
        echo "⚠️  Training dataset not found. Creating sample dataset..."
        create_dataset
    fi
    
    # Check if align-anything is set up
    if [ ! -d "align_anything/align-anything" ]; then
        echo "⚠️  align-anything not set up. Setting up now..."
        chmod +x setup_align_anything.sh
        ./setup_align_anything.sh
    fi
    
    # Start training
    chmod +x scripts/qwen_omni_sft.sh
    ./scripts/qwen_omni_sft.sh
}

# Function to evaluate model
evaluate_model() {
    echo "📊 Evaluating model..."
    
    # Check if model exists
    if [ ! -d "models/qwen25_omni_sft" ]; then
        echo "❌ No trained model found at models/qwen25_omni_sft"
        echo "Please train a model first or specify the correct path"
        return 1
    fi
    
    # Check if test data exists
    if [ ! -f "data/test_data.json" ]; then
        echo "⚠️  Test dataset not found. Creating sample dataset..."
        create_dataset
    fi
    
    # Run evaluation
    python3 scripts/evaluate_model.py \
        --model_path ./models/qwen25_omni_sft \
        --test_data ./data/test_data.json \
        --output ./evaluation_results.json
}

# Main menu loop
while true; do
    show_menu
    read -p "Enter your choice (1-11): " choice
    
    case $choice in
        1)
            echo "🔧 Setting up environment..."
            chmod +x setup_environment.sh
            ./setup_environment.sh
            ;;
        2)
            echo "🎯 Setting up align-anything..."
            chmod +x setup_align_anything.sh
            ./setup_align_anything.sh
            ;;
        3)
            echo "🎯 Setting up ms-swift..."
            chmod +x setup_ms_swift.sh
            ./setup_ms_swift.sh
            ;;
        4)
            create_dataset
            ;;
        5)
            echo "📥 Downloading English datasets..."
            chmod +x scripts/download_quick_datasets.sh
            ./scripts/download_quick_datasets.sh
            ;;
        6)
            echo "📥 Creating Chinese datasets..."
            python3 scripts/prepare_chinese_datasets.py --create_all
            ;;
        7)
            echo "📥 Downloading all datasets (English + Chinese)..."
            chmod +x scripts/download_all_datasets.sh
            ./scripts/download_all_datasets.sh
            ;;
        8)
            start_training
            ;;
        9)
            evaluate_model
            ;;
        10)
            show_system_info
            ;;
        11)
            echo "👋 Goodbye!"
            exit 0
            ;;
        *)
            echo "❌ Invalid choice. Please enter 1-11."
            ;;
    esac
    
    echo ""
    read -p "Press Enter to continue..."
done
