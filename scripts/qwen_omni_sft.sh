#!/bin/bash

# Qwen 2.5 Omni SFT (Supervised Fine-Tuning) Script
# Based on align-anything implementation

set -e

echo "🚀 Starting Qwen 2.5 Omni SFT training..."

# Configuration
MODEL_NAME="Qwen/Qwen2.5-Omni-7B"  # Change to 72B for larger model
DATASET_PATH="./data/train_data.json"
OUTPUT_DIR="./models/qwen25_omni_sft"
LOGS_DIR="./logs"

# Create directories
mkdir -p $OUTPUT_DIR
mkdir -p $LOGS_DIR
mkdir -p ./data

# Check if dataset exists
if [ ! -f "$DATASET_PATH" ]; then
    echo "⚠️  Dataset not found at $DATASET_PATH"
    echo "📝 Creating sample dataset..."
    python scripts/create_sample_dataset.py
fi

# Training parameters
BATCH_SIZE=1
GRADIENT_ACCUMULATION=8
LEARNING_RATE=1e-5
NUM_EPOCHS=3
SAVE_STEPS=500
LOGGING_STEPS=10
WARMUP_STEPS=100

echo "📊 Training Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Dataset: $DATASET_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  Batch Size: $BATCH_SIZE"
echo "  Gradient Accumulation: $GRADIENT_ACCUMULATION"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Epochs: $NUM_EPOCHS"

# Run training
python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port=29500 \
    scripts/train_qwen_omni.py \
    --model_name_or_path $MODEL_NAME \
    --dataset_path $DATASET_PATH \
    --output_dir $OUTPUT_DIR \
    --logging_dir $LOGS_DIR \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $NUM_EPOCHS \
    --save_steps $SAVE_STEPS \
    --logging_steps $LOGGING_STEPS \
    --warmup_steps $WARMUP_STEPS \
    --fp16 \
    --dataloader_num_workers 4 \
    --remove_unused_columns False \
    --report_to tensorboard \
    --run_name "qwen25_omni_sft_$(date +%Y%m%d_%H%M%S)"

echo "✅ Training completed!"
echo "📁 Model saved to: $OUTPUT_DIR"
echo "📊 Logs available in: $LOGS_DIR"
