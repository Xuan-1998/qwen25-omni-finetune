#!/bin/bash

# SLURM Job Script for Qwen 2.5 Omni Fine-tuning
# Use this script to submit training jobs to SLURM cluster

#SBATCH --job-name=qwen25_omni_ft
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

echo "🚀 Starting Qwen 2.5 Omni Fine-tuning Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"

# Load modules (adjust based on your cluster)
module purge
module load cuda/12.2
module load python/3.10

# Activate conda environment
source ~/.bashrc
conda activate qwen25_omni_ft

# Check GPU
echo "🔍 GPU Information:"
nvidia-smi

# Check disk space
echo "💾 Disk Usage:"
df -h .

# Create necessary directories
mkdir -p logs
mkdir -p models
mkdir -p data

# Set environment variables
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
export TOKENIZERS_PARALLELISM=false

# Create sample dataset if not exists
if [ ! -f "data/train_data.json" ]; then
    echo "📊 Creating sample dataset..."
    python3 scripts/create_sample_dataset.py
fi

# Run training
echo "🏋️ Starting training..."
python3 scripts/train_qwen_omni.py \
    --model_name_or_path "Qwen/Qwen2.5-Omni-7B" \
    --dataset_path "./data/train_data.json" \
    --output_dir "./models/qwen25_omni_sft" \
    --logging_dir "./logs" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --num_train_epochs 3 \
    --save_steps 500 \
    --logging_steps 10 \
    --warmup_steps 100 \
    --fp16 \
    --use_lora \
    --lora_r 8 \
    --lora_alpha 32 \
    --dataloader_num_workers 4 \
    --remove_unused_columns False \
    --report_to tensorboard \
    --run_name "qwen25_omni_sft_$(date +%Y%m%d_%H%M%S)"

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "✅ Training completed successfully!"
    
    # Run evaluation if test data exists
    if [ -f "data/test_data.json" ]; then
        echo "📊 Running evaluation..."
        python3 scripts/evaluate_model.py \
            --model_path "./models/qwen25_omni_sft" \
            --test_data "./data/test_data.json" \
            --output "./evaluation_results.json"
    fi
else
    echo "❌ Training failed!"
    exit 1
fi

echo "🏁 Job completed at: $(date)"
