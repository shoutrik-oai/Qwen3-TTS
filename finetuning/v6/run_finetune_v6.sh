#!/bin/bash
# Finetuning script for Qwen3-TTS with Entity Injection
# Usage: ./run_finetune.sh [--config accelerate_config.yaml]

set -e

# ============================================================================
# Configuration
# ============================================================================

# Model paths
INIT_MODEL_PATH="/speech/arjun/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-Base/snapshots/5d83992436eae1d760afd27aff78a71d676296fc"
OUTPUT_MODEL_PATH="/speech/arjun/shoutrik/Qwen3-TTS/finetuning/v6/experiments/entity_injection_v6.2"

# Data
TRAIN_JSONL="/speech/arjun/shoutrik/DATA/jsonl_files/metadata_with_entities.bkp.jsonl"

# Training hyperparameters
BATCH_SIZE=16
LEARNING_RATE=1e-6
NUM_EPOCHS=20
ENTITY_LOSS_WEIGHT=0.1
GRADIENT_ACCUMULATION_STEPS=4

# Speaker
SPEAKER_NAME="speaker_test"

# Wandb
WANDB_PROJECT="qwen3-tts-entity-injection"
WANDB_RUN_NAME="qwen3-0.6b-tts-entity-injection-exp-$(date +%Y%m%d-%H%M%S)"
# WANDB_ENTITY=""  # Uncomment and set if needed

# Load .env file if it exists (for WANDB_API_KEY, etc.)
ENV_FILE="$(dirname "${BASH_SOURCE[0]}")/.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
fi

# Accelerate config (optional - will use default if not provided)
ACCELERATE_CONFIG="/speech/arjun/shoutrik/Qwen3-TTS/finetuning/v6/default_config.yaml"

# GPU settings
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Change to "0,1" for multi-GPU

# ============================================================================
# Parse command line arguments
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            ACCELERATE_CONFIG="$2"
            shift 2
            ;;
        --train_jsonl)
            TRAIN_JSONL="$2"
            shift 2
            ;;
        --output)
            OUTPUT_MODEL_PATH="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --wandb_project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --wandb_run)
            WANDB_RUN_NAME="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============================================================================
# Setup
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Create output directory
mkdir -p "$OUTPUT_MODEL_PATH"

# ============================================================================
# Print configuration
# ============================================================================

echo "=============================================="
echo "Qwen3-TTS Entity Injection Finetuning"
echo "=============================================="
echo ""
echo "Model Configuration:"
echo "  Init model:     $INIT_MODEL_PATH"
echo "  Output path:    $OUTPUT_MODEL_PATH"
echo ""
echo "Data:"
echo "  Train JSONL:    $TRAIN_JSONL"
echo ""
echo "Training:"
echo "  Batch size:     $BATCH_SIZE"
echo "  Learning rate:  $LEARNING_RATE"
echo "  Epochs:         $NUM_EPOCHS"
echo "  Entity loss:    $ENTITY_LOSS_WEIGHT"
echo ""
echo "Wandb:"
echo "  Project:        $WANDB_PROJECT"
echo "  Run name:       $WANDB_RUN_NAME"
echo ""
echo "GPU:"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "=============================================="
echo ""

# ============================================================================
# Check requirements
# ============================================================================

if [ ! -f "$TRAIN_JSONL" ]; then
    echo "ERROR: Training data not found: $TRAIN_JSONL"
    exit 1
fi

# ============================================================================
# Run training
# ============================================================================

if [ -n "$ACCELERATE_CONFIG" ]; then
    echo "Using accelerate config: $ACCELERATE_CONFIG"
    ACCELERATE_CMD="accelerate launch --config_file $ACCELERATE_CONFIG"
else
    echo "Using default accelerate configuration (single GPU)"
    ACCELERATE_CMD="accelerate launch --mixed_precision bf16"
fi
export TORCH_DISTRIBUTED_DEBUG=DETAIL

$ACCELERATE_CMD "sft_12hz_with_EntityInjection_v6.py" \
    --init_model_path "$INIT_MODEL_PATH" \
    --output_model_path "$OUTPUT_MODEL_PATH" \
    --train_jsonl "$TRAIN_JSONL" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LEARNING_RATE" \
    --num_epochs "$NUM_EPOCHS" \
    --speaker_name "$SPEAKER_NAME" \
    --entity_loss_weight "$ENTITY_LOSS_WEIGHT" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "$WANDB_RUN_NAME"

echo ""
echo "=============================================="
echo "Training complete!"
echo "Checkpoints saved to: $OUTPUT_MODEL_PATH"
echo "=============================================="