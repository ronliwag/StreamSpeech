#!/bin/bash

# Training script for CVSS-T vocoder finetuning
# This script implements the multi-stage training strategy

# Configuration
CONFIG_PATH="pretrain_models/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.en/config_cvss_t.json"
PRETRAINED_MODEL="pretrain_models/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.en/g_00500000"
DATA_DIR="/path/to/cvss_t_data"  # Update this path
SAVE_DIR="checkpoints/cvss_t_vocoder"
DEVICE="cuda"
BATCH_SIZE=8

# Training parameters
NUM_EPOCHS_STAGE1=10
NUM_EPOCHS_STAGE2=20
NUM_EPOCHS_STAGE3=30

# Create save directory
mkdir -p $SAVE_DIR

echo "Starting CVSS-T vocoder training..."
echo "Config: $CONFIG_PATH"
echo "Pretrained model: $PRETRAINED_MODEL"
echo "Data directory: $DATA_DIR"
echo "Save directory: $SAVE_DIR"
echo "Device: $DEVICE"
echo "Batch size: $BATCH_SIZE"

# Run training
python train_cvss_t_vocoder.py \
    --config $CONFIG_PATH \
    --data-dir $DATA_DIR \
    --pretrained-model $PRETRAINED_MODEL \
    --save-dir $SAVE_DIR \
    --device $DEVICE \
    --batch-size $BATCH_SIZE \
    --num-epochs-stage1 $NUM_EPOCHS_STAGE1 \
    --num-epochs-stage2 $NUM_EPOCHS_STAGE2 \
    --num-epochs-stage3 $NUM_EPOCHS_STAGE3

echo "Training completed!"
echo "Checkpoints saved in: $SAVE_DIR"
