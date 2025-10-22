#!/bin/bash
# CVSS-T Fine-tuning Setup Script

echo "Setting up CVSS-T fine-tuning for StreamSpeech..."

# Set paths
ROOT=$(pwd)
DATA_ROOT=$ROOT/cvss-t-finetune
DATA=$DATA_ROOT/es-en/fbank2unit

echo "Data directory: $DATA"

# Create checkpoint directory
mkdir -p checkpoints/streamspeech.finetuned.es-en

echo "Created checkpoint directory"

# Verify data files exist
if [ -f "$DATA/train.tsv" ] && [ -f "$DATA/dev.tsv" ] && [ -f "$DATA/test.tsv" ]; then
    echo "Data files verified successfully"
    echo "Train samples: $(wc -l < $DATA/train.tsv)"
    echo "Dev samples: $(wc -l < $DATA/dev.tsv)"
    echo "Test samples: $(wc -l < $DATA/test.tsv)"
else
    echo "Error: Required data files not found"
    exit 1
fi

echo "Setup complete! Ready for fine-tuning."
echo "Run: bash cvss-t-finetune/train_finetune.sh"

