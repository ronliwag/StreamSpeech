# CVSS-T Fine-tuning for StreamSpeech

This directory contains scripts and configurations for fine-tuning the StreamSpeech model on CVSS-T dataset for domain adaptation from synthetic (CVSS-C) to real human speech (CVSS-T).

## Overview

Fine-tuning adapts the pre-trained StreamSpeech model (trained on CVSS-C synthetic speech) to work with real human speech from the CVSS-T dataset. This improves translation quality and naturalness for Spanish-to-English speech-to-speech translation.

## Dataset

- **Source**: CommonVoice v4 Spanish (es) and English (en)
- **1st Batch**: Clips 1-10 (1,247 paired samples)
- **2nd Batch**: Clips 11-20 (1,777 paired samples)
- **3rd Batch**: Clips 1-10 (1,644 paired samples)
- **Total**: 4,668 Spanish-English audio pairs

## Training Results

### 1st Batch Training (Clips 1-10)
- Samples: 1,247
- Updates: 1,000
- Final Loss: 37.281
- Training Time: ~28 minutes
- Checkpoint: `checkpoint_17_1000.pt`

### Extended Training (Clips 1-20)
- Samples: 3,024 (combined)
- Updates: 2,000
- Final Loss: 34.862
- Best Validation Loss: 33.027
- Training Time: ~26 minutes
- Best Checkpoint: `checkpoint_best.pt`

### 3rd Batch Training (Final Model - October 2025)
- Samples: 4,668 (1,247 + 1,777 + 1,644)
- Updates: 5,000 (continued from 1,600)
- Epochs: 21
- Final Training Loss: 24.564
- Best Validation Loss: 27.353
- Training Time: 1.5 hours
- Best Checkpoint: `checkpoint_best.pt` (822 MB)

### Performance Improvement
- Loss Improvement: 6.5% (37.281 → 34.862) after 2nd batch
- **Final Improvement: 26.6%** (37.281 → 27.353) after 3rd batch
- Dataset Size: 3.7x increase (1,247 → 4,668 samples)

## Fine-tuned Models

Models are available on Google Drive (see links in main README):

1. **1st Batch Model** (1,247 samples)
   - Location: `checkpoints/streamspeech.finetuned.es-en/`
   - Best for: Initial CVSS-T adaptation

2. **Extended Model** (3,024 samples)
   - Location: `checkpoints/streamspeech.finetuned.es-en.extended/`
   - Best for: Extended training baseline

3. **3rd Batch Model** (4,668 samples) - RECOMMENDED
   - Location: Google Drive
   - File: `streamspeech.3rd-batch.es-en.checkpoint_best.pt` (822 MB)
   - Best for: Production use and evaluation
   - Best validation loss: 27.353

## Files

### Scripts
- `prepare_cvss_t_subset.py` - Prepares 1st batch manifest files
- `prepare_cvss_t_2nd_batch.py` - Prepares 2nd batch manifest files
- `preprocess_cvss_t.py` - Preprocesses audio data
- `setup_finetune.sh` - Setup automation script
- `train_finetune.sh` - Training script (bash)
- `train_finetune.ps1` - Training script (PowerShell)

### Configuration
- `es-en/fbank2unit/config_gcmvn.yaml` - Audio processing config
- `es-en/fbank2unit/config_mtl_asr_st_ctcst.yaml` - Multitask learning config
- `es-en/src_unigram6000/` - Source language tokenizer

## Usage

### 1. Setup Environment
```bash
bash setup_finetune.sh
```

### 2. Prepare Data
```bash
# For 1st batch
python prepare_cvss_t_subset.py

# For 2nd batch
python prepare_cvss_t_2nd_batch.py
```

### 3. Start Training
```bash
# Bash
bash train_finetune.sh

# PowerShell (Windows)
.\train_finetune.ps1
```

## Requirements

- Python 3.8+
- PyTorch 1.10+
- fairseq (included in parent directory)
- CUDA-capable GPU (4GB+ VRAM recommended)
- Audio files in WAV format

## Key Modifications

### Fairseq Fixes Applied
1. **Windows Path Handling**: Fixed drive letter parsing in `audio_utils.py`
2. **UTF-8 Encoding**: Added UTF-8 support for TSV reading in `speech_to_text_dataset.py`
3. **Cython Fallback**: Added Python fallback for batch processing in `data_utils.py`

## Training Configuration

- Learning Rate: 0.0001
- Warmup Updates: 2,000
- Max Tokens: 12,000
- Batch Size: 16
- Updates per Epoch: ~60
- Validation Interval: 200 updates
- Save Interval: 100 updates

## Directory Structure

```
cvss-t-finetune/
├── es-en/
│   ├── fbank2unit/
│   │   ├── config_gcmvn.yaml
│   │   ├── config_mtl_asr_st_ctcst.yaml
│   │   ├── train.tsv
│   │   ├── dev.tsv
│   │   └── test.tsv
│   ├── src_unigram6000/
│   └── tgt_unigram6000/
├── prepare_cvss_t_subset.py
├── prepare_cvss_t_2nd_batch.py
├── preprocess_cvss_t.py
├── setup_finetune.sh
├── train_finetune.sh
├── train_finetune.ps1
└── README.md
```

## Citation

If you use this fine-tuning setup, please cite the original StreamSpeech paper:

```bibtex
@inproceedings{streamspeech,
  title={StreamSpeech: Simultaneous Speech-to-Speech Translation with Multi-task Learning},
  author={Shaolei Zhang and Qingkai Fang and Shoutao Guo and Zhengrui Ma and Min Zhang and Yang Feng},
  year={2024},
  booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics}
}
```

## License

MIT License (same as parent StreamSpeech project)

## Contact

For questions about this fine-tuning setup, please refer to the main repository or create an issue.
