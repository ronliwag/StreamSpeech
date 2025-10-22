Here’s the updated version of your README with the checkpoint location added clearly and professionally:

---

# StreamSpeech CVSS-T Fine-tuning Setup (5k Dataset)

## Overview

This setup prepares **StreamSpeech** for fine-tuning on the **CVSS-T subset** (Spanish-English pairs) using the full **5k samples** from both languages.

## Dataset

* **Source**: CVSS-T subset from CommonVoice v4 batches
* **Spanish**: 5,000 samples from all 10 batches in “1st Batch - 5000”
* **English**: 5,000 samples from all 10 batches in “1st Batch - 5000”
* **Total Pairs**: 1,247 matched Spanish-English pairs
* **Split**: 997 train, 124 dev, 126 test

## Files Created

```
cvss-t-finetune/
├── es-en/
│   ├── fbank2unit/
│   │   ├── config_gcmvn.yaml
│   │   ├── config_mtl_asr_st_ctcst.yaml
│   │   ├── train.tsv, dev.tsv, test.tsv
│   │   ├── train.src, dev.src, test.src
│   │   └── train.txt, dev.txt, test.txt
│   ├── src_unigram6000/ (Spanish vocabulary)
│   ├── tgt_unigram6000/ (English vocabulary)
│   └── gcmvn.npz (Global CMVN statistics)
├── prepare_cvss_t_subset.py
├── preprocess_cvss_t.py
├── setup_finetune.sh
└── train_finetune.sh
```

## Training Configuration

* **Base Model**: `pretrain_models/streamspeech.offline.es-en.pt`
* **Architecture**: StreamSpeech (encoder → translator → decoder)
* **Learning Rate**: 0.0001 (reduced for fine-tuning)
* **Batch Size**: Adjusted for 5k dataset (`max-tokens: 12000`)
* **Validation**: Every 200 updates (optimized for larger dataset)
* **Checkpoints**:

  * Local: `checkpoints/streamspeech.finetuned.es-en/`
  * Online Backup: [Google Drive Folder](https://drive.google.com/drive/folders/1C24MO57BBzVo5HYd6ntTIqMiTIwpglgL?usp=sharing)

## Next Steps

1. Run preprocessing:

   ```bash
   python cvss-t-finetune/preprocess_cvss_t.py
   ```
2. Start fine-tuning:

   ```bash
   bash cvss-t-finetune/train_finetune.sh
   ```
3. Monitor training progress
4. Evaluate on test set

## Notes

* Fine-tuning preserves web interface compatibility
* Model architecture remains unchanged
* Adapts only to CVSS-T domain characteristics
* Ready for **Thursday/Friday defense timeline**
* **5k Dataset**: Enables broader and more diverse training samples
* **Improved Performance**: 1,247 pairs vs. 455 pairs (≈2.7× more data)
* Checkpoints are safely stored both locally and on Google Drive for redundancy

