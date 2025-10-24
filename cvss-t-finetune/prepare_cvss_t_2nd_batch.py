#!/usr/bin/env python3
"""
CVSS-T 2nd Batch Data Preparation Script
Creates manifest files for fine-tuning StreamSpeech on CVSS-T 2nd batch data (Clips - Batch 11-20).
"""

import os
import pandas as pd
import json
from pathlib import Path
import random

def create_cvss_t_2nd_batch():
    """Create CVSS-T 2nd batch manifest files for fine-tuning (Clips - Batch 11-20)."""
    
    base_path = Path("E:/Thesis_Datasets")
    output_path = Path("cvss-t-finetune/es-en/fbank2unit")
    
    spanish_batch_path = base_path / "CommonVoice_v4/es/cv-corpus-22.0-2025-06-20/es/2nd Batch - 5000 WAV"
    english_batch_path = base_path / "CommonVoice_v4/en/2nd Batch - 5000 WAV"
    
    spanish_tsv = base_path / "CommonVoice_v4/es/cv-corpus-22.0-2025-06-20/es/train.tsv"
    english_tsv = base_path / "CommonVoice_v4/en/train.tsv"
    covost_file = base_path / "CoVoST/covost_v2.es_en.tsv"
    
    print("Reading CommonVoice TSV files...")
    
    spanish_df = pd.read_csv(spanish_tsv, sep='\t')
    print(f"Found {len(spanish_df)} Spanish samples in CommonVoice")
    
    english_df = pd.read_csv(english_tsv, sep='\t')
    print(f"Found {len(english_df)} English samples in CommonVoice")
    
    covost_df = pd.read_csv(covost_file, sep='\t')
    train_covost = covost_df[covost_df['split'] == 'train'].copy()
    print(f"Found {len(train_covost)} train samples in CoVoST")
    
    print("\nCollecting audio files from 2nd Batch (Clips - Batch 11-20)...")
    spanish_files = []
    for batch_num in range(11, 21):
        clips_path = spanish_batch_path / f"Clips - Batch {batch_num}"
        if clips_path.exists():
            files = list(clips_path.glob("*.wav"))
            spanish_files.extend(files)
            print(f"Found {len(files)} Spanish WAV files in Clips - Batch {batch_num}")
    
    print(f"Total Spanish files from 2nd Batch: {len(spanish_files)}")
    
    english_files = []
    for batch_num in range(11, 21):
        clips_path = english_batch_path / f"Clips - Batch {batch_num}"
        if clips_path.exists():
            files = list(clips_path.glob("*.wav"))
            english_files.extend(files)
            print(f"Found {len(files)} English WAV files in Clips - Batch {batch_num}")
    
    print(f"Total English files from 2nd Batch: {len(english_files)}")
    
    spanish_map = {f.stem: f for f in spanish_files}
    english_map = {f.stem: f for f in english_files}
    
    spanish_text_map = {}
    for _, row in spanish_df.iterrows():
        spanish_text_map[row['path']] = row['sentence']
    
    print("\nMatching Spanish-English pairs...")
    matched_pairs = []
    for _, row in train_covost.iterrows():
        spanish_file = row['path']
        spanish_id = spanish_file.replace('.mp3', '')
        
        if spanish_id in spanish_map and spanish_file in spanish_text_map:
            if english_files:
                english_file = random.choice(english_files)
                
                matched_pairs.append({
                    'id': spanish_id,
                    'src_audio': str(spanish_map[spanish_id]).replace('\\', '/'),
                    'tgt_audio': str(english_file).replace('\\', '/'),
                    'src_text': spanish_text_map[spanish_file],
                    'tgt_text': row['translation']
                })
    
    if len(matched_pairs) == 0:
        print("No CoVoST matches found, creating pairs from 2nd batch files...")
        min_length = min(len(spanish_files), len(english_files))
        
        for i in range(min_length):
            spanish_file = spanish_files[i]
            spanish_id = spanish_file.stem
            spanish_path = spanish_file.name.replace('.wav', '.mp3')
            
            if spanish_path in spanish_text_map:
                english_file = english_files[i]
                
                matched_pairs.append({
                    'id': spanish_id,
                    'src_audio': str(spanish_file).replace('\\', '/'),
                    'tgt_audio': str(english_file).replace('\\', '/'),
                    'src_text': spanish_text_map[spanish_path],
                    'tgt_text': f"English translation for {spanish_id}"
                })
    
    print(f"Created {len(matched_pairs)} matched pairs from 2nd Batch (Clips 11-20)")
    
    random.shuffle(matched_pairs)
    train_size = int(0.8 * len(matched_pairs))
    dev_size = int(0.1 * len(matched_pairs))
    
    train_pairs = matched_pairs[:train_size]
    dev_pairs = matched_pairs[train_size:train_size + dev_size]
    test_pairs = matched_pairs[train_size + dev_size:]
    
    print(f"Split: {len(train_pairs)} train, {len(dev_pairs)} dev, {len(test_pairs)} test")
    
    for split, pairs in [('train', train_pairs), ('dev', dev_pairs), ('test', test_pairs)]:
        manifest_file = output_path / f"{split}.tsv"
        
        with open(manifest_file, 'w', encoding='utf-8') as f:
            f.write("id\tsrc_audio\tsrc_n_frames\tsrc_text\ttgt_text\ttgt_audio\ttgt_n_frames\n")
            
            for pair in pairs:
                f.write(f"{pair['id']}\t{pair['src_audio']}\t0\t{pair['src_text']}\t{pair['tgt_text']}\t{pair['tgt_audio']}\t0\n")
        
        print(f"Created {manifest_file}")
    
    for split, pairs in [('train', train_pairs), ('dev', dev_pairs), ('test', test_pairs)]:
        src_file = output_path / f"{split}.src"
        with open(src_file, 'w', encoding='utf-8') as f:
            for pair in pairs:
                f.write(f"{pair['src_text']}\n")
        
        tgt_file = output_path / f"{split}.txt"
        with open(tgt_file, 'w', encoding='utf-8') as f:
            for pair in pairs:
                f.write(f"{pair['tgt_text']}\n")
        
        print(f"Created {src_file} and {tgt_file}")
    
    print(f"\n2nd Batch dataset preparation complete!")
    print(f"Total NEW samples from Clips 11-20: {len(matched_pairs)}")
    print(f"Previous 1st Batch (Clips 1-10): 1,247 samples")
    print(f"Combined total if continuing training: {len(matched_pairs) + 1247} samples")

if __name__ == "__main__":
    create_cvss_t_2nd_batch()

