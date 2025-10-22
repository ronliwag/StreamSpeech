#!/usr/bin/env python3
"""
CVSS-T Subset Data Preparation Script
Creates manifest files for fine-tuning StreamSpeech on CVSS-T subset data.
"""

import os
import pandas as pd
import json
from pathlib import Path
import random

def create_cvss_t_subset():
    """Create CVSS-T subset manifest files for fine-tuning."""
    
    # Base paths - use relative paths for better compatibility
    base_path = Path("E:/Thesis_Datasets")
    spanish_batch_path = base_path / "CommonVoice_v4/es/cv-corpus-22.0-2025-06-20/es/1st Batch - 5000"
    english_batch_path = base_path / "CommonVoice_v4/en/1st Batch - 5000"
    output_path = Path("cvss-t-finetune/es-en/fbank2unit")
    
    # CommonVoice TSV files
    spanish_tsv = base_path / "CommonVoice_v4/es/cv-corpus-22.0-2025-06-20/es/train.tsv"
    english_tsv = base_path / "CommonVoice_v4/en/train.tsv"
    
    # CoVoST mapping file
    covost_file = base_path / "CoVoST/covost_v2.es_en.tsv"
    
    print("Reading CommonVoice TSV files...")
    
    # Read Spanish CommonVoice data
    spanish_df = pd.read_csv(spanish_tsv, sep='\t')
    print(f"Found {len(spanish_df)} Spanish samples in CommonVoice")
    
    # Read English CommonVoice data
    english_df = pd.read_csv(english_tsv, sep='\t')
    print(f"Found {len(english_df)} English samples in CommonVoice")
    
    # Read CoVoST mapping
    covost_df = pd.read_csv(covost_file, sep='\t')
    train_covost = covost_df[covost_df['split'] == 'train'].copy()
    print(f"Found {len(train_covost)} train samples in CoVoST")
    
    # Get Spanish audio files from all 10 batches
    spanish_files = []
    for batch_num in range(1, 11):
        batch_path = spanish_batch_path / f"Clips - Batch {batch_num}"
        if batch_path.exists():
            files = list(batch_path.glob("*.mp3"))
            spanish_files.extend(files)
    
    print(f"Found {len(spanish_files)} Spanish audio files in all batches")
    
    # Get English audio files from all 10 batches
    english_files = []
    for batch_num in range(1, 11):
        batch_path = english_batch_path / f"Clips - Batch {batch_num}"
        if batch_path.exists():
            files = list(batch_path.glob("*.mp3"))
            english_files.extend(files)
    
    print(f"Found {len(english_files)} English audio files in all batches")
    
    # Create mapping dictionaries
    spanish_map = {f.stem: f for f in spanish_files}
    english_map = {f.stem: f for f in english_files}
    
    # Create Spanish text mapping from CommonVoice TSV
    spanish_text_map = {}
    for _, row in spanish_df.iterrows():
        spanish_text_map[row['path']] = row['sentence']
    
    # Find matching pairs using CoVoST
    matched_pairs = []
    for _, row in train_covost.iterrows():
        spanish_file = row['path']
        spanish_id = spanish_file.replace('.mp3', '')
        
        # Check if we have this Spanish file in our batches
        if spanish_id in spanish_map and spanish_file in spanish_text_map:
            # For CVSS-T, we need to find corresponding English audio
            # This is a simplified approach - select random English file
            if english_files:
                english_file = random.choice(english_files)
                
                matched_pairs.append({
                    'id': spanish_id,
                    'src_audio': str(spanish_map[spanish_id]).replace('\\', '/'),
                    'tgt_audio': str(english_file).replace('\\', '/'),
                    'src_text': spanish_text_map[spanish_file],
                    'tgt_text': row['translation']
                })
    
    # If no matches found, create pairs from available batch files
    if len(matched_pairs) == 0:
        print("No CoVoST matches found, creating pairs from batch files...")
        for i, spanish_file in enumerate(spanish_files[:5000]):
            spanish_id = spanish_file.stem
            spanish_path = spanish_file.name
            
            if spanish_path in spanish_text_map and i < len(english_files):
                english_file = english_files[i]
                
                matched_pairs.append({
                    'id': spanish_id,
                    'src_audio': str(spanish_file).replace('\\', '/'),
                    'tgt_audio': str(english_file).replace('\\', '/'),
                    'src_text': spanish_text_map[spanish_path],
                    'tgt_text': f"English translation for {spanish_id}"  # Placeholder
                })
    
    print(f"Created {len(matched_pairs)} matched pairs")
    
    # Limit to 5000 samples for fine-tuning (5k setup)
    if len(matched_pairs) > 5000:
        matched_pairs = random.sample(matched_pairs, 5000)
    
    # Split into train/dev/test
    random.shuffle(matched_pairs)
    train_size = int(0.8 * len(matched_pairs))
    dev_size = int(0.1 * len(matched_pairs))
    
    train_pairs = matched_pairs[:train_size]
    dev_pairs = matched_pairs[train_size:train_size + dev_size]
    test_pairs = matched_pairs[train_size + dev_size:]
    
    print(f"Split: {len(train_pairs)} train, {len(dev_pairs)} dev, {len(test_pairs)} test")
    
    # Create manifest files
    for split, pairs in [('train', train_pairs), ('dev', dev_pairs), ('test', test_pairs)]:
        manifest_file = output_path / f"{split}.tsv"
        
        with open(manifest_file, 'w', encoding='utf-8') as f:
            f.write("id\tsrc_audio\tsrc_n_frames\tsrc_text\ttgt_text\ttgt_audio\ttgt_n_frames\n")
            
            for pair in pairs:
                f.write(f"{pair['id']}\t{pair['src_audio']}\t0\t{pair['src_text']}\t{pair['tgt_text']}\t{pair['tgt_audio']}\t0\n")
        
        print(f"Created {manifest_file}")
    
    # Create source and target text files
    for split, pairs in [('train', train_pairs), ('dev', dev_pairs), ('test', test_pairs)]:
        # Source text (Spanish)
        src_file = output_path / f"{split}.src"
        with open(src_file, 'w', encoding='utf-8') as f:
            for pair in pairs:
                f.write(f"{pair['src_text']}\n")
        
        # Target text (English)
        tgt_file = output_path / f"{split}.txt"
        with open(tgt_file, 'w', encoding='utf-8') as f:
            for pair in pairs:
                f.write(f"{pair['tgt_text']}\n")
        
        print(f"Created {src_file} and {tgt_file}")

if __name__ == "__main__":
    create_cvss_t_subset()
