"""
Preprocessing script for CVSS-T dataset.
This script handles the specific format and requirements of CVSS-T data.
"""

import os
import json
import argparse
import pandas as pd
import torch
import torchaudio
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from examples.speech_to_text.data_utils import load_df_from_tsv, save_df_to_tsv
from preprocess_scripts.data_utils import gen_config_yaml


class CVSS_T_DataProcessor:
    """
    Data processor for CVSS-T dataset.
    Handles the specific format and preprocessing requirements.
    """
    
    def __init__(self, cvss_t_root: str, output_dir: str):
        self.cvss_t_root = Path(cvss_t_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # CVSS-T specific parameters
        self.sample_rate = 16000
        self.target_sample_rate = 16000
        
    def load_cvss_t_metadata(self) -> Dict[str, List[Dict]]:
        """
        Load CVSS-T dataset metadata.
        CVSS-T typically has train/dev/test splits with source and target audio.
        """
        metadata = {"train": [], "dev": [], "test": []}
        
        for split in ["train", "dev", "test"]:
            split_dir = self.cvss_t_root / split
            if not split_dir.exists():
                print(f"Warning: {split} directory not found in {self.cvss_t_root}")
                continue
                
            # Load split metadata (assuming JSON format)
            metadata_file = split_dir / f"{split}.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    split_data = json.load(f)
                    metadata[split] = split_data
            else:
                # Alternative: scan directory for audio files
                metadata[split] = self._scan_audio_files(split_dir)
        
        return metadata
    
    def _scan_audio_files(self, split_dir: Path) -> List[Dict]:
        """Scan directory for audio files and create metadata."""
        audio_files = []
        
        # Look for common audio formats
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
        
        for ext in audio_extensions:
            for audio_file in split_dir.glob(f"*{ext}"):
                # Extract metadata from filename or directory structure
                sample_id = audio_file.stem
                
                # Determine if it's source or target audio
                is_source = "source" in audio_file.name.lower() or "src" in audio_file.name.lower()
                is_target = "target" in audio_file.name.lower() or "tgt" in audio_file.name.lower()
                
                audio_files.append({
                    "id": sample_id,
                    "audio_path": str(audio_file),
                    "is_source": is_source,
                    "is_target": is_target,
                    "duration": self._get_audio_duration(audio_file)
                })
        
        return audio_files
    
    def _get_audio_duration(self, audio_path: Path) -> float:
        """Get audio duration in seconds."""
        try:
            info = torchaudio.info(str(audio_path))
            return info.num_frames / info.sample_rate
        except:
            return 0.0
    
    def extract_audio_features(self, audio_path: str) -> Dict[str, Any]:
        """
        Extract audio features for CVSS-T.
        This includes mel-spectrograms, speaker embeddings, and emotion features.
        """
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Resample if necessary
            if sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
                waveform = resampler(waveform)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Extract mel-spectrogram
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.target_sample_rate,
                n_fft=1024,
                hop_length=256,
                n_mels=80
            )
            mel_spec = mel_transform(waveform)
            
            # Extract features
            features = {
                "mel_spectrogram": mel_spec,
                "waveform": waveform,
                "duration": waveform.shape[1] / self.target_sample_rate,
                "sample_rate": self.target_sample_rate
            }
            
            return features
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None
    
    def extract_units_from_audio(self, audio_path: str, hubert_model_path: str) -> List[int]:
        """
        Extract discrete units from audio using HuBERT.
        This is a placeholder implementation.
        """
        # In real implementation, this would use HuBERT to extract units
        # For now, return placeholder units
        features = self.extract_audio_features(audio_path)
        if features is None:
            return []
        
        # Placeholder: generate random units based on duration
        duration = features["duration"]
        num_units = int(duration * 50)  # ~50 units per second
        units = list(range(num_units % 1000))  # Placeholder units
        
        return units
    
    def process_split(self, split_data: List[Dict], split_name: str) -> pd.DataFrame:
        """
        Process a single split of CVSS-T data.
        """
        processed_data = []
        
        for item in tqdm(split_data, desc=f"Processing {split_name}"):
            # Extract features from target audio
            if item.get("is_target", False):
                features = self.extract_audio_features(item["audio_path"])
                if features is None:
                    continue
                
                # Extract units (placeholder)
                units = self.extract_units_from_audio(item["audio_path"], "")
                
                # Create processed item
                processed_item = {
                    "id": item["id"],
                    "tgt_audio": item["audio_path"],
                    "tgt_text": "",  # Will be filled from transcriptions
                    "units": " ".join(map(str, units)),
                    "duration": features["duration"],
                    "speaker_id": self._extract_speaker_id(item["id"]),
                    "emotion_id": self._extract_emotion_id(item["id"]),
                }
                
                # Add source audio if available
                source_item = self._find_source_audio(item, split_data)
                if source_item:
                    processed_item["src_audio"] = source_item["audio_path"]
                
                processed_data.append(processed_item)
        
        return pd.DataFrame(processed_data)
    
    def _extract_speaker_id(self, sample_id: str) -> int:
        """Extract speaker ID from sample ID (placeholder)."""
        # In real implementation, this would extract from metadata or filename
        return hash(sample_id) % 1000
    
    def _extract_emotion_id(self, sample_id: str) -> int:
        """Extract emotion ID from sample ID (placeholder)."""
        # In real implementation, this would extract from metadata or filename
        return hash(sample_id) % 8
    
    def _find_source_audio(self, target_item: Dict, split_data: List[Dict]) -> Dict:
        """Find corresponding source audio for target item."""
        # Look for source audio with matching ID
        for item in split_data:
            if (item.get("is_source", False) and 
                item["id"] == target_item["id"]):
                return item
        return None
    
    def create_manifest_files(self, processed_data: Dict[str, pd.DataFrame]):
        """Create manifest files for each split."""
        for split_name, df in processed_data.items():
            if df.empty:
                continue
                
            # Save TSV file
            tsv_path = self.output_dir / f"{split_name}.tsv"
            save_df_to_tsv(df, tsv_path)
            
            # Create audio directory structure
            audio_dir = self.output_dir / "audio"
            audio_dir.mkdir(exist_ok=True)
            
            print(f"Created manifest for {split_name}: {tsv_path}")
    
    def create_config_yaml(self):
        """Create configuration YAML for CVSS-T data."""
        config_path = self.output_dir / "config.yaml"
        
        gen_config_yaml(
            manifest_root=self.output_dir,
            yaml_filename="config.yaml",
            specaugment_policy="lb",
            cmvn_type="global",
            gcmvn_path=self.output_dir / "gcmvn.npz",
            input_channels=1,
            input_feat_per_channel=80,
            audio_root=str(self.output_dir / "audio"),
            vocoder_type="CodeHiFiGANVocoderWithDur",
            vocoder_checkpoint="pretrain_models/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.en/g_00500000",
            vocoder_cfg="pretrain_models/unit-based_HiFi-GAN_vocoder/mHuBERT.layer11.km1000.en/config_cvss_t.json",
        )
        
        print(f"Created config YAML: {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess CVSS-T dataset")
    parser.add_argument("--cvss-t-root", type=str, required=True, 
                       help="Root directory of CVSS-T dataset")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for processed data")
    parser.add_argument("--hubert-model", type=str, 
                       help="Path to HuBERT model for unit extraction")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = CVSS_T_DataProcessor(args.cvss_t_root, args.output_dir)
    
    # Load metadata
    print("Loading CVSS-T metadata...")
    metadata = processor.load_cvss_t_metadata()
    
    # Process each split
    processed_data = {}
    for split_name, split_data in metadata.items():
        if split_data:
            print(f"Processing {split_name} split...")
            df = processor.process_split(split_data, split_name)
            processed_data[split_name] = df
    
    # Create manifest files
    print("Creating manifest files...")
    processor.create_manifest_files(processed_data)
    
    # Create config YAML
    print("Creating configuration...")
    processor.create_config_yaml()
    
    print("CVSS-T preprocessing completed!")
    print(f"Processed data saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
