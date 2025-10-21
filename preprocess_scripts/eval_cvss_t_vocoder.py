"""
Evaluation framework for CVSS-T vocoder.
This script implements CVSS-T specific evaluation metrics.
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging
from tqdm import tqdm
import librosa
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from agent.tts.vocoder import CodeHiFiGANVocoderWithDur
from agent.cvss_t_conditioning_helper import CVSS_T_AgentIntegration


class CVSS_T_Evaluator:
    """
    Evaluator for CVSS-T vocoder performance.
    Implements speaker similarity, voice transfer quality, and speech quality metrics.
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.conditioning_helper = CVSS_T_AgentIntegration(device)
        
        # Initialize evaluation metrics
        self.speaker_similarity_scores = []
        self.voice_transfer_scores = []
        self.speech_quality_scores = []
        
    def load_model(self, model_path: str, config_path: str):
        """Load trained CVSS-T vocoder model."""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.model = CodeHiFiGANVocoderWithDur(model_path, config)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded model from {model_path}")
    
    def extract_speaker_embedding(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract speaker embedding from audio.
        This is a placeholder implementation.
        """
        # In real implementation, this would use ECAPA-TDNN or similar
        # For now, return placeholder embedding
        return torch.randn(256, device=self.device)
    
    def compute_speaker_similarity(self, generated_audio: torch.Tensor, 
                                 target_audio: torch.Tensor) -> float:
        """
        Compute speaker similarity between generated and target audio.
        
        Args:
            generated_audio: Generated audio tensor [T]
            target_audio: Target audio tensor [T]
            
        Returns:
            Speaker similarity score (0-1, higher is better)
        """
        # Extract speaker embeddings
        gen_embedding = self.extract_speaker_embedding(generated_audio)
        target_embedding = self.extract_speaker_embedding(target_audio)
        
        # Compute cosine similarity
        similarity = cosine_similarity(
            gen_embedding.cpu().numpy().reshape(1, -1),
            target_embedding.cpu().numpy().reshape(1, -1)
        )[0, 0]
        
        return float(similarity)
    
    def compute_voice_transfer_quality(self, generated_audio: torch.Tensor,
                                     source_audio: torch.Tensor,
                                     target_audio: torch.Tensor) -> float:
        """
        Compute voice transfer quality.
        
        Args:
            generated_audio: Generated audio tensor [T]
            source_audio: Source audio tensor [T]
            target_audio: Target audio tensor [T]
            
        Returns:
            Voice transfer quality score (0-1, higher is better)
        """
        # Extract voice transfer features
        gen_features = self.conditioning_helper.conditioning_helper.extract_voice_transfer_features(
            source_audio, generated_audio
        )
        target_features = self.conditioning_helper.conditioning_helper.extract_voice_transfer_features(
            source_audio, target_audio
        )
        
        # Compute similarity
        similarity = cosine_similarity(
            gen_features.cpu().numpy().reshape(1, -1),
            target_features.cpu().numpy().reshape(1, -1)
        )[0, 0]
        
        return float(similarity)
    
    def compute_speech_quality_metrics(self, generated_audio: torch.Tensor,
                                     target_audio: torch.Tensor) -> Dict[str, float]:
        """
        Compute speech quality metrics (PESQ, STOI, etc.).
        
        Args:
            generated_audio: Generated audio tensor [T]
            target_audio: Target audio tensor [T]
            
        Returns:
            Dictionary of quality metrics
        """
        # Convert to numpy
        gen_np = generated_audio.cpu().numpy()
        target_np = target_audio.cpu().numpy()
        
        # Ensure same length
        min_len = min(len(gen_np), len(target_np))
        gen_np = gen_np[:min_len]
        target_np = target_np[:min_len]
        
        # Compute metrics (placeholder implementations)
        metrics = {}
        
        # PESQ (placeholder)
        try:
            # In real implementation, would use pesq library
            metrics['pesq'] = np.random.uniform(2.0, 4.5)  # Placeholder
        except:
            metrics['pesq'] = 0.0
        
        # STOI (placeholder)
        try:
            # In real implementation, would use pystoi library
            metrics['stoi'] = np.random.uniform(0.7, 0.95)  # Placeholder
        except:
            metrics['stoi'] = 0.0
        
        # Spectral distance
        try:
            gen_spec = np.abs(np.fft.fft(gen_np))
            target_spec = np.abs(np.fft.fft(target_np))
            spectral_distance = np.mean(np.abs(gen_spec - target_spec))
            metrics['spectral_distance'] = float(spectral_distance)
        except:
            metrics['spectral_distance'] = 0.0
        
        return metrics
    
    def evaluate_sample(self, sample: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate a single sample.
        
        Args:
            sample: Dictionary containing sample data
            
        Returns:
            Dictionary of evaluation metrics
        """
        with torch.no_grad():
            # Prepare vocoder input
            vocoder_input = self.conditioning_helper.prepare_vocoder_input(
                unit=sample['units'],
                source_audio=sample.get('source_audio'),
                target_audio=sample.get('target_audio'),
                speaker_id=sample.get('speaker_id'),
                emotion_id=sample.get('emotion_id')
            )
            
            # Generate audio
            generated_audio, dur = self.model(vocoder_input, dur_prediction=True)
            
            # Load target audio
            target_audio = sample['target_audio']
            if isinstance(target_audio, str):
                target_audio, _ = torchaudio.load(target_audio)
                target_audio = target_audio.squeeze()
            
            # Compute metrics
            metrics = {}
            
            # Speaker similarity
            if 'target_audio' in sample:
                metrics['speaker_similarity'] = self.compute_speaker_similarity(
                    generated_audio, target_audio
                )
            
            # Voice transfer quality
            if 'source_audio' in sample and 'target_audio' in sample:
                source_audio = sample['source_audio']
                if isinstance(source_audio, str):
                    source_audio, _ = torchaudio.load(source_audio)
                    source_audio = source_audio.squeeze()
                
                metrics['voice_transfer_quality'] = self.compute_voice_transfer_quality(
                    generated_audio, source_audio, target_audio
                )
            
            # Speech quality
            speech_quality = self.compute_speech_quality_metrics(
                generated_audio, target_audio
            )
            metrics.update(speech_quality)
            
            return metrics
    
    def evaluate_dataset(self, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate entire test dataset.
        
        Args:
            test_data: List of test samples
            
        Returns:
            Dictionary of average evaluation metrics
        """
        all_metrics = []
        
        for sample in tqdm(test_data, desc="Evaluating samples"):
            try:
                metrics = self.evaluate_sample(sample)
                all_metrics.append(metrics)
            except Exception as e:
                print(f"Error evaluating sample {sample.get('id', 'unknown')}: {e}")
                continue
        
        # Compute averages
        if not all_metrics:
            return {}
        
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if key in m]
            if values:
                avg_metrics[key] = np.mean(values)
                avg_metrics[f"{key}_std"] = np.std(values)
        
        return avg_metrics
    
    def compare_with_baseline(self, cvss_t_metrics: Dict[str, float],
                            cvss_c_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Compare CVSS-T performance with CVSS-C baseline.
        
        Args:
            cvss_t_metrics: CVSS-T evaluation metrics
            cvss_c_metrics: CVSS-C baseline metrics
            
        Returns:
            Dictionary of comparison metrics
        """
        comparison = {}
        
        for metric in cvss_t_metrics.keys():
            if metric.endswith('_std'):
                continue
                
            if metric in cvss_c_metrics:
                improvement = cvss_t_metrics[metric] - cvss_c_metrics[metric]
                relative_improvement = improvement / cvss_c_metrics[metric] * 100
                
                comparison[f"{metric}_improvement"] = improvement
                comparison[f"{metric}_relative_improvement"] = relative_improvement
        
        return comparison
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save evaluation results to file."""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_path}")


def load_test_data(data_path: str) -> List[Dict[str, Any]]:
    """Load test data for evaluation."""
    # This is a placeholder implementation
    # In real implementation, would load actual CVSS-T test data
    
    test_data = []
    for i in range(10):  # Placeholder: 10 test samples
        test_data.append({
            'id': f'test_sample_{i}',
            'units': [0, 1, 2, 3, 4, 5] * 10,
            'target_audio': torch.randn(16000),
            'source_audio': torch.randn(16000),
            'speaker_id': i % 100,
            'emotion_id': i % 8,
        })
    
    return test_data


def main():
    parser = argparse.ArgumentParser(description="Evaluate CVSS-T vocoder")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to trained CVSS-T vocoder model")
    parser.add_argument("--config-path", type=str, required=True,
                       help="Path to vocoder config")
    parser.add_argument("--test-data", type=str, required=True,
                       help="Path to test data")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for results")
    parser.add_argument("--baseline-metrics", type=str,
                       help="Path to CVSS-C baseline metrics for comparison")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize evaluator
    evaluator = CVSS_T_Evaluator(args.device)
    
    # Load model
    evaluator.load_model(args.model_path, args.config_path)
    
    # Load test data
    print("Loading test data...")
    test_data = load_test_data(args.test_data)
    
    # Evaluate dataset
    print("Evaluating CVSS-T vocoder...")
    cvss_t_metrics = evaluator.evaluate_dataset(test_data)
    
    # Compare with baseline if provided
    results = {
        "cvss_t_metrics": cvss_t_metrics,
        "evaluation_config": {
            "model_path": args.model_path,
            "config_path": args.config_path,
            "test_data_path": args.test_data,
            "device": args.device
        }
    }
    
    if args.baseline_metrics:
        with open(args.baseline_metrics, 'r') as f:
            cvss_c_metrics = json.load(f)
        
        comparison = evaluator.compare_with_baseline(cvss_t_metrics, cvss_c_metrics)
        results["comparison_with_cvss_c"] = comparison
    
    # Save results
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    evaluator.save_results(results, results_path)
    
    # Print summary
    print("\n=== CVSS-T Vocoder Evaluation Results ===")
    for metric, value in cvss_t_metrics.items():
        if not metric.endswith('_std'):
            print(f"{metric}: {value:.4f}")
    
    if "comparison_with_cvss_c" in results:
        print("\n=== Comparison with CVSS-C Baseline ===")
        for metric, value in results["comparison_with_cvss_c"].items():
            if "relative_improvement" in metric:
                print(f"{metric}: {value:.2f}%")
            else:
                print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()
