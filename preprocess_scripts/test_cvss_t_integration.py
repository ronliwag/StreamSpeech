"""
Integration testing script for CVSS-T system.
This script tests the complete pipeline from audio input to synthesized output.
"""

import os
import json
import argparse
import torch
import torchaudio
from pathlib import Path
from typing import Dict, List, Any
import logging
import time
import numpy as np

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from agent.speech_to_speech.streamspeech.agent import StreamSpeechS2STAgent
from agent.cvss_t_conditioning_helper import CVSS_T_AgentIntegration


class CVSS_T_IntegrationTester:
    """
    Integration tester for CVSS-T system.
    Tests the complete pipeline with real audio inputs.
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.conditioning_helper = CVSS_T_AgentIntegration(device)
        
        # Performance metrics
        self.latency_measurements = []
        self.quality_scores = []
        
    def load_streamspeech_agent(self, model_path: str, data_bin: str, 
                              config_yaml: str, vocoder_path: str, 
                              vocoder_config: str) -> StreamSpeechS2STAgent:
        """Load StreamSpeech S2ST agent with CVSS-T enhanced vocoder."""
        
        # Create agent arguments
        class AgentArgs:
            def __init__(self):
                self.model_path = model_path
                self.data_bin = data_bin
                self.config_yaml = config_yaml
                self.vocoder = vocoder_path
                self.vocoder_cfg = vocoder_config
                self.device = "gpu" if self.device == "cuda" else "cpu"
                self.dur_prediction = True
                self.lagging_k1 = 0
                self.lagging_k2 = 0
                self.segment_size = 320
                self.stride_n = 1
                self.stride_n2 = 1
                self.unit_per_subword = 15
                self.extra_output_dir = None
                self.output_asr_translation = False
                self.source_segment_size = 640
                self.max_len = 200
                self.force_finish = False
                self.shift_size = 10
                self.window_size = 25
                self.sample_rate = 48000
                self.feature_dim = 80
                self.global_cmvn = None
                self.multitask_config_yaml = None
                self.tgt_splitter_type = "SentencePiece"
                self.tgt_splitter_path = None
                self.user_dir = "researches/ctc_unity"
                self.agent_dir = "agent"
        
        args = AgentArgs()
        args.model_path = model_path
        args.data_bin = data_bin
        args.config_yaml = config_yaml
        args.vocoder = vocoder_path
        args.vocoder_cfg = vocoder_config
        args.device = "gpu" if device == "cuda" else "cpu"
        
        # Initialize agent
        agent = StreamSpeechS2STAgent(args)
        
        print(f"Loaded StreamSpeech agent with CVSS-T enhanced vocoder")
        return agent
    
    def test_audio_processing_pipeline(self, agent: StreamSpeechS2STAgent, 
                                     test_audio_path: str) -> Dict[str, Any]:
        """
        Test the complete audio processing pipeline.
        
        Args:
            agent: StreamSpeech S2ST agent
            test_audio_path: Path to test audio file
            
        Returns:
            Dictionary containing test results
        """
        print(f"Testing audio processing pipeline with {test_audio_path}")
        
        # Load test audio
        try:
            waveform, sample_rate = torchaudio.load(test_audio_path)
            print(f"Loaded audio: {waveform.shape}, sample_rate: {sample_rate}")
        except Exception as e:
            print(f"Error loading audio: {e}")
            return {"error": str(e)}
        
        # Test results
        results = {
            "audio_path": test_audio_path,
            "input_shape": list(waveform.shape),
            "sample_rate": sample_rate,
            "processing_time": 0.0,
            "output_audio_length": 0,
            "success": False
        }
        
        try:
            # Reset agent state
            agent.reset()
            
            # Process audio in chunks (simulating streaming)
            chunk_size = 16000  # 1 second chunks
            total_chunks = waveform.shape[1] // chunk_size
            
            start_time = time.time()
            output_audio = []
            
            for chunk_idx in range(total_chunks):
                start_sample = chunk_idx * chunk_size
                end_sample = min(start_sample + chunk_size, waveform.shape[1])
                chunk = waveform[:, start_sample:end_sample]
                
                # Convert to numpy for agent processing
                chunk_np = chunk.squeeze().numpy()
                
                # Process chunk through agent
                action = agent.policy()
                
                if hasattr(action, 'content') and action.content:
                    output_audio.extend(action.content)
                
                print(f"Processed chunk {chunk_idx + 1}/{total_chunks}")
            
            processing_time = time.time() - start_time
            results["processing_time"] = processing_time
            results["output_audio_length"] = len(output_audio)
            results["success"] = True
            
            print(f"Pipeline test completed successfully")
            print(f"Processing time: {processing_time:.2f} seconds")
            print(f"Output audio length: {len(output_audio)} samples")
            
        except Exception as e:
            print(f"Error in pipeline test: {e}")
            results["error"] = str(e)
        
        return results
    
    def test_conditioning_features(self, test_audio_path: str) -> Dict[str, Any]:
        """
        Test CVSS-T conditioning feature extraction.
        
        Args:
            test_audio_path: Path to test audio file
            
        Returns:
            Dictionary containing conditioning test results
        """
        print(f"Testing conditioning features with {test_audio_path}")
        
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(test_audio_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Test conditioning feature extraction
            conditioning = self.conditioning_helper.conditioning_helper.prepare_conditioning_for_vocoder(
                target_audio=waveform.squeeze(),
                speaker_id=torch.tensor([0], dtype=torch.long, device=self.device),
                emotion_id=torch.tensor([0], dtype=torch.long, device=self.device)
            )
            
            results = {
                "audio_path": test_audio_path,
                "conditioning_features": {
                    key: value.shape if hasattr(value, 'shape') else str(value)
                    for key, value in conditioning.items()
                },
                "success": True
            }
            
            print("Conditioning feature extraction test completed successfully")
            for key, value in conditioning.items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: {value.shape}")
                else:
                    print(f"  {key}: {value}")
            
        except Exception as e:
            print(f"Error in conditioning test: {e}")
            results = {"error": str(e), "success": False}
        
        return results
    
    def test_vocoder_integration(self, vocoder_path: str, vocoder_config: str,
                               test_units: List[int]) -> Dict[str, Any]:
        """
        Test vocoder integration with CVSS-T conditioning.
        
        Args:
            vocoder_path: Path to vocoder model
            vocoder_config: Path to vocoder config
            test_units: List of test units
            
        Returns:
            Dictionary containing vocoder test results
        """
        print("Testing vocoder integration with CVSS-T conditioning")
        
        try:
            # Load vocoder
            with open(vocoder_config, 'r') as f:
                config = json.load(f)
            
            from agent.tts.vocoder import CodeHiFiGANVocoderWithDur
            vocoder = CodeHiFiGANVocoderWithDur(vocoder_path, config)
            vocoder = vocoder.to(self.device)
            
            # Test basic vocoder functionality
            basic_input = {
                "code": torch.tensor(test_units, dtype=torch.long, device=self.device).view(1, -1)
            }
            
            start_time = time.time()
            basic_output, dur = vocoder(basic_input, dur_prediction=True)
            basic_time = time.time() - start_time
            
            # Test CVSS-T enhanced vocoder functionality
            enhanced_input = self.conditioning_helper.prepare_vocoder_input(
                unit=test_units,
                speaker_id=0,
                emotion_id=0
            )
            
            start_time = time.time()
            enhanced_output, dur = vocoder(enhanced_input, dur_prediction=True)
            enhanced_time = time.time() - start_time
            
            results = {
                "vocoder_path": vocoder_path,
                "basic_output_shape": list(basic_output.shape),
                "enhanced_output_shape": list(enhanced_output.shape),
                "basic_processing_time": basic_time,
                "enhanced_processing_time": enhanced_time,
                "success": True
            }
            
            print("Vocoder integration test completed successfully")
            print(f"Basic output shape: {basic_output.shape}")
            print(f"Enhanced output shape: {enhanced_output.shape}")
            print(f"Basic processing time: {basic_time:.4f}s")
            print(f"Enhanced processing time: {enhanced_time:.4f}s")
            
        except Exception as e:
            print(f"Error in vocoder integration test: {e}")
            results = {"error": str(e), "success": False}
        
        return results
    
    def run_comprehensive_test(self, test_config: Dict[str, str]) -> Dict[str, Any]:
        """
        Run comprehensive integration test.
        
        Args:
            test_config: Dictionary containing test configuration
            
        Returns:
            Dictionary containing all test results
        """
        print("Running comprehensive CVSS-T integration test...")
        
        all_results = {
            "test_config": test_config,
            "tests": {},
            "overall_success": False
        }
        
        try:
            # Test 1: Conditioning features
            if "test_audio" in test_config:
                conditioning_results = self.test_conditioning_features(
                    test_config["test_audio"]
                )
                all_results["tests"]["conditioning_features"] = conditioning_results
            
            # Test 2: Vocoder integration
            if "vocoder_path" in test_config and "vocoder_config" in test_config:
                test_units = [0, 1, 2, 3, 4, 5] * 10  # Placeholder units
                vocoder_results = self.test_vocoder_integration(
                    test_config["vocoder_path"],
                    test_config["vocoder_config"],
                    test_units
                )
                all_results["tests"]["vocoder_integration"] = vocoder_results
            
            # Test 3: Complete pipeline (if agent config available)
            if all(key in test_config for key in ["model_path", "data_bin", "config_yaml"]):
                agent = self.load_streamspeech_agent(
                    test_config["model_path"],
                    test_config["data_bin"],
                    test_config["config_yaml"],
                    test_config.get("vocoder_path", ""),
                    test_config.get("vocoder_config", "")
                )
                
                if "test_audio" in test_config:
                    pipeline_results = self.test_audio_processing_pipeline(
                        agent, test_config["test_audio"]
                    )
                    all_results["tests"]["audio_processing_pipeline"] = pipeline_results
            
            # Determine overall success
            all_results["overall_success"] = all(
                test_result.get("success", False) 
                for test_result in all_results["tests"].values()
            )
            
        except Exception as e:
            print(f"Error in comprehensive test: {e}")
            all_results["error"] = str(e)
        
        return all_results
    
    def save_test_results(self, results: Dict[str, Any], output_path: str):
        """Save test results to file."""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Test results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Test CVSS-T integration")
    parser.add_argument("--test-config", type=str, required=True,
                       help="Path to test configuration JSON")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for test results")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load test configuration
    with open(args.test_config, 'r') as f:
        test_config = json.load(f)
    
    # Initialize tester
    tester = CVSS_T_IntegrationTester(args.device)
    
    # Run comprehensive test
    results = tester.run_comprehensive_test(test_config)
    
    # Save results
    results_path = os.path.join(args.output_dir, "integration_test_results.json")
    tester.save_test_results(results, results_path)
    
    # Print summary
    print("\n=== CVSS-T Integration Test Summary ===")
    print(f"Overall success: {results['overall_success']}")
    
    for test_name, test_result in results["tests"].items():
        status = "PASS" if test_result.get("success", False) else "FAIL"
        print(f"{test_name}: {status}")
        if "error" in test_result:
            print(f"  Error: {test_result['error']}")


if __name__ == "__main__":
    main()
