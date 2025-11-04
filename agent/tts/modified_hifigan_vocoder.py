import json
import logging
from typing import Dict, Optional
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from fairseq.models import BaseFairseqModel, register_model

logger = logging.getLogger(__name__)


@register_model("ModifiedHiFiGANVocoder")
class ModifiedHiFiGANVocoder(BaseFairseqModel):
    """
    Wrapper for modified HiFiGAN with FiLM conditioning support.
    Compatible with StreamSpeech's vocoder interface.
    """
    
    def __init__(
        self, 
        checkpoint_path: str, 
        model_cfg: Dict[str, str], 
        fp16: bool = False
    ) -> None:
        super().__init__()
        
        import sys
        import os
        
        # Add workspace root to path so we can import from modifications package
        workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        if workspace_root not in sys.path:
            sys.path.insert(0, workspace_root)
        
        from modifications.hifigan_generator import UnitHiFiGANGenerator
        
        self.model = UnitHiFiGANGenerator(config=model_cfg, use_film=True)
        
        if torch.cuda.is_available():
            state_dict = torch.load(checkpoint_path)
        else:
            state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        
        if "ema_generator" in state_dict:
            logger.info("Loading EMA generator weights (most stable)")
            self.model.load_state_dict(state_dict["ema_generator"])
        elif "generator" in state_dict:
            logger.info("Loading generator weights")
            self.model.load_state_dict(state_dict["generator"])
        else:
            logger.info("Loading state dict directly")
            self.model.load_state_dict(state_dict)
        
        self.model.eval()
        
        if fp16:
            self.model.half()
        
        self.model.remove_weight_norm()
        
        logger.info(f"Loaded modified HiFiGAN checkpoint from {checkpoint_path}")
        
        self.speaker_embedding = None
        self.emotion_embedding = None
    
    def set_film_conditioning(
        self, 
        speaker: Optional[torch.Tensor] = None,
        emotion: Optional[torch.Tensor] = None
    ):
        """
        Set FiLM conditioning embeddings for voice cloning.
        
        Args:
            speaker: Speaker embedding [192] from ECAPA-TDNN
            emotion: Emotion embedding [768] from Emotion2Vec
        """
        self.speaker_embedding = speaker
        self.emotion_embedding = emotion
        
        if speaker is not None and emotion is not None:
            logger.info(
                f"FiLM conditioning set: speaker={speaker.shape}, emotion={emotion.shape}"
            )
    
    def forward(
        self, 
        x: Dict[str, torch.Tensor], 
        dur_prediction: bool = False
    ) -> tuple:
        """
        Generate audio from discrete units with optional FiLM conditioning.
        
        Args:
            x: Dictionary containing 'code' key with unit IDs [B, T]
            dur_prediction: Compatibility parameter (not used by modified HiFiGAN)
        
        Returns:
            wav: Generated waveform
            dur: Duration predictions (dummy for compatibility)
        """
        assert "code" in x, "Input must contain 'code' key with unit IDs"
        
        mask = x["code"] >= 0
        units = x["code"][mask].unsqueeze(dim=0)
        
        device = units.device
        
        speaker = self.speaker_embedding
        emotion = self.emotion_embedding
        
        if speaker is not None:
            speaker = speaker.unsqueeze(0).to(device)
        if emotion is not None:
            emotion = emotion.unsqueeze(0).to(device)
        
        wav = self.model(
            units=units,
            speaker=speaker,
            emotion=emotion
        )
        
        wav = wav.detach().squeeze()
        
        dummy_dur = torch.ones(1, units.size(1), dtype=torch.long, device=device)
        
        return wav, dummy_dur
    
    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        config_path: str,
        fp16: bool = False,
        **kwargs
    ):
        """
        Load pretrained modified HiFiGAN model.
        
        Args:
            checkpoint_path: Path to best_model.pt
            config_path: Path to base_hifigan_config.json
            fp16: Use half precision
        """
        with open(config_path, 'r') as f:
            model_cfg = json.load(f)
        
        return cls(checkpoint_path, model_cfg, fp16=fp16)

