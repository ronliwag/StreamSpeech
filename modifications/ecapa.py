import torch
import torchaudio
from speechbrain.inference.speaker import SpeakerRecognition
import os
import shutil
from pathlib import Path


class ECAPA:

    def __init__(self, device="cuda"):
        # Patch speechbrain's symlink function to use copy instead on Windows
        if os.name == 'nt':  # Windows
            self._patch_speechbrain_symlinks()
        
        self.ecapa = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
            use_auth_token=False
        )
        self.device = device
    
    @staticmethod
    def _patch_speechbrain_symlinks():
        """
        Patch speechbrain's link_with_strategy to use file copy on Windows
        instead of symlinks (which require admin privileges)
        """
        import speechbrain.utils.fetching as fetching
        
        original_link = fetching.link_with_strategy
        
        def copy_instead_of_link(src, dst, local_strategy):
            """Copy files instead of creating symlinks on Windows"""
            src = Path(src)
            dst = Path(dst)
            
            # Create parent directory if it doesn't exist
            dst.parent.mkdir(parents=True, exist_ok=True)
            
            # If destination already exists and is a symlink or file, remove it
            if dst.exists() or dst.is_symlink():
                if dst.is_dir() and not dst.is_symlink():
                    shutil.rmtree(dst)
                else:
                    dst.unlink()
            
            # Copy instead of symlink
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
            
            return dst
        
        fetching.link_with_strategy = copy_instead_of_link


    @torch.no_grad()
    def extract_speaker_embeddings(self, wav_path):
        signal, sample_rate = torchaudio.load(wav_path)

        # Move audio to same device as model
        signal = signal.to(self.device)

        # Returns tensor with shape [1, 1, 192] (need to convert to 1D)
        raw_embeddings = self.ecapa.encode_batch(signal)

        # Remove unneccesary dimensions (batch=1, time=1)
        # Reshapes to [192]
        embeddings = raw_embeddings.squeeze()
        return embeddings