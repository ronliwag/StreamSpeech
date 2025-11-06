# StreamSpeech Intermediate Data Extraction Guide

This guide shows you **exactly where** to extract the source Spanish audio and discrete speech units from StreamSpeech.

---

## 📁 What You'll Extract

1. **Source Spanish Audio Input** (`.wav` file)
   - Original/resampled Spanish audio before feature extraction
   - Location: `demo/app.py`, `run()` function

2. **Discrete Speech Units** (`.pt` file) 
   - Integer codes representing phonetic units (before vocoder)
   - Location: `agent/speech_to_speech.streamspeech.agent.py`, `policy()` method

---

## 🎯 Code Location 1: Source Audio Input

### File: `demo/app.py`

**Location**: In the `run()` function, around **line 859**

### Current Code:
```python
def run(source):
    # if len(S2ST)!=0: return
    
    # Handle MP3 files by converting to WAV first
    if source.lower().endswith('.mp3'):
        print(f"Converting MP3 to WAV: {source}")
        audio = AudioSegment.from_mp3(source)
        # Create a temporary WAV file
        wav_path = source.rsplit('.', 1)[0] + '_temp.wav'
        audio.export(wav_path, format='wav')
        samples, sr = soundfile.read(wav_path, dtype="float32")
        # Clean up temp file
        try:
            os.remove(wav_path)
        except:
            pass
    else:
        samples, sr = soundfile.read(source, dtype="float32")
    
    # Resample to expected sample rate if needed
    if sr != ORG_SAMPLE_RATE:
        print(f"Resampling from {sr}Hz to {ORG_SAMPLE_RATE}Hz")
        # Simple resampling using torch
        samples_tensor = torch.tensor(samples).unsqueeze(0).unsqueeze(0)
        target_length = int(len(samples) * ORG_SAMPLE_RATE / sr)
        samples_tensor = torch.nn.functional.interpolate(
            samples_tensor, size=target_length, mode='linear', align_corners=False
        )
        samples = samples_tensor.squeeze().numpy()
    
    # Normalize input audio to prevent loud playback
    max_val = np.max(np.abs(samples))
    if max_val > 0:
        samples = samples / max_val * 0.8
    
    # 👇 ADD EXTRACTION CODE HERE 👇
```

### Modified Code (ADD THIS):
```python
    # Normalize input audio to prevent loud playback
    max_val = np.max(np.abs(samples))
    if max_val > 0:
        samples = samples / max_val * 0.8
    
    # ==========================================
    # EXTRACT SOURCE AUDIO AT 16kHz
    # ==========================================
    # Save source audio for analysis (resampled to 16kHz)
    import soundfile, torch
    source_filename = os.path.basename(source).rsplit('.', 1)[0]
    extract_dir = os.path.join(os.path.dirname(__file__), 'extracted_intermediates')
    os.makedirs(extract_dir, exist_ok=True)
    
    # Resample to 16kHz (model's processing rate)
    TARGET_SR = 16000
    if ORG_SAMPLE_RATE != TARGET_SR:
        samples_tensor = torch.tensor(samples).unsqueeze(0).unsqueeze(0)
        target_length = int(len(samples) * TARGET_SR / ORG_SAMPLE_RATE)
        samples_tensor = torch.nn.functional.interpolate(
            samples_tensor, size=target_length, mode='linear', align_corners=False
        )
        samples_16k = samples_tensor.squeeze().numpy()
    else:
        samples_16k = samples
    
    source_audio_path = os.path.join(extract_dir, f"{source_filename}_source_audio_16k.wav")
    soundfile.write(source_audio_path, samples_16k, TARGET_SR)
    print(f"✅ EXTRACTED: Source audio saved to {source_audio_path}")
    print(f"   Sample rate: {TARGET_SR} Hz (16kHz), Duration: {len(samples_16k)/TARGET_SR:.2f}s")
    # ==========================================
    
    agent.reset()
    # ... rest of the function
```

---

## 🎯 Code Location 2: Discrete Speech Units

### File: `agent/speech_to_speech.streamspeech.agent.py`

**Location**: In the `policy()` method, around **line 713-748**

### Current Code:
```python
        for i, hypo in enumerate(finalized):
            i_beam = 0
            tmp = hypo[i_beam]["tokens"].int()  # hyp + eos
            if tmp[-1] == self.generator.eos:
                tmp = tmp[:-1]
            unit = []
            for c in tmp:
                u = self.generator.tgt_dict[c].replace("<s>", "").replace("</s>", "")
                if u != "":
                    unit.append(int(u))

            if len(unit) > 0 and unit[0] == " ":
                unit = unit[1:]
            text = " ".join([str(_) for _ in unit])
            if self.states.source_finished and not self.quiet:
                with open(self.unit_file, "a") as file:
                    print(text, file=file)
        cur_unit = unit if self.unit is None else unit[len(self.unit) :]
        if len(unit) < 1 or len(cur_unit) < 1:
            # ... return ReadAction or WriteAction
            
        x = {
            "code": torch.tensor(unit, dtype=torch.long, device=self.device).view(
                1, -1
            ),
        }
        wav, dur = self.vocoder(x, self.dur_prediction)
```

### Modified Code (ADD THIS):
```python
        for i, hypo in enumerate(finalized):
            i_beam = 0
            tmp = hypo[i_beam]["tokens"].int()  # hyp + eos
            if tmp[-1] == self.generator.eos:
                tmp = tmp[:-1]
            unit = []
            for c in tmp:
                u = self.generator.tgt_dict[c].replace("<s>", "").replace("</s>", "")
                if u != "":
                    unit.append(int(u))

            if len(unit) > 0 and unit[0] == " ":
                unit = unit[1:]
            text = " ".join([str(_) for _ in unit])
            if self.states.source_finished and not self.quiet:
                with open(self.unit_file, "a") as file:
                    print(text, file=file)
        cur_unit = unit if self.unit is None else unit[len(self.unit) :]
        if len(unit) < 1 or len(cur_unit) < 1:
            # ... return ReadAction or WriteAction
        
        # ==========================================
        # EXTRACT DISCRETE UNITS (before vocoder)
        # ==========================================
        # Only save when source is finished (final complete units)
        if self.states.source_finished and len(unit) > 0:
            import torch
            import os
            from datetime import datetime
            
            extract_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'demo', 'extracted_intermediates')
            os.makedirs(extract_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            units_pt_path = os.path.join(extract_dir, f"discrete_units_{timestamp}.pt")
            units_txt_path = os.path.join(extract_dir, f"discrete_units_{timestamp}.txt")
            
            # Save as PyTorch tensor
            units_tensor = torch.tensor(unit, dtype=torch.long)
            torch.save(units_tensor, units_pt_path)
            
            # Also save as readable text
            with open(units_txt_path, 'w') as f:
                f.write(' '.join(map(str, unit)) + '\n')
                f.write(f"\n# Number of units: {len(unit)}\n")
                f.write(f"# Unit range: [{min(unit)}, {max(unit)}]\n")
            
            print(f"✅ EXTRACTED: Discrete units saved to {units_pt_path}")
            print(f"   Number of units: {len(unit)}, Range: [{min(unit)}, {max(unit)}]")
        # ==========================================
            
        x = {
            "code": torch.tensor(unit, dtype=torch.long, device=self.device).view(
                1, -1
            ),
        }
        wav, dur = self.vocoder(x, self.dur_prediction)
```

---

## 📊 Data Flow Visualization

```
┌─────────────────────────────────────────────────────────────┐
│                    StreamSpeech Pipeline                     │
└─────────────────────────────────────────────────────────────┘

1. Source Spanish Audio (MP3/WAV)
   │
   ├─> Load & Resample (demo/app.py, run())
   │   ├─> samples: numpy array (48000 Hz)
   │   └─> 💾 EXTRACT HERE: source_audio.wav
   │
   ├─> Feature Extraction (agent, OnlineFeatureExtractor)
   │   └─> fbank features (80-dim)
   │
   ├─> StreamSpeech Model (encoder + decoder)
   │   ├─> ASR output (Spanish text)
   │   ├─> Translation output (English text)
   │   └─> Text-to-Unit decoder
   │
   ├─> Discrete Speech Units (agent, policy())
   │   ├─> unit: list of integers [234, 567, 891, ...]
   │   └─> 💾 EXTRACT HERE: discrete_units.pt
   │
   ├─> HiFi-GAN Vocoder (CodeHiFiGAN)
   │   └─> Synthesized English speech (16000 Hz)
   │
   └─> Output Audio (WAV)
```

---

## 🔍 Understanding the Extracted Data

### Source Audio (`.wav` file)
- **Format**: WAV, float32
- **Sample Rate**: 16000 Hz (16kHz - model's processing rate)
- **Content**: Original Spanish speech, resampled and normalized to [-0.8, 0.8]
- **Use Case**: Input to acoustic feature extraction (same rate the model uses)

### Discrete Units (`.pt` file)
- **Format**: PyTorch tensor (torch.long)
- **Content**: Integer codes representing phonetic units
- **Range**: Typically 0-999 (for 1000-unit codebook)
- **Length**: Variable, depends on speech duration
- **Example**: `tensor([234, 567, 891, 123, 456, ...])`

**How to Load:**
```python
import torch

# Load units
units = torch.load('discrete_units_20251107_123456.pt')
print(f"Shape: {units.shape}")
print(f"Units: {units}")

# Or load as text
with open('discrete_units_20251107_123456.txt', 'r') as f:
    units_str = f.readline().strip()
    units_list = [int(x) for x in units_str.split()]
```

---

## 📂 Output Structure

After running the demo, you'll find:

```
demo/
├── extracted_intermediates/
│   ├── common_voice_es_18311412_source_audio_16k.wav
│   ├── discrete_units_20251107_025030.pt
│   ├── discrete_units_20251107_025030.txt
│   ├── another_audio_source_audio_16k.wav
│   └── discrete_units_20251107_030145.pt
└── ...
```

---

## 🚀 Quick Implementation

### Option 1: Manual Copy-Paste (Recommended)
1. Open `demo/app.py`
2. Find line ~859 (after `samples = samples / max_val * 0.8`)
3. Copy-paste the "EXTRACT SOURCE AUDIO" code block
4. Open `agent/speech_to_speech.streamspeech.agent.py`
5. Find line ~742 (before `x = {"code": ...}`)
6. Copy-paste the "EXTRACT DISCRETE UNITS" code block
7. Restart the demo app

### Option 2: Use the Helper Script
The `demo/extract_intermediates.py` file contains reusable functions you can import.

---

## 🧪 Testing

1. Start the demo:
   ```powershell
   cd demo
   python app.py
   ```

2. Upload a Spanish audio file

3. Process it

4. Check the console output for:
   ```
   ✅ EXTRACTED: Source audio saved to extracted_intermediates/...
   ✅ EXTRACTED: Discrete units saved to extracted_intermediates/...
   ```

5. Verify files in `demo/extracted_intermediates/`

---

## 📝 Notes

- **Source audio** is saved at the beginning of processing (immediately available)
- **Discrete units** are saved only when `source_finished=True` (at the end)
- Both use timestamps to avoid overwriting files
- `.txt` files are human-readable for inspection
- `.pt` files can be loaded back into PyTorch for further processing

---

## 🔬 Advanced: Using the Extracted Data

### Analyzing Discrete Units
```python
import torch
import matplotlib.pyplot as plt

# Load units
units = torch.load('discrete_units_20251107_025030.pt')

# Statistics
print(f"Total units: {len(units)}")
print(f"Unique units: {len(torch.unique(units))}")
print(f"Most common unit: {torch.mode(units).values.item()}")

# Histogram
plt.hist(units.numpy(), bins=50)
plt.xlabel('Unit Index')
plt.ylabel('Frequency')
plt.title('Discrete Unit Distribution')
plt.show()
```

### Analyzing Source Audio
```python
import soundfile
import numpy as np

# Load 16kHz source audio
audio, sr = soundfile.read('common_voice_es_18311412_source_audio_16k.wav')
print(f"Sample rate: {sr} Hz (should be 16000)")
print(f"Duration: {len(audio)/sr:.2f} seconds")
print(f"Shape: {audio.shape}")
print(f"Range: [{audio.min():.3f}, {audio.max():.3f}]")
```

### Reusing Units with Vocoder
```python
import torch

# Load saved units
units = torch.load('discrete_units_20251107_025030.pt')

# Feed directly to vocoder (without running full model)
# This would be in the agent context with vocoder loaded
x = {
    "code": units.view(1, -1).to(device)
}
wav, dur = vocoder(x, dur_prediction=True)
# wav is now synthesized speech!
```

---

## ✅ Summary

**Two extraction points:**
1. 📍 `demo/app.py:859` → Save source Spanish audio WAV
2. 📍 `agent/speech_to_speech.streamspeech.agent.py:742` → Save discrete units PT

Both files will be in `demo/extracted_intermediates/` directory.

