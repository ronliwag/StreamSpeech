# 🎯 Quick Reference: Extract Intermediates from StreamSpeech

## Two Code Locations

### 1️⃣ Source Spanish Audio @ 16kHz → `demo/app.py` line ~859

**Add after**: `samples = samples / max_val * 0.8`

```python
# Save source audio at 16kHz
import soundfile, os, torch
source_filename = os.path.basename(source).rsplit('.', 1)[0]
extract_dir = os.path.join(os.path.dirname(__file__), 'extracted_intermediates')
os.makedirs(extract_dir, exist_ok=True)

# Resample to 16kHz (model's processing rate)
TARGET_SR = 16000
if ORG_SAMPLE_RATE != TARGET_SR:
    samples_tensor = torch.tensor(samples).unsqueeze(0).unsqueeze(0)
    target_length = int(len(samples) * TARGET_SR / ORG_SAMPLE_RATE)
    samples_16k = torch.nn.functional.interpolate(
        samples_tensor, size=target_length, mode='linear', align_corners=False
    ).squeeze().numpy()
else:
    samples_16k = samples

source_audio_path = os.path.join(extract_dir, f"{source_filename}_source_audio_16k.wav")
soundfile.write(source_audio_path, samples_16k, TARGET_SR)
print(f"✅ Source audio (16kHz): {source_audio_path}")
```

**Output**: `demo/extracted_intermediates/<filename>_source_audio_16k.wav`

---

### 2️⃣ Discrete Units → `agent/speech_to_speech.streamspeech.agent.py` line ~742

**Add before**: `x = {"code": torch.tensor(unit, ...}`

```python
# Save discrete units
if self.states.source_finished and len(unit) > 0:
    import torch, os
    from datetime import datetime
    extract_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'demo', 'extracted_intermediates')
    os.makedirs(extract_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as PyTorch tensor
    units_tensor = torch.tensor(unit, dtype=torch.long)
    torch.save(units_tensor, os.path.join(extract_dir, f"units_{timestamp}.pt"))
    
    # Save as text
    with open(os.path.join(extract_dir, f"units_{timestamp}.txt"), 'w') as f:
        f.write(' '.join(map(str, unit)))
    print(f"✅ Discrete units: {len(unit)} units saved")
```

**Output**: 
- `demo/extracted_intermediates/units_<timestamp>.pt`
- `demo/extracted_intermediates/units_<timestamp>.txt`

---

## 📊 What You Get

| File | Format | Content | Size |
|------|--------|---------|------|
| `*_source_audio_16k.wav` | WAV, **16kHz**, float32 | Spanish speech at model's rate | ~1-2 MB/min |
| `units_*.pt` | PyTorch tensor | Discrete phonetic codes | ~1-2 KB |
| `units_*.txt` | Text | Human-readable units | ~1-2 KB |

---

## 🔬 Usage

### Load Source Audio
```python
import soundfile
audio, sr = soundfile.read('common_voice_es_18311412_source_audio_16k.wav')
print(f"Sample rate: {sr} Hz")  # Should be 16000
```

### Load Discrete Units
```python
import torch
units = torch.load('units_20251107_025030.pt')
# tensor([234, 567, 891, ...])
```

---

## ✨ Quick Test

1. Add the two code snippets above
2. Restart demo: `python demo/app.py`
3. Upload & process Spanish audio
4. Check `demo/extracted_intermediates/` folder

---

See **EXTRACTION_GUIDE.md** for detailed explanations and advanced usage.

