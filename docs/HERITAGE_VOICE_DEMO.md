# Heritage Voice Clone Demo — Walkthrough

This document accompanies the notebook `notebooks/heritage_voice_clone_demo.ipynb` and explains how to clone a heritage voice using VintageVoice in about an hour on an 8GB GPU.

## Overview

VintageVoice is an F5-TTS fine-tune that transfers historical speech patterns (transatlantic, newsreel, radio drama, etc.) onto any reference voice. You provide:
- A **reference audio** clip (5–15 seconds, clean, 24 kHz WAV recommended)
- The **exact transcript** of that clip
- A **text prompt** you want the cloned voice to say

The model outputs a WAV file with the same speaker identity but historical delivery.

## Step-by-Step

### 1. System Requirements
- NVIDIA GPU with ≥8GB VRAM (e.g., RTX 3070/4060/5070, T4, V100-8GB)
- Python 3.10+ 
- ~10GB free disk space for model weights

### 2. Install Dependencies

```bash
pip install f5-tts huggingface-hub
```

### 3. Clone this Repository

```bash
git clone https://github.com/Scottcjn/vintage-voice.git
cd vintage-voice
```

### 4. Download Model Weights

```bash
huggingface-cli download AutomatedJanitor/vintage-voice \
    model.safetensors vocab.txt \
    --local-dir ./weights
```

### 5. Prepare Your Reference Audio
- Use a recording of the speaker you want to clone (family member, community elder)
- Trim to 5–15 seconds, remove silence at start/end
- Save as 24 kHz mono WAV (or 16 kHz; 24 kHz is optimal)
- **Critically:** Write down the exact words spoken in that clip — this is `--ref-text`

### 6. Run Generation

```bash
# Example with transatlantic preset
python scripts/generate.py \
    "Good evening, I am speaking in my heritage voice." \
    --preset transatlantic \
    --model ./weights/model.safetensors \
    --vocab ./weights/vocab.txt \
    --ref-audio /path/to/reference.wav \
    --ref-text "The exact transcript of the reference audio." \
    --output ./output/cloned_voice.wav
```

### 7. Listen and Share
- Output is in `./output/cloned_voice.wav`
- Try different presets: `newsreel`, `fireside`, `radio_drama`, `edison`, `wartime`, `announcer`
- For best results, choose a preset that matches the heritage style you want

## Tips for Best Quality

- **Use a clean, well-mic'ed reference** — background noise degrades output
- **Provide `--ref-text` exactly** — leaving it empty leaks ~0.5s of the reference into the generation
- **Keep generation text under 200 characters** for v0.1.0 (longer texts may lose coherence)
- **Lower `--speed` (e.g., 0.85)** for more vintage-like pacing

## What If I Want to Fine-Tune for a Specific Heritage Language?

The notebook above is for *inference* only. If you want to fine-tune VintageVoice on your own heritage language data (e.g., Cajun French, Yiddish, Navajo), see:
- [`scripts/prep_cosyvoice_data.py`](../scripts/prep_cosyvoice_data.py) — data preparation
- [`scripts/run_pipeline.sh`](../scripts/run_pipeline.sh) — full training pipeline
- [`FAMILY_RECORDING_GUIDE.md`](../FAMILY_RECORDING_GUIDE.md) — how to collect heritage recordings

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `CUDA out of memory` | Reduce batch size (not used here) or use smaller reference; ensure no other GPU tasks |
| Garbled audio | Check `--model` points to the correct architecture (`F5TTS_v1_Base`) |
| Reference voice bleeding | Provide `--ref-text`; see warning in scripts/generate.py |
| Slow generation | Expect ~2–5 seconds per real-time second on a 8GB GPU |

## License
- Notebook code: MIT
- Model weights: CC-BY-NC-4.0 (inherited from F5-TTS)
- This walkthrough: MIT