# Clone Your Heritage Voice in About an Hour — Walkthrough

This guide accompanies the `heritage_voice_clone_demo.ipynb` notebook.

## Overview

VintageVoice is an F5-TTS fine-tune trained on 164 hours of pre-1955 public-domain audio. It applies historical speech patterns (transatlantic, newsreel, fireside chat, etc.) to any modern voice while preserving speaker identity.

## Steps

### 1. Prepare a reference audio clip
- Record 5-15 seconds of someone speaking clearly
- Export as 16-bit mono WAV, 24kHz sample rate (or let the pipeline resample)
- Write down the exact transcript of what was said

### 2. Set up the environment
```bash
pip install f5-tts huggingface-hub -q
```

### 3. Download the model weights
```bash
huggingface-cli download AutomatedJanitor/vintage-voice model.safetensors vocab.txt --local-dir ./weights
```

### 4. Run generation
```bash
python scripts/generate.py "Your text here" --preset transatlantic --model ./weights/model.safetensors --vocab ./weights/vocab.txt --ref-audio /path/to/ref.wav --ref-text "exact transcript" --output ./output/cloned_voice.wav
```

### 5. Listen to the result
Output saved to `./output/cloned_voice.wav`.

## Troubleshooting

| Problem | Likely cause | Fix |
|---------|-------------|-----|
| CUDA out of memory | Batch too large for 8GB GPU | Use --batch-size 1, close other GPU apps |
| No audio output | Wrong ref-text | The transcript must match exactly |
| Robotic voice | Model weights not loaded | Re-download weights |
| Module not found | f5-tts not installed | `pip install f5-tts` |

## Presets

| Preset | Style |
|--------|-------|
| transatlantic | Mid-Atlantic accent (validated) |
| newsreel | 1940s newsreel delivery (experimental) |
| fireside | Roosevelt-style fireside chat (experimental) |
| radio_drama | Old-time radio drama (experimental) |
| edison | Early Edison recording (experimental) |
| wartime | WWII-era broadcast (experimental) |
| announcer | Formal announcer (experimental) |

## RTC Wallet

xiaoduo8
