---
language:
  - en
license: mit
tags:
  - tts
  - text-to-speech
  - vintage
  - transatlantic
  - voice-cloning
  - f5-tts
  - digital-preservation
  - historical
  - accent
datasets:
  - custom (Archive.org public domain)
base_model: SWivid/F5-TTS
pipeline_tag: text-to-speech
---

# VintageVoice — Historical Speech Pattern TTS

**The first open-source TTS model for extinct speech patterns.**

*Proof of Antiquity for AI voices.*

## Model Description

VintageVoice is a fine-tuned [F5-TTS](https://github.com/SWivid/F5-TTS) model trained on **164 hours** of public domain vintage audio (1888-1955) from Archive.org. It learns historical speech patterns — transatlantic accent, newsreel cadence, radio drama delivery — and applies them to any modern voice via reference audio cloning.

**This is not a voice filter.** The model learns actual speech patterns: clipped consonants, rounded vowels, measured theatrical cadence, period-appropriate prosody. A filter adds crackle on top of modern speech. VintageVoice generates speech that *sounds like it was recorded in 1940*.

## Training Data

| Source | Era | Hours | Segments |
|--------|-----|-------|----------|
| Old Time Radio (The Shadow, Suspense, Mercury Theatre) | 1930s-1950s | ~80 hrs | ~18,000 |
| Prelinger Archive Newsreels | 1930s-1950s | ~30 hrs | ~8,000 |
| FDR Fireside Chats & Speeches | 1933-1944 | ~15 hrs | ~4,000 |
| WWII Broadcasts (Churchill, Murrow) | 1939-1945 | ~20 hrs | ~5,000 |
| Edison Cylinder Recordings | 1888-1920s | ~5 hrs | ~2,000 |
| Radio Commercials & Station IDs | 1930s-1960s | ~14 hrs | ~7,000 |
| **Total** | **1888-1955** | **164 hrs** | **44,345** |

All training data is **public domain**, sourced from [Archive.org](https://archive.org).

## Voice Presets

| Preset | Description |
|--------|-------------|
| `transatlantic` | Trained mid-Atlantic accent — Katharine Hepburn, Cary Grant, FDR |
| `newsreel` | "And now, the news!" Movietone/Pathe narrator |
| `fireside` | FDR Fireside Chat intimate broadcast delivery |
| `radio_drama` | The Shadow, Suspense, Mercury Theatre performance |
| `edison` | Oldest recorded humans — voices born before the Civil War |
| `wartime` | Churchill/Murrow WWII broadcast urgency |
| `announcer` | Professional radio announcer, station IDs |

## Usage

```python
from f5_tts.api import F5TTS

# Load model
tts = F5TTS(device="cuda:0")

# Load vintage fine-tuned weights
from safetensors.torch import load_file
state_dict = load_file("vintage-voice-transatlantic.safetensors")
tts.ema_model.load_state_dict(state_dict, strict=False)

# Generate — YOUR voice reference + vintage delivery
wav, sr, _ = tts.infer(
    ref_file="your_voice_reference.wav",   # Any modern voice
    ref_text="",                            # Auto-transcribe
    gen_text="Good evening, ladies and gentlemen.",
    speed=0.9,                              # Measured transatlantic pace
)
```

## Key Innovation

F5-TTS separates **voice identity** (from reference audio) from **speech style** (from training).
- Fine-tuning teaches transatlantic speech patterns
- Reference audio provides the speaker's voice/timbre
- Result: Anyone's voice + vintage delivery

## Model Details

| Spec | Value |
|------|-------|
| Base Model | F5-TTS v1 (337M params) |
| Architecture | Flow-matching DiT |
| Training Data | 164 hours, 44,345 segments |
| Sample Rate | 24kHz |
| Training Hardware | 2x Tesla V100 32GB |
| Training Time | ~7 days |
| License | MIT |

## Limitations

- Best results with clear, single-speaker reference audio (5-15 seconds)
- Edison preset has lower quality due to source material limitations
- Model learns speech patterns, not specific historical voices
- Some segments may contain background music from radio dramas

## Citation

```bibtex
@misc{vintagevoice2026,
  title={VintageVoice: Open-Source TTS for Historical Speech Patterns},
  author={Elyan Labs},
  year={2026},
  url={https://github.com/Scottcjn/vintage-voice}
}
```

## Built By

[Elyan Labs](https://elyanlabs.ai) — Where vintage hardware meets cutting-edge AI.

Built on a $69 refurb hard drive with pawn shop GPUs. Proof that world-class AI doesn't require world-class budgets.
