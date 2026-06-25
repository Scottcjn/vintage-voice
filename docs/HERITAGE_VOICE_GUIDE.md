# Clone Your Heritage Voice in About an Hour

A step-by-step guide for preserving heritage voices using AI voice cloning.

## Overview

This guide shows you how to clone a heritage voice (grandparent, elder, traditional speaker) in about an hour using an 8GB GPU. The cloned voice can be used to preserve language, create educational materials, or keep family memories alive.

## Prerequisites

### Hardware
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3070, RTX 4060 Ti, etc.)
- **RAM**: 16GB+ system memory
- **Storage**: 10GB+ free space

### Software
- Python 3.10+
- CUDA 11.8+
- Git

## Quick Start

### Step 1: Install Dependencies

```bash
# Clone the repository
git clone https://github.com/Scottcjn/vintage-voice.git
cd vintage-voice

# Install Python dependencies
pip install -r requirements.txt

# Install F5-TTS
pip install f5-tts
```

### Step 2: Prepare Reference Audio

**Requirements for good results:**
- **Duration**: 30-60 seconds of speech
- **Quality**: Clean, noise-free audio
- **Content**: Varied sentences, natural speech
- **Format**: WAV, MP3, or FLAC

**Tips for recording:**
```bash
# Record in a quiet room
# Use a good microphone if possible
# Include different emotions and tones
# Speak clearly but naturally
```

### Step 3: Run the Demo

```bash
# Basic usage
python clone_heritage_voice.py \
    --input reference_audio.wav \
    --output cloned_voice.wav

# With custom text
python clone_heritage_voice.py \
    --input grandma.wav \
    --output grandma_clone.wav \
    --text "Hello dear, this is grandma's voice preserved forever"

# Quick mode (faster, lower quality)
python clone_heritage_voice.py \
    --input grandpa.wav \
    --output grandpa_clone.wav \
    --epochs 5
```

### Step 4: Listen and Share

```bash
# Play the cloned voice
afplay cloned_voice.wav  # macOS
aplay cloned_voice.wav   # Linux
start cloned_voice.wav   # Windows
```

## Detailed Walkthrough

### 1. Recording Reference Audio

**Best practices:**
- Record in a quiet environment
- Use a condenser microphone if available
- Keep the speaker 6-12 inches from the mic
- Record at least 30 seconds of speech
- Include varied sentences and emotions

**Example script for the speaker:**
```
"Hello, my name is [Name]. I am [Age] years old. 
I grew up in [Place] and learned to speak [Language] from my parents.
Today I want to share some memories from my childhood.
[Continue with personal stories...]"
```

### 2. Audio Preprocessing

The script automatically:
- Converts stereo to mono
- Resamples to 24kHz
- Trims silence
- Normalizes volume

### 3. Fine-Tuning Process

**What happens during training:**
1. Load F5-TTS base model (~2GB VRAM)
2. Prepare training data from reference audio
3. Fine-tune for 10 epochs (~30-45 minutes)
4. Save checkpoint to output directory

**Memory usage on 8GB GPU:**
- Model: ~2GB
- Training data: ~1GB
- Batch processing: ~3GB
- **Total: ~6GB** (fits in 8GB)

### 4. Voice Generation

**What happens during generation:**
1. Load fine-tuned model
2. Process input text
3. Generate mel spectrogram
4. Convert to audio waveform
5. Save output file

## Troubleshooting

### Out of Memory Error
```bash
# Reduce batch size
python clone_heritage_voice.py --input audio.wav --output out.wav --batch-size 1

# Use gradient accumulation
python clone_heritage_voice.py --input audio.wav --output out.wav --gradient-accumulation 4

# Reduce audio length
# Trim reference audio to 30 seconds
```

### Poor Quality Output
```bash
# Increase training epochs
python clone_heritage_voice.py --input audio.wav --output out.wav --epochs 20

# Use better reference audio
# - Cleaner recording
# - More varied speech
# - 60+ seconds duration
```

### Slow Training
```bash
# Use mixed precision (default: enabled)
python clone_heritage_voice.py --input audio.wav --output out.wav --mixed-precision

# Reduce max duration
python clone_heritage_voice.py --input audio.wav --output out.wav --max-duration 10
```

## Community Use Cases

### Language Preservation
- Record elder speakers of endangered languages
- Create voice models for language learning apps
- Generate pronunciation guides

### Family Heritage
- Preserve grandparent voices for future generations
- Create family history recordings
- Share memories across time zones

### Education
- Create historical figure voice models
- Generate language lesson audio
- Build interactive learning tools

## Sharing Your Results

### Upload to GitHub
```bash
# Fork the repository
# Add your demo script
# Submit a pull request
```

### Community Discord
Join the vintage-voice Discord to share your results and get help.

### Social Media
Share your cloned voice with #HeritageVoice #VoiceCloning #LanguagePreservation

## Advanced Usage

### Custom Model Training
```bash
# Use the full training script
python scripts/train_f5.py \
    --manifest train.csv \
    --base-model model.safetensors \
    --epochs 50 \
    --batch-size 2 \
    --gradient-accumulation 4
```

### Multi-Speaker Cloning
```bash
# Train on multiple speakers
python scripts/train_f5.py \
    --manifest multi_speaker.csv \
    --preset transatlantic \
    --epochs 100
```

### Production Deployment
```bash
# Export model for production
python scripts/export_model.py \
    --checkpoint output/checkpoint_epoch_50.pt \
    --output model_export
```

## Resources

- [F5-TTS Documentation](https://github.com/SWivid/F5-TTS)
- [Vintage Voice GitHub](https://github.com/Scottcjn/vintage-voice)
- [Language Preservation Guide](docs/ENDANGERED_LANGUAGES.md)

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions welcome! See CONTRIBUTING.md for guidelines.

## Support

- GitHub Issues: https://github.com/Scottcjn/vintage-voice/issues
- Discord: [Join community]
- Email: [Contact maintainer]

---

**Preserve your heritage. Clone your voice. Share with the world.**
