# Contributing to VintageVoice

VintageVoice is an open-source TTS model for historical speech patterns — transatlantic accent, newsreel narrators, Edison cylinders, and lost voices of the 1880s-1960s.

## Project Structure

```
vintage-voice/
├── configs/          # Voice style configurations
├── data/            # Training data manifests
├── models/          # Model weights and architecture definitions
├── scripts/         # Training and inference scripts
├── test_videos/     # Test output samples
├── FAMILY_RECORDING_GUIDE.md
└── WEBSITE_BRIEF.md
```

## Getting Started

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (for training; inference works on CPU)
- FFmpeg (for audio processing)

### Setup

```bash
git clone https://github.com/Scottcjn/vintage-voice.git
cd vintage-voice
```

### Running Inference

```bash
# Generate a newsreel-style narration
python scripts/inference.py --config configs/newsreel.yaml --text "The year is 1945..."

# Generate Edison cylinder style
python scripts/inference.py --config configs/edison.yaml --text "Good evening..."
```

## Voice Styles

Each style has a corresponding config in `configs/`:

- **Newsreel** — 1940s radio news delivery, authoritative, measured pace
- **Transatlantic** — Early 20th century upper-class American with British influence
- **Edison Cylinder** — Authentic 1890s-1910s recorded sound quality
- **Newsreel Dark** — 1930s cinema newsreel, dramatic lighting and delivery

## Adding a New Voice Style

1. **Create a config** in `configs/<style-name>.yaml`
2. **Record reference samples** using FAMILY_RECORDING_GUIDE.md guidelines
3. **Add model weights** manifest to `models/`
4. **Update this CONTRIBUTING.md** with the new style
5. **Test inference** and verify output quality

## Audio Quality Guidelines

- Sample rate: 22050 Hz (authentic to the era)
- Format: WAV (Edison cylinders), MP3 (modern listening)
- Keep recordings under 3 minutes for test samples
- Historical accuracy over modern clarity

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
