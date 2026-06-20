# Clone Your Heritage Voice in an Hour

Welcome! This guide is designed for community language workers, linguists, and archivists who want to preserve and synthesize endangered or heritage languages. 

Using just an **8 GB consumer GPU** (like an RTX 4060 or 4070) and roughly **1 hour of clean audio**, you can finetune a text-to-speech model (CosyVoice2-0.5B) to speak in a native heritage accent.

> [!WARNING]
> **Data Sovereignty First**: Always ensure you have explicit, informed consent from the speaker or the community before cloning a voice. Do not scrape or use rights-encumbered audio. Please read our [Endangered Languages Data Policy](ENDANGERED_LANGUAGES.md) before proceeding.

## 1. What You Need
- A machine with **Linux or Windows (WSL2)**
- An NVIDIA GPU with at least **8 GB VRAM**
- Python 3.12+ and `torch` (version 2.3.1 recommended)
- Roughly **1 to 8 hours** of clean, conversational audio of your target language/accent.

## 2. Preparing Your Audio

The golden rule of voice cloning: **Garbage in, garbage out**. 
Long-form, single-dialect recordings (like podcasts, local radio interviews, or family oral history tapes) yield the best results.

### Step 2a: Gather the Audio
Put all your `.mp3` or `.wav` files into a folder (e.g., `data/raw_audio`).

### Step 2b: Process and Transcribe
We provide a unified script that handles the tedious work of splitting your long audio into short sentences, removing long silences, and transcribing it using OpenAI's Whisper model. 

Run the automated script:
```bash
python scripts/clone_heritage_voice.py \
    --audio_dir data/raw_audio \
    --language "fr" \
    --output_dir data/processed
```
*(Replace `"fr"` with the ISO code of the language closest to your heritage language. Whisper will use this to generate the text transcripts).*

## 3. Finetuning (The 8 GB Magic)

Training AI models usually requires massive server GPUs. However, we've optimized the CosyVoice2 recipe to fit entirely inside **5.1 GB of VRAM**. We achieve this by:
- Using 8-bit Adam optimizer (`AdamW8bit`).
- Disabling deepspeed (which causes issues on 8GB cards).
- Accumulating gradients (`accum_grad: 1`).
- Using precise checkpointing.

To start the finetuning process, simply run:
```bash
python scripts/clone_heritage_voice.py --train --config configs/cosyvoice2_8gb.yaml
```

> [!TIP]
> The model will save a checkpoint every epoch. For a small dataset (1-2 hours), the model usually sounds best around **Epoch 2 or 3**. If you train it too long (Epoch 6+), it might overfit and sound robotic.

## 4. Synthesize Your Heritage Voice

Once the training is done, you can test your new voice profile!
The script automatically symlinks the best checkpoint so you can use it immediately.

```bash
python scripts/clone_heritage_voice.py --synthesize "Hello, this is my heritage voice speaking!"
```
Check the `data/output` folder for your generated `.wav` file. 

## Next Steps
- Try finetuning the flow model for even better acoustic texture.
- Share your `epoch_N_whole.pt` weights with your community!
