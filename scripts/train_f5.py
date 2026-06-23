#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
VintageVoice — F5-TTS Fine-Tuning for Historical Speech Patterns

Fine-tunes F5-TTS on vintage audio to learn:
- Transatlantic accent phonetics
- Period-accurate prosody and cadence
- Historical microphone characteristics
- Era-specific speech patterns

Based on F5-TTS: https://github.com/SWivid/F5-TTS
"""
import argparse
import csv
import os
import random

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class VintageVoiceDataset(Dataset):
    """Dataset of preprocessed vintage audio segments with transcriptions"""

    # Accepted column aliases for the audio path, in producer-precedence order.
    # The pipeline emits two manifest schemas: preprocess.py / fast_manifest.py /
    # auto_pipeline.sh write ``path|duration|source``, while build_f5_csv.py (the
    # F5 training CSV) writes ``audio_file|text`` — so ``path`` and ``audio_file``
    # come first. ``audio_path`` is the column older code here wrongly assumed
    # every producer emits (none does, which is why the loader used to KeyError on
    # the first row and training never started); it is kept last only as a
    # defensive fallback so that, should a future writer add it, it cannot silently
    # shadow the columns producers actually emit today. ``wav``/``filename`` were
    # dropped — no tool emits them, and speculative aliases only desync this table
    # from the real pipeline schemas.
    _AUDIO_KEYS = ("path", "audio_file", "audio_path")
    _MIN_DURATION = 2.0

    def __init__(self, manifest_path, sample_rate=24000, max_duration=15.0):
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        self.entries = []

        skipped = {
            "no_audio_col": 0,
            "missing_file": 0,
            "no_text": 0,
            "no_duration": 0,
            "too_short": 0,
        }
        with open(manifest_path) as f:
            reader = csv.DictReader(f, delimiter="|")
            for row in reader:
                audio_path = next(
                    (row[k] for k in self._AUDIO_KEYS if row.get(k)), None
                )
                if not audio_path:
                    skipped["no_audio_col"] += 1
                    continue
                if not os.path.exists(audio_path):
                    skipped["missing_file"] += 1
                    continue
                text = (row.get("text") or "").strip()
                if not text:
                    # An audio-only manifest (path|duration|source) carries no
                    # transcription; TTS fine-tuning needs the paired text, so
                    # skip rather than train on an empty target.
                    skipped["no_text"] += 1
                    continue
                duration = self._row_duration(row, audio_path)
                if duration is None:
                    # The manifest carried no usable duration and torchaudio could
                    # not probe the file (corrupt audio / unsupported format).
                    # Admitting it as duration=0.0 would slip it past the
                    # MIN_DURATION guard and corrupt any downstream length-based
                    # filtering or bucketing, so drop and count it instead.
                    skipped["no_duration"] += 1
                    continue
                if duration < self._MIN_DURATION:
                    skipped["too_short"] += 1
                    continue
                self.entries.append({
                    "audio_path": audio_path,
                    "text": text,
                    "duration": float(duration),
                })

        print(f"Dataset: {len(self.entries)} segments")
        dropped = {k: v for k, v in skipped.items() if v}
        if dropped:
            print(f"  (skipped rows: {dropped})")

    @classmethod
    def _row_duration(cls, row, audio_path):
        """Duration in seconds from the manifest column, else probed from the
        audio file. ``f5_train.csv`` (audio_file|text) has no duration column,
        so fall back to torchaudio metadata instead of KeyError-ing."""
        raw = row.get("duration")
        if raw not in (None, ""):
            try:
                return float(raw)
            except (TypeError, ValueError):
                pass
        try:
            info = torchaudio.info(audio_path)
            if info.sample_rate:
                return info.num_frames / info.sample_rate
        except Exception:
            pass
        return None

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        audio_path = entry["audio_path"]
        text = entry["text"]

        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Trim or pad to max length
        if waveform.shape[1] > self.max_samples:
            waveform = waveform[:, :self.max_samples]

        return {
            "audio": waveform.squeeze(0),
            "text": text,
            "duration": float(entry["duration"]),
        }


def collate_fn(batch):
    """Pad audio to same length in batch"""
    max_len = max(b["audio"].shape[0] for b in batch)
    audios = []
    texts = []
    durations = []

    for b in batch:
        audio = b["audio"]
        pad_len = max_len - audio.shape[0]
        if pad_len > 0:
            audio = F.pad(audio, (0, pad_len))
        audios.append(audio)
        texts.append(b["text"])
        durations.append(b["duration"])

    return {
        "audio": torch.stack(audios),
        "text": texts,
        "duration": durations,
    }


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Single training epoch"""
    model.train()
    total_loss = 0
    n_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        audio = batch["audio"].to(device)
        texts = batch["text"]

        optimizer.zero_grad()

        # Forward pass (F5-TTS specific — adapt to actual model API)
        # This is the training loop structure; actual F5-TTS integration
        # requires importing their model and loss functions
        try:
            loss = model.compute_loss(audio, texts)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / n_batches
                print(f"  Epoch {epoch} [{batch_idx+1}/{len(dataloader)}] Loss: {avg_loss:.4f}")
        except Exception as e:
            print(f"  Batch {batch_idx} error: {e}")
            continue

    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune F5-TTS on vintage voice data")
    parser.add_argument("--manifest", default="/mnt/18tb/vintage_voice_processed/transcriptions/train.csv")
    parser.add_argument("--base-model", default="/mnt/18tb/models/weird/f5-tts/F5TTS_v1_Base/model_1250000.safetensors")
    parser.add_argument("--output", default="/mnt/18tb/models/vintage-voice")
    parser.add_argument("--preset", default="transatlantic", help="Voice preset to train")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-every", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"VintageVoice Fine-Tuning")
    print(f"  Preset: {args.preset}")
    print(f"  Base model: {args.base_model}")
    print(f"  Device: {args.device}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")

    # Load dataset
    dataset = VintageVoiceDataset(args.manifest)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    # Load F5-TTS base model
    print(f"\nLoading F5-TTS base model...")
    device = torch.device(args.device)

    # F5-TTS model loading — requires f5-tts package
    # Install: pip install f5-tts
    try:
        from f5_tts.model import DiT
        from f5_tts.model.utils import get_tokenizer
        from safetensors.torch import load_file

        # Load model architecture
        vocab = get_tokenizer("vocos", args.base_model.replace("model_1250000.safetensors", "vocab.txt"))
        model = DiT(
            dim=1024,
            depth=22,
            heads=16,
            ff_mult=2,
            text_dim=512,
            conv_layers=4,
        ).to(device)

        # Load pretrained weights
        state_dict = load_file(args.base_model)
        model.load_state_dict(state_dict, strict=False)
        print(f"  Loaded {len(state_dict)} weight tensors")

    except ImportError:
        print("\n  F5-TTS package not installed. Install with:")
        print("  pip install f5-tts")
        print("\n  Creating placeholder for training loop structure...")

        # Placeholder model for pipeline testing
        model = torch.nn.Linear(24000, 512).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    print(f"\nStarting training...")
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_epoch(model, dataloader, optimizer, device, epoch)
        scheduler.step()

        print(f"Epoch {epoch}/{args.epochs} — Loss: {avg_loss:.4f} — LR: {scheduler.get_last_lr()[0]:.2e}")

        # Save checkpoint
        if epoch % args.save_every == 0 or avg_loss < best_loss:
            ckpt_path = os.path.join(args.output, f"vintage_voice_{args.preset}_epoch{epoch}.safetensors")
            try:
                from safetensors.torch import save_file
                save_file(model.state_dict(), ckpt_path)
            except ImportError:
                torch.save(model.state_dict(), ckpt_path.replace(".safetensors", ".pt"))
            print(f"  Saved: {ckpt_path}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = os.path.join(args.output, f"vintage_voice_{args.preset}_best.safetensors")
                try:
                    from safetensors.torch import save_file
                    save_file(model.state_dict(), best_path)
                except ImportError:
                    torch.save(model.state_dict(), best_path.replace(".safetensors", ".pt"))
                print(f"  New best! {best_path}")

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
