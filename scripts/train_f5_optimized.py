#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
VintageVoice - F5-TTS Fine-Tuning for Historical Speech Patterns (8GB Optimized)

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

    def __init__(self, manifest_path, sample_rate=24000, max_duration=10.0):  # Reduced from 15.0
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        self.entries = []

        with open(manifest_path) as f:
            reader = csv.DictReader(f, delimiter="|")
            for row in reader:
                if os.path.exists(row["audio_path"]) and float(row["duration"]) >= 2.0:
                    self.entries.append(row)

        print(f"Dataset: {len(self.entries)} segments")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        audio_path = entry["audio_path"]
        text = entry["text"]

        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample if needed
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        
        # Truncate if too long
        if waveform.shape[1] > self.max_samples:
            waveform = waveform[:, :self.max_samples]
        
        return {
            "audio": waveform.squeeze(0),
            "text": text,
            "duration": waveform.shape[1] / self.sample_rate
        }


def collate_fn(batch):
    """Collate function with padding"""
    # Find max length in batch
    max_len = max(b["audio"].shape[0] for b in batch)
    
    # Pad audio
    audios = []
    texts = []
    for b in batch:
        audio = b["audio"]
        padding = max_len - audio.shape[0]
        if padding > 0:
            audio = F.pad(audio, (0, padding))
        audios.append(audio)
        texts.append(b["text"])
    
    return {
        "audio": torch.stack(audios),
        "text": texts
    }


def train_epoch(model, dataloader, optimizer, device, epoch, scaler=None, gradient_accumulation_steps=4):
    """Single training epoch with mixed precision and gradient accumulation"""
    model.train()
    total_loss = 0
    n_batches = 0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        audio = batch["audio"].to(device)
        texts = batch["text"]

        # Mixed precision training
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            try:
                loss = model.compute_loss(audio, texts)
                loss = loss / gradient_accumulation_steps  # Scale loss for accumulation
            except Exception as e:
                print(f"  Batch {batch_idx} error: {e}")
                continue

        # Backward pass with scaler
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient accumulation
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * gradient_accumulation_steps
        n_batches += 1

        if (batch_idx + 1) % 10 == 0:
            avg_loss = total_loss / n_batches
            print(f"  Epoch {epoch} [{batch_idx+1}/{len(dataloader)}] Loss: {avg_loss:.4f}")

    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune F5-TTS on vintage voice data (8GB optimized)")
    parser.add_argument("--manifest", default="/mnt/18tb/vintage_voice_processed/transcriptions/train.csv")
    parser.add_argument("--base-model", default="/mnt/18tb/models/weird/f5-tts/F5TTS_v1_Base/model_1250000.safetensors")
    parser.add_argument("--output", default="/mnt/18tb/models/vintage-voice")
    parser.add_argument("--preset", default="transatlantic", help="Voice preset to train")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)  # Reduced from 8
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--gradient-accumulation", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--mixed-precision", action="store_true", default=True, help="Use mixed precision training")
    parser.add_argument("--max-duration", type=float, default=10.0, help="Max audio duration in seconds")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of data loading workers")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"VintageVoice Fine-Tuning (8GB Optimized)")
    print(f"  Preset: {args.preset}")
    print(f"  Base model: {args.base_model}")
    print(f"  Device: {args.device}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation}")
    print(f"  Mixed precision: {args.mixed_precision}")
    print(f"  Max duration: {args.max_duration}s")
    print(f"  Learning rate: {args.lr}")

    # Load dataset
    dataset = VintageVoiceDataset(args.manifest, max_duration=args.max_duration)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,  # Reduced from 4
        pin_memory=True,
    )

    # Load F5-TTS base model
    print(f"\nLoading F5-TTS base model...")
    device = torch.device(args.device)

    # Initialize mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision and device.type == 'cuda' else None

    # F5-TTS model loading - requires f5-tts package
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

        # Enable gradient checkpointing if available
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            print("  Gradient checkpointing enabled")

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
        avg_loss = train_epoch(
            model, dataloader, optimizer, device, epoch,
            scaler=scaler,
            gradient_accumulation_steps=args.gradient_accumulation
        )
        scheduler.step()

        print(f"Epoch {epoch}/{args.epochs} - Loss: {avg_loss:.4f} - LR: {scheduler.get_last_lr()[0]:.2e}")

        # Save checkpoint
        if epoch % args.save_every == 0 or avg_loss < best_loss:
            if avg_loss < best_loss:
                best_loss = avg_loss
            
            checkpoint_path = os.path.join(args.output, f"checkpoint_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")

    print(f"\nTraining complete! Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
