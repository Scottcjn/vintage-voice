#!/bin/bash
# VintageVoice — Train NOW on 37K snapshot
# Runs on cuda:0 while preprocessing continues on CPU
#
# Track 1 (this script): Transcribe 37K → build dataset → train on cuda:0
# Track 2 (auto_pipeline): Preprocessing keeps running → full dataset later

set -e
BASE=/mnt/18tb
PROCESSED=$BASE/vintage_voice_processed
MANIFEST=$PROCESSED/manifest_37k.csv
TRANSCRIPTIONS=$PROCESSED/transcriptions_37k
F5_DATASET=$BASE/vintage_voice_f5_37k
TRAIN_CSV=$PROCESSED/f5_train_37k.csv

echo "=========================================="
echo "  VintageVoice — Train on 37K Snapshot"
echo "  $(date)"
echo "  GPU: cuda:0 (V100 #1)"
echo "=========================================="

# Wait for manifest to be ready
while [ ! -f "$MANIFEST" ] || [ "$(wc -l < $MANIFEST)" -lt 1000 ]; do
    echo "Waiting for manifest... $(wc -l < $MANIFEST 2>/dev/null || echo 0) entries"
    sleep 10
done
ENTRIES=$(wc -l < $MANIFEST)
echo "Manifest ready: $ENTRIES entries"

# Step 1: Transcribe with Whisper on cuda:0
echo ""
echo "=== STEP 1: Whisper Transcription (cuda:0) ==="
mkdir -p $TRANSCRIPTIONS
python3 $BASE/transcribe_simple.py \
    --manifest $MANIFEST \
    --output $TRANSCRIPTIONS \
    --device cuda:0

TRANS=$(find $TRANSCRIPTIONS -name "*.json" | wc -l)
echo "Transcribed: $TRANS segments"

# Step 2: Build F5-TTS CSV
echo ""
echo "=== STEP 2: Building F5-TTS dataset ==="
python3 $BASE/build_f5_csv.py $TRANSCRIPTIONS $TRAIN_CSV

SAMPLES=$(wc -l < $TRAIN_CSV)
echo "Training samples: $SAMPLES"

# Step 3: Prepare Arrow dataset
echo ""
echo "=== STEP 3: Preparing Arrow dataset ==="
python3 -m f5_tts.train.datasets.prepare_csv_wavs \
    $TRAIN_CSV \
    $F5_DATASET

echo "Dataset ready at $F5_DATASET"

# Step 4: Fine-tune F5-TTS on cuda:0
echo ""
echo "=== STEP 4: Fine-tuning F5-TTS ==="
echo "  Model: F5TTS_v1_Base (337M params)"
echo "  Dataset: 37K segments"
echo "  GPU: cuda:0"

CUDA_VISIBLE_DEVICES=0 python3 -m f5_tts.train.finetune_cli \
    --exp_name F5TTS_v1_Base \
    --dataset_name vintage_voice_f5_37k \
    --learning_rate 1e-5 \
    --batch_size_per_gpu 3200 \
    --batch_size_type frame \
    --max_samples 64 \
    --grad_accumulation_steps 1 \
    --max_grad_norm 1.0 \
    --epochs 50 \
    --num_warmup_updates 200 \
    --save_per_updates 1000 \
    --last_per_updates 500 \
    --keep_last_n_checkpoints 3 \
    --finetune \
    --tokenizer custom \
    --tokenizer_path $F5_DATASET/vocab.txt \
    --log_samples

echo ""
echo "=========================================="
echo "  Training complete! $(date)"
echo "  Test with: python3 /mnt/18tb/generate_sophia.py --all-test"
echo "=========================================="
