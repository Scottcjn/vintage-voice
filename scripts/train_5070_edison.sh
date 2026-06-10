#!/bin/bash
# SPDX-License-Identifier: MIT
# VintageVoice — Edison Preset Training on RTX 5070
# Machine: .106 (sophia5070node, RTX 5070 12GB)
#
# Trains on same 44K dataset but with smaller batch size for 12GB VRAM
# The Edison character comes from the training data itself (oldest recordings)

set -e
BASE=/home/sophia5070node/vintage-voice
SEGMENTS=$BASE/segments
F5_DATASET=$BASE/f5_dataset
CKPT_DIR=$BASE/ckpts

echo "=========================================="
echo "  VintageVoice Edison — RTX 5070 Training"
echo "  $(date)"
echo "=========================================="

# Wait for rsync to finish
while pgrep -f "rsync.*segments" > /dev/null 2>&1; do
    COUNT=$(find $SEGMENTS -name "*.wav" 2>/dev/null | wc -l)
    echo "Waiting for segment sync... $COUNT files so far"
    sleep 30
done

TOTAL=$(find $SEGMENTS -name "*.wav" | wc -l)
echo "Segments ready: $TOTAL"

# Rebuild manifest with local paths
echo "Building local manifest..."
python3 -c "
import soundfile as sf
import os, glob

segments = sorted(glob.glob('$SEGMENTS/*.wav'))
print(f'Building manifest for {len(segments)} segments...')

with open('$BASE/manifest_local.csv', 'w') as f:
    f.write('path|duration|source\n')
    good = 0
    for seg in segments:
        try:
            info = sf.info(seg)
            if info.duration >= 2.0:
                f.write(f'{seg}|{info.duration:.2f}|{os.path.basename(seg)}\n')
                good += 1
        except:
            pass
    print(f'Valid segments: {good}')
"

# Transcribe locally with faster-whisper on 5070
echo ""
echo "=== Transcribing with faster-whisper on RTX 5070 ==="
mkdir -p $BASE/transcriptions
python3 /home/sophia5070node/vintage-voice/transcribe_simple.py \
    --manifest $BASE/manifest_local.csv \
    --output $BASE/transcriptions \
    --device cuda:0

# Build F5-TTS CSV
echo ""
echo "=== Building F5-TTS CSV ==="
python3 /home/sophia5070node/vintage-voice/build_f5_csv.py \
    $BASE/transcriptions \
    $BASE/f5_train_local.csv

# Prepare Arrow dataset
echo ""
echo "=== Preparing Arrow dataset ==="

# Need the vocab file
mkdir -p /home/sophia5070node/.local/lib/python3.12/data/Emilia_ZH_EN_pinyin/
cp $F5_DATASET/vocab.txt /home/sophia5070node/.local/lib/python3.12/data/Emilia_ZH_EN_pinyin/vocab.txt

python3 -m f5_tts.train.datasets.prepare_csv_wavs \
    $BASE/f5_train_local.csv \
    $BASE/f5_edison_dataset

# Symlink for F5-TTS to find it
mkdir -p /home/sophia5070node/.local/lib/python3.12/data/
ln -sf $BASE/f5_edison_dataset /home/sophia5070node/.local/lib/python3.12/data/f5_edison_dataset_custom

# Train with smaller batch for 12GB VRAM
echo ""
echo "=== Training F5-TTS Edison on RTX 5070 ==="
echo "  Batch: 1600 frames (half of V100 setting for 12GB)"
echo "  Max samples: 32"

python3 -m f5_tts.train.finetune_cli \
    --exp_name F5TTS_v1_Base \
    --dataset_name f5_edison_dataset \
    --learning_rate 1e-5 \
    --batch_size_per_gpu 1600 \
    --batch_size_type frame \
    --max_samples 32 \
    --grad_accumulation_steps 2 \
    --max_grad_norm 1.0 \
    --epochs 50 \
    --num_warmup_updates 200 \
    --save_per_updates 1000 \
    --last_per_updates 500 \
    --keep_last_n_checkpoints 3 \
    --finetune \
    --tokenizer custom \
    --tokenizer_path $BASE/f5_edison_dataset/vocab.txt \
    --log_samples

echo "Edison training complete! $(date)"
