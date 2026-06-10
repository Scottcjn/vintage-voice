#!/bin/bash
# SPDX-License-Identifier: MIT
# VintageVoice — Full Training Pipeline
# Run on .136 (2x V100 32GB)
#
# Usage: bash run_pipeline.sh [step]
#   step 1: Preprocess raw audio → segments
#   step 2: Transcribe segments with Whisper
#   step 3: Prepare F5-TTS dataset (CSV → Arrow)
#   step 4: Fine-tune F5-TTS
#   step 5: Test generation

set -e

BASE=/mnt/18tb
RAW=$BASE/vintage_voice
PROCESSED=$BASE/vintage_voice_processed
SEGMENTS=$PROCESSED/segments
TRANSCRIPTIONS=$PROCESSED/transcriptions
F5_DATASET=$BASE/vintage_voice_f5_dataset
MODEL_OUT=$BASE/models/vintage-voice
STEP=${1:-all}

echo "=========================================="
echo "  VintageVoice Training Pipeline"
echo "  $(date)"
echo "=========================================="

# Step 1: Preprocess
if [[ "$STEP" == "1" || "$STEP" == "all" ]]; then
    echo ""
    echo "=== STEP 1: Preprocessing raw audio ==="
    python3 $BASE/preprocess.py \
        --input $RAW \
        --output $PROCESSED \
        --workers 8
    echo "Segments created: $(find $SEGMENTS -name '*.wav' | wc -l)"
fi

# Step 2: Transcribe with Whisper on GPU
if [[ "$STEP" == "2" || "$STEP" == "all" ]]; then
    echo ""
    echo "=== STEP 2: Transcribing with Whisper (GPU) ==="
    python3 $BASE/transcribe_whisper.py \
        --manifest $PROCESSED/manifest.csv \
        --output $TRANSCRIPTIONS \
        --device cuda:1
    echo "Transcriptions: $(find $TRANSCRIPTIONS -name '*.json' | wc -l)"
fi

# Step 3: Prepare F5-TTS dataset format
if [[ "$STEP" == "3" || "$STEP" == "all" ]]; then
    echo ""
    echo "=== STEP 3: Preparing F5-TTS dataset ==="

    # Create the CSV in F5-TTS format: audio_file|text
    TRAIN_CSV=$PROCESSED/f5_train.csv
    echo "audio_file|text" > $TRAIN_CSV

    # Read transcriptions and build CSV
    python3 -c "
import json, os, glob
csv_lines = []
for jf in sorted(glob.glob('$TRANSCRIPTIONS/*.json')):
    with open(jf) as f:
        data = json.load(f)
    audio = data.get('audio_path', '')
    text = data.get('text', '').strip()
    if audio and text and os.path.exists(audio):
        # Clean text: remove pipes, normalize whitespace
        text = text.replace('|', ' ').replace('\n', ' ').strip()
        if len(text) > 5:  # Skip very short transcriptions
            csv_lines.append(f'{audio}|{text}')
with open('$TRAIN_CSV', 'a') as f:
    f.write('\n'.join(csv_lines) + '\n')
print(f'Training samples: {len(csv_lines)}')
"

    # Run F5-TTS data preparation (CSV → Arrow + vocab)
    python3 -m f5_tts.train.datasets.prepare_csv_wavs \
        $TRAIN_CSV \
        $F5_DATASET

    echo "F5 dataset prepared at $F5_DATASET"
fi

# Step 4: Fine-tune F5-TTS
if [[ "$STEP" == "4" || "$STEP" == "all" ]]; then
    echo ""
    echo "=== STEP 4: Fine-tuning F5-TTS ==="

    mkdir -p $MODEL_OUT

    python3 -m f5_tts.train.finetune_cli \
        --exp_name F5TTS_v1_Base \
        --dataset_name vintage_voice_f5_dataset \
        --learning_rate 1e-5 \
        --batch_size_per_gpu 3200 \
        --batch_size_type frame \
        --max_samples 64 \
        --grad_accumulation_steps 1 \
        --max_grad_norm 1.0 \
        --epochs 100 \
        --num_warmup_updates 200 \
        --save_per_updates 1000 \
        --last_per_updates 500 \
        --keep_last_n_checkpoints 3 \
        --finetune \
        --tokenizer custom \
        --tokenizer_path $F5_DATASET/vocab.txt \
        --log_samples

    echo "Fine-tuning complete!"
fi

# Step 5: Test generation
if [[ "$STEP" == "5" || "$STEP" == "all" ]]; then
    echo ""
    echo "=== STEP 5: Test Generation ==="

    # Find best checkpoint
    BEST_CKPT=$(ls -t $HOME/.cache/f5_tts/ckpts/vintage_voice_f5_dataset/model_*.safetensors 2>/dev/null | head -1)
    if [ -z "$BEST_CKPT" ]; then
        BEST_CKPT=$(ls -t $HOME/.cache/f5_tts/ckpts/vintage_voice_f5_dataset/model_*.pt 2>/dev/null | head -1)
    fi

    if [ -n "$BEST_CKPT" ]; then
        echo "Using checkpoint: $BEST_CKPT"

        # Find a reference audio from our training data
        REF_AUDIO=$(find $SEGMENTS -name "*.wav" -size +100k | head -1)

        # Generate test samples
        for text in \
            "One simply must attest ones hardware before the epoch settles, dahling." \
            "Good evening, ladies and gentlemen. This is your announcer speaking from the studios of Elyan Labs." \
            "And now, from the laboratories of the future, a breakthrough that will change computing forever."; do

            echo "  Generating: ${text:0:60}..."
            python3 -m f5_tts.infer.infer_cli \
                --model F5TTS_v1_Base \
                --ckpt_file "$BEST_CKPT" \
                --vocab_file $F5_DATASET/vocab.txt \
                --ref_audio "$REF_AUDIO" \
                --ref_text "" \
                --gen_text "$text" \
                --output_dir $MODEL_OUT/samples/ \
                2>/dev/null
        done

        echo "Samples saved to $MODEL_OUT/samples/"
    else
        echo "No checkpoint found yet — run step 4 first"
    fi
fi

echo ""
echo "=========================================="
echo "  Pipeline complete! $(date)"
echo "=========================================="
