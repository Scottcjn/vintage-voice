#!/bin/bash
# SPDX-License-Identifier: MIT
# Auto-pipeline: waits for preprocessing, then runs steps 2-3
BASE=/mnt/18tb
PROCESSED=$BASE/vintage_voice_processed

echo "[$(date)] Waiting for preprocessing to finish..."

# Wait for preprocess.py to exit
while pgrep -f "preprocess.py" > /dev/null; do
    SEGS=$(find $PROCESSED/segments -name "*.wav" 2>/dev/null | wc -l)
    echo "[$(date)] Preprocessing... $SEGS segments so far"
    sleep 30
done

echo "[$(date)] Preprocessing complete!"
TOTAL_SEGS=$(find $PROCESSED/segments -name "*.wav" | wc -l)
echo "Total segments: $TOTAL_SEGS"

# Check if manifest exists, if not create it from segments
if [ ! -f "$PROCESSED/manifest.csv" ]; then
    echo "[$(date)] Creating manifest from segments..."
    echo "path|duration|source" > $PROCESSED/manifest.csv
    for f in $PROCESSED/segments/*.wav; do
        dur=$(ffprobe -v quiet -show_entries format=duration -of csv=p=0 "$f" 2>/dev/null)
        if [ -n "$dur" ]; then
            echo "$f|$dur|$(basename $f .wav)" >> $PROCESSED/manifest.csv
        fi
    done
    ENTRIES=$(wc -l < $PROCESSED/manifest.csv)
    echo "Manifest entries: $ENTRIES"
fi

# Step 2: Transcribe with Whisper on GPU 1
echo ""
echo "[$(date)] === STEP 2: Transcribing with Whisper on cuda:1 ==="
mkdir -p $PROCESSED/transcriptions
python3 $BASE/transcribe_whisper.py \
    --manifest $PROCESSED/manifest.csv \
    --output $PROCESSED/transcriptions \
    --device cuda:1

echo "[$(date)] Transcription complete!"
TRANS=$(find $PROCESSED/transcriptions -name "*.json" | wc -l)
echo "Transcriptions: $TRANS"

# Step 3: Build F5-TTS training CSV
echo ""
echo "[$(date)] === STEP 3: Building F5-TTS dataset ==="
TRAIN_CSV=$PROCESSED/f5_train.csv

python3 /mnt/18tb/build_f5_csv.py $PROCESSED/transcriptions $TRAIN_CSV

# Prepare Arrow dataset
F5_DATASET=$BASE/vintage_voice_f5_dataset
python3 -m f5_tts.train.datasets.prepare_csv_wavs \
    $TRAIN_CSV \
    $F5_DATASET

echo "[$(date)] Dataset prepared! Ready for training."
echo "Run: bash /mnt/18tb/run_pipeline.sh 4"
