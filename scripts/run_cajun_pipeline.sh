#!/bin/bash
# VintageVoice — Cajun French end-to-end pipeline (Victus RTX 4070 8GB)
#
# Stages:
#   A. Wait for archive.org download to finish
#   B. Preprocess raw audio -> 24kHz mono segments + manifest (CPU)
#   C. Free GPU (kill llama-servers by process; abort if VRAM won't drain)
#   D. Transcribe with Whisper medium + language/QC capture
#   E. Build F5 train CSV (French, QC-clean only) + Arrow dataset
#   F. Fine-tune from OpenF5-TTS-Base (Apache 2.0) — batch sized for 8GB
#   G. Restore llama-servers
#   H. Generate Cajun French test lines in Sophia's voice
#
# Resume-safe: each stage skips if its output already exists.
set -Eeuo pipefail

BASE=/home/scott/vintage-voice
VENV=$BASE/venv/bin
RAW=$BASE/data/raw/louisiana
PROCESSED=$BASE/data/processed/louisiana
TRANSCRIBED=$BASE/data/transcribed/louisiana
F5_CSV=$PROCESSED/f5_train_cajun_fr.csv
F5_DATASET=$BASE/data/f5_cajun_french
DATASET_NAME=cajun_french
# model_finetune_init.pt = model.pt with optimizer dropped and update counter
# zeroed — the raw OpenF5 ckpt reports update=1000000, which makes finetune_cli
# think training is already complete and exit 0 without training.
PRETRAIN=$BASE/models/OpenF5-TTS-Base/model_finetune_init.pt
OPENF5_VOCAB=$BASE/models/OpenF5-TTS-Base/vocab.txt
SOPHIA_REF=/home/scott/vintage-voice-samples-50ep/sophia_ref.wav
OUT_AUDIO=$BASE/data/output/sophia_cajun
STATE=$BASE/data/pipeline_state.log
LLAMA_RESTORE=$BASE/data/llama_restore.sh

mkdir -p "$BASE/data" "$OUT_AUDIO"
stage() { echo "[$(date '+%F %T')] === $* ===" | tee -a "$STATE"; }

restore_llama() {
    stage "Restoring llama-servers"
    # The openclaw-llama unit is in a failed state and its process was orphaned;
    # restore from captured command lines only — do NOT systemctl start (port clash).
    if [ -f "$LLAMA_RESTORE" ]; then bash "$LLAMA_RESTORE" || true; fi
}
trap 'code=$?; stage "PIPELINE FAILED (code $code) at line $LINENO"; restore_llama; exit $code' ERR

# ---------- Stage A: wait for download ----------
stage "A: waiting for download_louisiana.py to finish"
while pgrep -f "download_louisiana.py" > /dev/null; do
    sleep 60
done
stage "A: download complete: $(du -sh $RAW 2>/dev/null | cut -f1)"

# ---------- Stage B: preprocess ----------
if [ ! -f "$PROCESSED/manifest.csv" ]; then
    stage "B: preprocessing to 24kHz segments"
    $VENV/python $BASE/scripts/preprocess.py \
        --input "$RAW" --output "$PROCESSED" --workers 6
else
    stage "B: skip (manifest exists)"
fi
SEGS=$(find $PROCESSED/segments -name '*.wav' 2>/dev/null | wc -l)
stage "B: $SEGS segments"
if [ "$SEGS" -lt 50 ]; then
    stage "FATAL: too few segments"
    exit 1
fi

# ---------- Stage C: free the GPU ----------
stage "C: freeing GPU"
# Kill ALL llama-servers by process (systemd unit is failed/orphaned — pkill is truth).
# Capture their command lines first so Stage G can restore them.
PIDS=$(pgrep -f "llama.cpp/build-cuda/bin/llama-server" || true)
if [ -n "$PIDS" ]; then
    ps -o args= -p $PIDS | sed 's/^/nohup /; s|$| >> /tmp/llama_restored.log 2>\&1 \&|' > "$LLAMA_RESTORE"
    stage "C: killing llama-servers: $PIDS (restore saved)"
    kill $PIDS 2>/dev/null || true
    sleep 5
    kill -9 $PIDS 2>/dev/null || true
fi
DRAINED=0
for i in $(seq 1 24); do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    if [ "$USED" -lt 800 ]; then DRAINED=1; break; fi
    stage "C: waiting for VRAM to drain (${USED}MiB used)"
    sleep 5
done
if [ "$DRAINED" -ne 1 ]; then
    stage "FATAL: VRAM never drained — something else holds the GPU:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv | tee -a "$STATE"
    exit 1
fi
stage "C: GPU free ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader))"

# ---------- Stage D: transcribe ----------
stage "D: Whisper medium transcription (language + QC capture, watchdog-wrapped)"
mkdir -p "$TRANSCRIBED"
$VENV/python $BASE/scripts/transcribe_watchdog.py \
    --manifest "$PROCESSED/manifest.csv" \
    --output "$TRANSCRIBED" \
    --model medium --device cuda --stale-min 15
stage "D: done — manifests: $(ls $TRANSCRIBED/train_*.csv 2>/dev/null | xargs -n1 basename | tr '\n' ' ' || true)"

# ---------- Stage E: build F5 dataset (French, QC-clean) ----------
# Resume guard (matches Stage B pattern): on a re-run after a later-stage crash,
# skip the expensive CSV + Arrow rebuild — BUT only if the existing CSV is
# actually complete AND up-to-date. A crashed Stage E could leave a
# truncated/partial CSV on disk; skipping on mere existence would feed garbage
# into training. We therefore (a) write the CSV atomically (temp file +
# os.replace, so it only appears once fully written), (b) verify completeness
# (header present + >= the required 100 sample rows), and (c) require the CSV to
# be NEWER than every transcribed input it derives from — otherwise a re-run
# after Stage D produced fresh/changed transcriptions would silently train on a
# stale CSV.
stage_e_csv_complete() {
    [ -s "$F5_CSV" ] || return 1
    # Staleness check: if any transcribed *.json is newer than the CSV, the
    # transcriptions changed since the CSV was built — force a rebuild. `find
    # -newer` prints the offending file(s); a non-empty result means stale.
    if [ -d "$TRANSCRIBED" ] && \
       [ -n "$(find "$TRANSCRIBED" -maxdepth 1 -name '*.json' -newer "$F5_CSV" -print -quit 2>/dev/null)" ]; then
        return 1
    fi
    $VENV/python - "$F5_CSV" <<'PYEOF'
import sys
path = sys.argv[1]
try:
    with open(path, encoding="utf-8") as fh:
        lines = [ln for ln in (l.rstrip("\n") for l in fh) if ln]
except OSError:
    sys.exit(1)
# Must have the header plus at least the 100 sample rows Stage E enforces.
if not lines or lines[0] != "audio_file|text":
    sys.exit(1)
sys.exit(0 if (len(lines) - 1) >= 100 else 1)
PYEOF
}

if stage_e_csv_complete; then
    stage "E: skip (complete F5 CSV exists)"
else
    stage "E: building F5 CSV from clean French segments"
    $VENV/python - "$TRANSCRIBED" "$F5_CSV" <<'EOF'
import glob, json, os, sys, unicodedata, tempfile
trans_dir, out_csv = sys.argv[1], sys.argv[2]
vocab = set(l.rstrip("\n") for l in open("/home/scott/vintage-voice/models/OpenF5-TTS-Base/vocab.txt"))
rows, dropped_chars, hours = [], set(), 0.0
for jf in sorted(glob.glob(os.path.join(trans_dir, "*.json"))):
    d = json.load(open(jf))
    if d.get("language") != "fr" or d.get("qc_flags"):
        continue
    text = unicodedata.normalize("NFC", d["text"]).replace("ÿ", "y")
    text = text.replace("|", " ").replace("\n", " ").strip()
    bad = set(c for c in text if c not in vocab and c != " ")
    if bad:
        dropped_chars |= bad
        text = "".join(c for c in text if c in vocab or c == " ")
    if len(text) > 5 and os.path.exists(d["audio_path"]):
        rows.append(f'{d["audio_path"]}|{text}')
        hours += d.get("duration", 0) / 3600
print(f"French clean samples: {len(rows)}  ({hours:.2f}h)")
if dropped_chars:
    print(f"chars stripped (not in vocab): {sorted(dropped_chars)}")
if len(rows) < 100:
    # Abort BEFORE writing — never leave a partial/undersized CSV that a later
    # resume could mistake for valid output.
    print("FATAL: under 100 clean French samples — not enough to fine-tune")
    sys.exit(3)
# Atomic write: build the full file in a temp path, then os.replace() it into
# place so the CSV at out_csv is only ever the complete article.
out_dir = os.path.dirname(out_csv) or "."
fd, tmp = tempfile.mkstemp(suffix=".tmp", dir=out_dir)
try:
    with os.fdopen(fd, "w") as f:
        f.write("audio_file|text\n" + "\n".join(rows) + "\n")
    os.replace(tmp, out_csv)
except Exception:
    if os.path.exists(tmp):
        os.unlink(tmp)
    raise
EOF
fi

stage "E: preparing Arrow dataset"
# Pre-seed pretrained vocab where prepare_csv_wavs looks for it in finetune mode
F5_DATA_ROOT=$($VENV/python -c "from importlib.resources import files; import os; print(os.path.normpath(str(files('f5_tts').joinpath('../../data'))))")
mkdir -p "$F5_DATA_ROOT/Emilia_ZH_EN_pinyin"
cp "$OPENF5_VOCAB" "$F5_DATA_ROOT/Emilia_ZH_EN_pinyin/vocab.txt"
$VENV/python -m f5_tts.train.datasets.prepare_csv_wavs "$F5_CSV" "$F5_DATASET"
# Vocab MUST match the pretrained checkpoint's embedding table exactly
cp "$OPENF5_VOCAB" "$F5_DATASET/vocab.txt"
ln -sfn "$F5_DATASET" "$F5_DATA_ROOT/${DATASET_NAME}_custom"
stage "E: dataset at $F5_DATASET (linked as ${DATASET_NAME}_custom)"

# ---------- Stage F: fine-tune (8GB-sized) ----------
stage "F: fine-tuning OpenF5 on Cajun French (8GB: fp16 + 8-bit AdamW + 800 frames x accum 4)"
cd $BASE
# fp32 training of 337M params = ~6.7GB in weights/grads/Adam/EMA alone -> OOM on 8GB.
# fp16 autocast (env honored: Trainer passes no mixed_precision) + bnb 8-bit AdamW
# cuts optimizer state ~2GB and halves activation memory.
# FIX: removed dead PYTORCH_ALLOC_CONF (typo — wrong variable name, had no
# effect). Only PYTORCH_CUDA_ALLOC_CONF is honored for expandable_segments.
WANDB_MODE=offline \
ACCELERATE_MIXED_PRECISION=fp16 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 $VENV/python -m f5_tts.train.finetune_cli \
    --bnb_optimizer \
    --exp_name F5TTS_v1_Base \
    --pretrain "$PRETRAIN" \
    --dataset_name $DATASET_NAME \
    --tokenizer custom \
    --tokenizer_path "$F5_DATASET/vocab.txt" \
    --learning_rate 1e-5 \
    --batch_size_per_gpu 800 \
    --batch_size_type frame \
    --max_samples 16 \
    --grad_accumulation_steps 4 \
    --max_grad_norm 1.0 \
    --epochs 50 \
    --num_warmup_updates 200 \
    --save_per_updates 1000 \
    --last_per_updates 500 \
    --keep_last_n_checkpoints 3 \
    --logger tensorboard \
    --finetune
stage "F: training finished"

# ---------- Stage G: restore llama-servers ----------
restore_llama

# ---------- Stage H: Sophia speaks Cajun French ----------
stage "H: generating Sophia-voice Cajun French samples"
# finetune_cli writes checkpoints relative to the f5_tts package, not CWD
CKPT_DIR="$(dirname "$F5_DATA_ROOT")/ckpts/$DATASET_NAME"
CKPT=$(ls -t "$CKPT_DIR"/model_*.pt 2>/dev/null | head -1 || true)
if [ -z "$CKPT" ]; then
    stage "H: no checkpoint found, skipping"
    exit 1
fi
i=0
while IFS= read -r line; do
    i=$((i+1))
    $VENV/f5-tts_infer-cli \
        --model F5TTS_v1_Base \
        --ckpt_file "$CKPT" \
        --vocab_file "$F5_DATASET/vocab.txt" \
        --ref_audio "$SOPHIA_REF" \
        --ref_text "reporting from the Serengeti. Here at Elion Labs, we've successfully trapped and tagged approximately 15.5 of these majestic little creatures." \
        --gen_text "$line" \
        --output_dir "$OUT_AUDIO" \
        --output_file "sophia_cajun_$i.wav" || true
done <<'LINES'
Comment ça va, mon ami ? Ça fait longtemps que je t'ai pas vu, ouais.
Laissez les bons temps rouler ! On va faire un bon gombo ce soir, cher.
Mais regarde donc ça. La récolte de cannes est belle cette année, oui.
LINES
stage "H: samples in $OUT_AUDIO"
stage "PIPELINE COMPLETE"
