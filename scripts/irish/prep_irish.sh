#!/bin/bash
# Build CosyVoice2 parquet for the prairie (St. Landry) corpus — recipe stages 0-3.
# Uses speech_tokenizer_v2.onnx to match the existing 8h parquet.
set -u
exec >/tmp/prep_irish.log 2>&1
BASE=/home/scott/vintage-voice
REPO=$BASE/models/cosyvoice-repo
M=$BASE/models/CosyVoice2-0.5B
PY=$BASE/venv-cosy/bin/python
DATA=$BASE/data/cosyvoice_irish
CSV=$BASE/data/transcribed/irish_tadhg/train_en_cosy.csv
RESTORE=$BASE/data/llama_restore_irish_prep.sh
export PYTHONPATH=$REPO:$REPO/third_party/Matcha-TTS
export PYTHONIOENCODING=UTF-8
cd "$BASE"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

restore_llamas(){
  systemctl --user start openclaw-llama 2>/dev/null || true
  if ! pgrep -f 'llama-server.*port 8082' >/dev/null && [ -f "$RESTORE" ]; then
    bash "$RESTORE" 2>/dev/null || true
  fi
  log "llamas restored ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null))"
}
trap restore_llamas EXIT   # never strand the llamas, even on error

# stage 0: kaldi files
log "stage0: prep kaldi files"
$PY scripts/prep_cosyvoice_data.py --csv "$CSV" --output-dir "$DATA" || { log "stage0 FAILED"; exit 1; }

# free GPU
P8082=$(pgrep -f 'llama-server.*port 8082' || true)
[ -n "$P8082" ] && ps -o args= -p $P8082 | sed 's/^/nohup /; s|$| >> /tmp/llama_restored.log 2>\&1 \&|' > "$RESTORE"
log "stopping llamas for onnx extraction"
systemctl --user stop openclaw-llama 2>/dev/null || true
[ -n "$P8082" ] && { kill $P8082 2>/dev/null; sleep 4; kill -9 $P8082 2>/dev/null || true; }
for i in $(seq 1 24); do U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$U" -lt 800 ] && break; sleep 5; done
log "GPU free ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader))"

for x in train dev; do
  log "stage1: campplus embeddings ($x)"
  $PY $REPO/tools/extract_embedding.py --dir "$DATA/$x" --onnx_path "$M/campplus.onnx" || { log "stage1 $x FAILED"; exit 1; }
  log "stage2: speech tokens v2 ($x)"
  $PY $REPO/tools/extract_speech_token.py --dir "$DATA/$x" --onnx_path "$M/speech_tokenizer_v2.onnx" || { log "stage2 $x FAILED"; exit 1; }
  log "stage3: parquet ($x)"
  mkdir -p "$DATA/$x/parquet"
  $PY $REPO/tools/make_parquet_list.py --num_utts_per_parquet 1000 --num_processes 8 \
      --src_dir "$DATA/$x" --des_dir "$DATA/$x/parquet" || { log "stage3 $x FAILED"; exit 1; }
done

log "=== IRISH PARQUET DONE ==="
echo "train parquet: $(ls "$DATA"/train/parquet/*.tar 2>/dev/null | wc -l) tars"
echo "dev parquet:   $(ls "$DATA"/dev/parquet/*.tar 2>/dev/null | wc -l) tars"
cat "$DATA"/train/parquet/data.list 2>/dev/null
log "PREP COMPLETE"
