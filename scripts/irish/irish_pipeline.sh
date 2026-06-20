#!/bin/bash
# Irish Sophia corpus pipeline — Tadhg Hynes (Dubliners + Portrait of the Artist).
# Mirrors charrer_pipeline.sh: preprocess -> free GPU -> whisper transcribe ->
# yield. English segments (Irish English) land in train_en.csv automatically.
set -u
BASE=/home/scott/vintage-voice
VENV=$BASE/venv
RAW=$BASE/data/raw/irish_tadhg
PROC=$BASE/data/processed/irish_tadhg
TRANS=$BASE/data/transcribed/irish_tadhg
RESTORE=$BASE/data/llama_restore_irish.sh
cd "$BASE"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

restore_llamas(){
  systemctl --user start openclaw-llama 2>/dev/null || true
  [ -f "$RESTORE" ] && bash "$RESTORE" 2>/dev/null
  log "llamas restored (GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null))"
}
trap restore_llamas EXIT   # always bring the local LLMs back, even on failure/kill

log "irish files: $(ls "$RAW"/*.mp3 2>/dev/null | wc -l)"

# 1. preprocess (CPU) -> 24kHz mono segments + manifest (drops silence/boilerplate via VAD)
log "preprocess start (CPU, long for ~15h corpus)"
$VENV/bin/python scripts/preprocess.py --input "$RAW" --output "$PROC" --workers 8 > /tmp/irish_prep.log 2>&1
[ -f "$PROC/manifest.csv" ] || { log "FATAL no manifest — see /tmp/irish_prep.log"; exit 1; }
log "preprocess done: $(($(wc -l < "$PROC/manifest.csv")-1)) segments"

# 2. free the 4070 (stop local llama servers; restore script captures :8082 cmdline)
P8082=$(pgrep -f 'llama-server.*port 8082' || true)
[ -n "$P8082" ] && ps -o args= -p $P8082 | sed 's/^/nohup /; s|$| >> /tmp/llama_restored.log 2>\&1 \&|' > "$RESTORE"
log "stopping llamas to free GPU"
systemctl --user stop openclaw-llama 2>/dev/null || true
[ -n "$P8082" ] && { kill $P8082 2>/dev/null; sleep 4; kill -9 $P8082 2>/dev/null || true; }
for i in $(seq 1 24); do U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null); [ "${U:-9999}" -lt 800 ] && break; sleep 5; done
log "GPU free ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null))"

# 3. transcribe (whisper-medium, watchdog restarts if it wedges)
mkdir -p "$TRANS"
log "whisper transcribe start (~2-3h)"
$VENV/bin/python scripts/transcribe_watchdog.py --manifest "$PROC/manifest.csv" --output "$TRANS" --model medium --device cuda --stale-min 15
log "transcribe rc=$?"

# 4. llamas restored by EXIT trap

# 5. yield — Irish English segments
log "=== IRISH ENGLISH YIELD ==="
TE="$TRANS/train_en.csv"
if [ -f "$TE" ]; then
  awk -F'|' 'NR>1{s+=$3;n++} END{printf "clean EN: %d segs, %.1f min (%.2fh)\n", n, s/60, s/3600}' "$TE"
else
  log "no train_en.csv yet"
fi
$VENV/bin/python - <<'PY'
import json, glob, collections
c = collections.Counter()
for fp in glob.glob("/home/scott/vintage-voice/data/transcribed/irish_tadhg/*.json"):
    try: c[json.load(open(fp)).get("language", "?")] += 1
    except Exception: pass
print("by language:", dict(c.most_common(6)))
PY
log "DONE"
