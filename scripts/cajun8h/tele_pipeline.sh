#!/bin/bash
set -u
BASE=/home/scott/vintage-voice
VENV=$BASE/venv
RAW=$BASE/data/raw/tele
PROC=$BASE/data/processed/tele
TRANS=$BASE/data/transcribed/tele
RESTORE=$BASE/data/llama_restore_tele.sh
cd "$BASE"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

# 0. wait for the download script to fully finish
log "waiting for download to finish..."
while pgrep -f yt_cajun_pull2 >/dev/null; do sleep 10; done

# 1. isolate Télé-Louisiane files (English ones included; whisper keeps only fr)
mkdir -p "$RAW"
cp -n data/raw/youtube/cajun_fr/Tele-Louisiane__*.mp3 "$RAW"/ 2>/dev/null
log "tele files: $(ls "$RAW"/*.mp3 2>/dev/null | wc -l)"

# 2. preprocess (CPU)
log "preprocess start (CPU, ~30h audio — this takes a while)"
$VENV/bin/python scripts/preprocess.py --input "$RAW" --output "$PROC" --workers 8 > /tmp/tele_prep.log 2>&1
[ -f "$PROC/manifest.csv" ] || { log "FATAL no manifest"; exit 1; }
log "preprocess done: $(($(wc -l < "$PROC/manifest.csv")-1)) segments"

# 3. free GPU
P8082=$(pgrep -f 'llama-server.*port 8082' || true)
[ -n "$P8082" ] && ps -o args= -p $P8082 | sed 's/^/nohup /; s|$| >> /tmp/llama_restored.log 2>\&1 \&|' > "$RESTORE"
log "stopping llamas"
systemctl --user stop openclaw-llama 2>/dev/null || true
[ -n "$P8082" ] && { kill $P8082 2>/dev/null; sleep 4; kill -9 $P8082 2>/dev/null || true; }
for i in $(seq 1 24); do U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$U" -lt 800 ] && break; sleep 5; done
log "GPU free ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader))"

# 4. transcribe (LONG: ~2-3h)
mkdir -p "$TRANS"
log "whisper transcribe start (~2-3h)"
$VENV/bin/python scripts/transcribe_watchdog.py --manifest "$PROC/manifest.csv" --output "$TRANS" --model medium --device cuda --stale-min 15
log "transcribe rc=$?"

# 5. restore llamas
systemctl --user start openclaw-llama 2>/dev/null || true
[ -f "$RESTORE" ] && bash "$RESTORE"
sleep 3
log "llamas restored (GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader))"

# 6. yield
log "=== TELE-LOUISIANE FRENCH YIELD ==="
TF="$TRANS/train_fr.csv"
[ -f "$TF" ] && awk -F'|' 'NR>1{s+=$3;n++} END{printf "clean FR: %d segs, %.1f min (%.2fh)\n", n, s/60, s/3600}' "$TF"
$VENV/bin/python - <<PY
import json,glob,collections
c=collections.Counter()
for fp in glob.glob("$TRANS/*.json"):
    try: c[json.load(open(fp)).get("language","?")]+=1
    except: pass
print("by language:", dict(c.most_common(8)))
PY
log "DONE"
