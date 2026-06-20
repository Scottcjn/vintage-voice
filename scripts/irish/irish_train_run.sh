#!/bin/bash
# Launch the Irish CosyVoice2 finetune on the 4070: free the GPU (stop llamas),
# train, and ALWAYS restore the llamas afterward (trap), even on crash/kill.
set -u
BASE=/home/scott/vintage-voice
RESTORE=$BASE/data/llama_restore_irish_train.sh
log(){ echo "[$(date +%H:%M:%S)] $*"; }

restore_llamas(){
  systemctl --user start openclaw-llama 2>/dev/null || true
  if ! pgrep -f 'llama-server.*port 8082' >/dev/null && [ -f "$RESTORE" ]; then
    bash "$RESTORE" 2>/dev/null || true
  fi
  log "llamas restored ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null))"
}
trap restore_llamas EXIT

# free the 4070
P8082=$(pgrep -f 'llama-server.*port 8082' || true)
[ -n "$P8082" ] && ps -o args= -p $P8082 | sed 's/^/nohup /; s|$| >> /tmp/llama_restored.log 2>\&1 \&|' > "$RESTORE"
log "stopping llamas to free GPU for training"
systemctl --user stop openclaw-llama 2>/dev/null || true
[ -n "$P8082" ] && { kill $P8082 2>/dev/null; sleep 4; kill -9 $P8082 2>/dev/null || true; }
for i in $(seq 1 24); do U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null); [ "${U:-9999}" -lt 800 ] && break; sleep 5; done
log "GPU free ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null))"

log "=== IRISH FINETUNE START (CosyVoice2-0.5B llm SFT, 10.12h corpus) ==="
bash ~/cosyvoice_train_irish.sh
rc=$?
log "train rc=$rc === IRISH FINETUNE END ==="
log "checkpoints: $(ls $BASE/exp/irish/llm/*.pt 2>/dev/null | wc -l) in exp/irish/llm/"
