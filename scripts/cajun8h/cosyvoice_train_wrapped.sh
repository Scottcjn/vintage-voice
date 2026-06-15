#!/bin/bash
# Wraps the CosyVoice2 llm finetune: free GPU (stop llamas) -> train -> ALWAYS restore llamas.
BASE=/home/scott/vintage-voice
RESTORE=$BASE/data/llama_restore_train.sh
log(){ echo "[$(date +%H:%M:%S)] [train-wrap] $*"; }

restore_llamas() {
  log "restoring llamas"
  systemctl --user start openclaw-llama 2>/dev/null || true
  [ -f "$RESTORE" ] && bash "$RESTORE"
  sleep 3
  log "llamas restored (GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader))"
}
trap restore_llamas EXIT

# save :8082 restore command
P8082=$(pgrep -f 'llama-server.*port 8082' || true)
[ -n "$P8082" ] && ps -o args= -p $P8082 | sed 's/^/nohup /; s|$| >> /tmp/llama_restored.log 2>\&1 \&|' > "$RESTORE"
log "stopping llamas to free GPU"
systemctl --user stop openclaw-llama 2>/dev/null || true
[ -n "$P8082" ] && { kill $P8082 2>/dev/null; sleep 4; kill -9 $P8082 2>/dev/null || true; }
for i in $(seq 1 24); do U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$U" -lt 800 ] && break; sleep 5; done
log "GPU free ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader)) — starting training (max 30 epochs)"

bash /home/scott/cosyvoice_train_llm.sh
log "training process exited rc=$?"
# EXIT trap restores llamas
