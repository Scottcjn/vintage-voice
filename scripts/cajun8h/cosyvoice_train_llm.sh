#!/bin/bash
# =============================================================================
# CosyVoice2-0.5B  --  Cajun French LLM finetune  (single 8GB GPU)
# =============================================================================
# Trains ONLY the `llm` model (not flow / hifigan) on the prepared Cajun data.
#   - Single GPU, torch_ddp (NOT deepspeed)
#   - max_frames_in_batch=800 caps dynamic batcher at 800 frames (8GB headroom).
#     NOTE: filter.max_length in the YAML is a *sanity bound* (drops pathological
#     outliers), NOT the OOM cap -- the dynamic batcher is.
#   - Pre-extracted speaker embeddings + speech tokens already baked into parquet
#
# Generated for the vintage-voice Cajun finetune. DO NOT auto-run blindly --
# this WILL use GPU 0. Review, then run:  bash /home/scott/cosyvoice_train_llm.sh
# =============================================================================
set -e

# --- Paths -------------------------------------------------------------------
REPO=/home/scott/vintage-voice/models/cosyvoice-repo
MODEL=/home/scott/vintage-voice/models/CosyVoice2-0.5B
DATA=/home/scott/vintage-voice/data/cosyvoice_cajun_8h
PY=/home/scott/vintage-voice/venv-cosy/bin/python
TORCHRUN=/home/scott/vintage-voice/venv-cosy/bin/torchrun

# --- Environment (per recipe path.sh) ----------------------------------------
export PYTHONIOENCODING=UTF-8
export PYTHONPATH=${REPO}:${REPO}/third_party/Matcha-TTS:$PYTHONPATH

# --- Single-GPU settings -----------------------------------------------------
export CUDA_VISIBLE_DEVICES=0
num_gpus=1
job_id=1986
dist_backend="nccl"          # nccl is fine for single-GPU torch_ddp
train_engine=torch_ddp       # NOT deepspeed
num_workers=2
prefetch=100

# --- Build combined data lists (idempotent) ----------------------------------
# data.list from make_parquet_list holds ROOT-relative paths; absolutize them so
# they resolve regardless of cwd (the cd into DATA otherwise breaks them).
sed 's|^data/|/home/scott/vintage-voice/data/|' ${DATA}/train/parquet/data.list > ${DATA}/train.data.list
sed 's|^data/|/home/scott/vintage-voice/data/|' ${DATA}/dev/parquet/data.list   > ${DATA}/dev.data.list

# --- Train (llm only) --------------------------------------------------------
# conf is run relative to the cajun data dir so conf/cosyvoice2_8gb.yaml resolves.
cd ${DATA}

# Single-GPU: bypass torchrun's flaky c10d elastic rendezvous (segfaults on
# torch 2.3). Set the distributed env vars directly for a 1-process run.
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1
export LOCAL_WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=29533
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MEM_DEBUG=1

${PY} ${REPO}/cosyvoice/bin/train.py \
  --train_engine ${train_engine} \
  --config ${DATA}/conf/cosyvoice2_8gb.yaml \
  --train_data ${DATA}/train.data.list \
  --cv_data ${DATA}/dev.data.list \
  --qwen_pretrain_path ${MODEL}/CosyVoice-BlankEN \
  --onnx_path ${MODEL} \
  --model llm \
  --checkpoint ${MODEL}/llm.pt \
  --model_dir /home/scott/vintage-voice/exp/cajun8h/llm \
  --tensorboard_dir /home/scott/vintage-voice/exp/cajun8h/llm/tensorboard \
  --ddp.dist_backend ${dist_backend} \
  --num_workers ${num_workers} \
  --prefetch ${prefetch} \
  --pin_memory \
  --use_amp
