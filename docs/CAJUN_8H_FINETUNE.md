# Cajun French Sophia — 8.1h CosyVoice2 Finetune

Finetuning **CosyVoice2-0.5B** to speak **Louisiana / Cajun French** in a target
voice, trained on a **single 8 GB laptop GPU** (RTX 4070, Python 3.12, torch 2.3.1).

This documents the corpus build, the training recipe (incl. the non-obvious fixes
required to make the official recipe run on 8 GB + py3.12), and how to synthesize.

## Result

- Corpus grown **1.84h → ~8.1h** of clean Cajun French (3,750 QC-passed segments).
- Intelligible Cajun French in the target voice. Whisper round-trip is verbatim on
  conversational lines (e.g. *"on va faire un bon gombo ce soir, cher"*); the model
  still has a robotic texture (only the LLM was finetuned — see *Next steps*).
- Best checkpoint by CV loss = **epoch 2** (the small corpus overfits by ~epoch 6).

## Corpus

Audio gathered with `yt-dlp` (audio-only `.mp3`, ~60 MB/hour — disk-cheap) +
archive.org, then `preprocess.py` (silence-split to 24 kHz segments) →
`transcribe_cajun.py` (whisper-medium, per-segment language + QC signals) →
French-only QC-clean CSV.

| Source | Clean FR segs | Hours | French density |
|--------|--------------:|------:|---------------:|
| archive.org (Louisiana collections) | 830 | 1.72h | 6.5% |
| YouTube clips (interviews) | 212 | 0.45h | 37% |
| **Charrer-Veiller** (Cajun French podcast) | 665 | 1.56h | **88%** |
| **Télé-Louisiane** (Louisiana French media) | 2,043 | 4.36h | 54% |
| **Total** | **3,750** | **~8.1h** | |

**Lesson:** long-form, single-dialect **podcasts / radio** are the goldmine —
far higher French density than broad archive.org searches (which mostly return
English Louisiana-culture content). Keep European/Parisian French OUT (whisper
tags everything `fr` and can't tell Cajun from standard French — filter by source).

Pipeline scripts: `scripts/cajun8h/{yt_cajun_pull2,charrer_pipeline,tele_pipeline}.sh`
(each: download → preprocess → whisper transcribe → French-yield report).

## Training (single 8 GB GPU)

The official `examples/libritts/cosyvoice2` recipe assumes 4 GPUs + deepspeed +
Python ≤3.11. Getting it onto an 8 GB / py3.12 / torch-2.3.1 box took **11 fixes**:

1. **Launcher env** — use the project venv's `torchrun`, or bypass torchrun entirely.
2. **deepspeed hard import** in `train.py`/`train_utils.py` → wrap `try/except → None`
   (torch_ddp never uses it).
3. **torchrun c10d rendezvous segfault** (torch 2.3) → run `train.py` directly with
   `RANK/LOCAL_RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT` env vars set.
4. **"Dynamo is not supported on Python 3.12+"** → patch torch so `torch.compile` is a
   no-op on py3.12 (we run eager); transformers calls it at import.
5. **deepspeed poisons transformers** — if deepspeed is installed,
   `is_deepspeed_available()` → transformers imports it → needs `torch.library.custom_op`
   (torch 2.4+). Fix: **uninstall deepspeed** (the try/except handles its absence).
6. **CV data path** — `data.list` holds root-relative paths; absolutize them and use an
   absolute `--config` (empty CV loader → `KeyError: 'tag'`).
7. **fp32 Adam OOM** → 8-bit Adam (bitsandbytes `AdamW8bit`); `optim: adam8bit`.
8. **Wasted LM head** — `Qwen2Encoder.forward` ran full `Qwen2ForCausalLM` (151k-vocab
   logits CosyVoice never uses) → call the base `self.model.model` instead.
9. **accum_grad 2→1** (free gradients every step).
10. **DDP grad-bucket duplication** → `DistributedDataParallel(..., gradient_as_bucket_view=True)`.
11. **Gradient checkpointing × DDP** — reentrant checkpointing fires DDP hooks twice
    ("marked ready twice") → `gradient_checkpointing_enable(use_reentrant=False)` +
    `config.use_cache=False`.

With these, the 0.5B LLM finetune is stable at **~5.1 GB** VRAM. Config:
`configs/cosyvoice2_8gb.yaml` (lr 1e-5 constant SFT, `max_frames_in_batch: 800`,
`adam8bit`, `accum_grad: 1`, `max_epoch: 30`, checkpoint per epoch).

Launch (frees the GPU, trains, restores other GPU services via an EXIT trap):
`scripts/cajun8h/cosyvoice_train_wrapped.sh` → `cosyvoice_train_llm.sh`.

## Synthesis

Build a model dir = symlink the base `CosyVoice2-0.5B` files + overwrite `llm.pt`
with a finetuned `epoch_N_whole.pt` (drop its `epoch`/`step` keys). Then:

```bash
CUDA_VISIBLE_DEVICES="" venv-cosy/bin/python scripts/cajun8h/cajun_say.py \
    "Mais comment ça va, cher ?" sortie
# -> data/output/cajun_say/sortie.wav
```

A/B compare base vs finetuned checkpoints: `scripts/cosy_cajun_compare.py`.

## Next steps (toward natural, not just intelligible)

- **Finetune the flow model** too (`--model flow`), not just the LLM — the flow +
  vocoder produce the acoustic texture, so this is the biggest naturalness gain.
- **More hours** — remaining Télé-Louisiane, KVPI radio, family recordings.
- **Deeper run on a 32 GB V100** — lifts the 8 GB batch/precision constraints.
