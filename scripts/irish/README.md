# Irish English voice setup (Tadhg Hynes → CosyVoice2)

Reproducible pipeline to finetune a CosyVoice2-0.5B Irish-English voice and render
it in any speaker's timbre via zero-shot reference. Companion to the `irish_english`
profile (`configs/profiles/irish_english_librivox.yaml`).

## Why finetune (not instruct)
CosyVoice2 instruct mode ("speak with an Irish accent") cannot hold a real,
consistent accent — in testing it drifted Australian/British and sometimes read
the style instruction aloud. The accent has to be **learned from audio**. So we
build a single-speaker Irish corpus and SFT the `llm` on it; the target voice
comes from the zero-shot reference at inference, exactly like the Cajun setup.

## Corpus (public domain)
Anchor reader: **Tadhg Hynes**, a Dubliner who reads solo on LibriVox.
- `dubliners_1302_librivox` — *Dubliners* (Joyce), 6.66 h
- `portraitartist_1402_librivox` — *A Portrait of the Artist as a Young Man* v2, 8.43 h

License: Public Domain (LibriVox). Download: `archive.org/download/<item>/<file>.mp3`.
**Verified yield: 10.12 h / 6,591 clean English segments** (99.96% English, 26%
QC-flagged and excluded).

## Pipeline (run in order)
| Stage | Script | What it does |
|-------|--------|--------------|
| 1. corpus | `irish_pipeline.sh` | preprocess (24 kHz mono, VAD 5–15 s, drop LibriVox boilerplate) → Whisper-medium transcribe → `data/transcribed/irish_tadhg/train_en.csv` |
| 2. parquet | `prep_irish.sh` | kaldi files → campplus speaker embeddings + speech tokens (v2 ONNX) → packed parquet under `data/cosyvoice_irish/{train,dev}/parquet` |
| 3. finetune | `irish_train_run.sh` → `cosyvoice_train_irish.sh` | CosyVoice2-0.5B `llm` SFT on the 8 GB GPU; checkpoints to `exp/irish/llm/epoch_N_whole.pt` |
| 4. render | `irish_render.py <epoch> ["text"] [out.wav]` | builds `models/CosyVoice2-irish-ep<N>` (symlink base + stripped `llm.pt`) and runs `inference_zero_shot` with a reference clip |

`irish_pipeline.sh`, `prep_irish.sh`, and `irish_train_run.sh` free the GPU by
stopping local llama-servers and **always restore them** via an EXIT trap.

## Inference
```bash
CUDA_VISIBLE_DEVICES="" venv-cosy/bin/python scripts/irish/irish_render.py 2 \
  "Good evening, I'm Sophia." out.wav
```
`irish_render.py` uses a zero-shot reference clip (`REF_AUDIO`/`REF_TEXT` — set to
your own speaker; the reference WAV is intentionally not committed) so the output
is *your voice* with the *learned Irish accent*. Speed 1.12 reads naturally.

## Notes
- Pick the epoch by ear: like the Cajun run, expect the sweet spot early (~ep2–4)
  before overfitting — watch CV loss in `exp/irish/llm`.
- Paths are absolute to this lab (`/home/scott/vintage-voice`); adjust for yours.
- Data, checkpoints, and reference WAVs are gitignored — this dir is the recipe only.

## Keeper (verified 2026-06-20)
Best result by ear: **llm epoch 0 + flow epoch 9** → Sophia's voice with a genuine
thick Dublin accent. Key finding: the `llm` overfits immediately (ep0 = CV-best),
but the `flow` keeps improving for ~9 epochs and is what *thickens* the accent
while the zero-shot reference keeps Sophia's female timbre. Locked model dir:
`models/CosyVoice2-irish` (base symlinks + stripped llm.pt[ep0] + flow.pt[ep9]).

Say anything in her Irish voice:
```bash
CUDA_VISIBLE_DEVICES="" venv-cosy/bin/python scripts/irish/irish_say.py "Top of the morning." out
# -> data/output/irish_say/out.wav
```

## Talking head (lip-synced video)
Turn a still portrait + a generated line into a **lip-synced talking-head video**.
The audio drives the mouth directly — LTX-2.3 `LoadAudio → TrimAudioDuration →
LTXVAudioVAEEncode` into the conditioned latent (true sync, not a mouth pasted on).

```bash
# 1. generate the line in Sophia's Irish voice (see above), then
# 2. flatten the LTX-2.3 ia2v workflow ONCE (browser console / Playwright):
#      JSON.stringify((await app.graphToPrompt()).output)  ->  ia2v_graphToPrompt.json
# 3. build the headless prompt:
python3 scripts/irish/ltx_lipsync.py --image portrait.png --audio out.wav \
    --audio-dur 3.46 --prefix sophia_irish
# 4. POST ltx_ia2v_prompt.json to your ComfyUI /prompt  (the POST is the validator:
#    HTTP 400 + node_errors = bad graph, no GPU burned). Poll /history/<id>;
#    output lands in ComfyUI/output/sophia_irish_00001_.mp4
```

`ltx_lipsync.py` resolves the ia2v subgraph **structurally** (by node role), so it
keeps working even when the server lacks the `ComfyMathExpression` custom node — it
resolves those four derived constants (latent W/2, H/2, fps, length=dur·fps+1) to
literals and deletes the nodes. ~6–8 min per clip on a single 32 GB V100; keep
clips **< ~4 s** (lip-sync drifts on longer takes). The capture only needs redoing
if the workflow itself changes.

**Live demo:** [elyanlabs.ai/vintage-voice.html#talking-sophia](https://elyanlabs.ai/vintage-voice.html#talking-sophia)
— Sophia Elya speaking in her fine-tuned Dublin Irish voice, lip-synced on lab hardware.
