# Portable Inference Notes

VintageVoice now supports an `auto` inference device mode for hardware where
CUDA cannot be assumed. The goal is to keep the existing F5-TTS generation path
intact while making PPC, POWER, ARM, and Apple Silicon runs explicit and
measurable.

## Device Selection

`scripts.generate.generate_speech(..., device="auto")` resolves the runtime
device as follows:

| Host | Device | Reason |
| --- | --- | --- |
| macOS arm64/aarch64 with PyTorch MPS | `mps` | Use Apple Silicon acceleration |
| PPC, POWER, Linux ARM, or other CPU-first portable hosts | `cpu` | Avoid unsupported accelerator assumptions |
| Conventional host with CUDA available | `cuda:0` | Preserve the current fast GPU path |
| Any other host | `cpu` | Safe fallback |

Passing an explicit F5-TTS device string such as `cpu`, `cuda:1`, or `mps`
still overrides the automatic plan.

## Dry-Run The Target

Use the dry-run mode first on any new machine. It does not import or run F5-TTS,
so it can validate platform detection before model dependencies are installed.

```bash
python scripts/benchmark_inference.py \
  --device auto \
  --dry-run \
  --json benchmark-plan.json
```

Example JSON shape:

```json
{
  "dry_run": true,
  "plan": {
    "device": "cpu",
    "machine": "ppc64le",
    "reason": "portable CPU-first target architecture",
    "system": "linux"
  },
  "runs": [],
  "warmup": 0
}
```

## Run A Live Benchmark

After installing F5-TTS and downloading the model/vocab, use the same model,
reference audio, and transcript you would pass to `scripts/generate.py`.

```bash
python scripts/benchmark_inference.py \
  --device auto \
  --model ./weights/model.safetensors \
  --vocab ./weights/vocab.txt \
  --ref-audio path/to/reference.wav \
  --ref-text "Exact transcript of the reference clip." \
  --runs 3 \
  --warmup 1 \
  --output-dir benchmark_out \
  --json benchmark-result.json
```

The live JSON includes the resolved device and one timing entry per measured
run. Warmup runs are executed but not included in `runs`.

## Generate Directly

`scripts/generate.py` also defaults to portable device selection:

```bash
python scripts/generate.py \
  "Good evening, ladies and gentlemen." \
  --model ./weights/model.safetensors \
  --vocab ./weights/vocab.txt \
  --ref-audio path/to/reference.wav \
  --ref-text "Exact transcript of the reference clip." \
  --output out.wav
```

Use `--device cpu` when you need to force the slow but predictable path, or
`--device mps` / `--device cuda:0` when you want to bypass auto-detection.
