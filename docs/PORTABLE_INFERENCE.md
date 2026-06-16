# Portable inference on PPC, POWER, ARM, and Apple Silicon

VintageVoice inference is pure Python on top of F5-TTS and PyTorch. The normal
CUDA path is fastest, but the model can also run on CPU-only or Apple MPS hosts
when the Python packages support that architecture.

This document covers the supported portable path added for PowerPC, IBM POWER,
ARM Linux, and Apple Silicon machines.

## Hardware targets

| Target | Device default | Notes |
| --- | --- | --- |
| Apple Silicon (arm64 macOS) | `mps` when available, otherwise `cpu` | Use current Python and PyTorch wheels. |
| ARM Linux (aarch64/armv7) | `cpu` | Prefer a 64-bit OS on aarch64 boards. |
| IBM POWER / PowerPC (ppc64le/ppc64/ppc) | `cpu` | PyTorch may need distro, conda-forge, or source builds. |
| x86_64 with CUDA | `cuda:0` | Existing fast path remains unchanged. |

The portable CLI chooses a conservative thread count, disables CUDA by default
on PPC/POWER/ARM targets, and reuses `scripts/generate.py` for the actual
F5-TTS call.

## Install

Use Python 3.10 or 3.11 where possible. F5-TTS and PyTorch dependency support
varies by architecture, so install PyTorch first for your platform, then install
F5-TTS.

### Apple Silicon

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools
pip install torch torchaudio
pip install f5-tts huggingface_hub
```

### ARM Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install f5-tts huggingface_hub
```

### IBM POWER / PowerPC

PowerPC wheel availability changes by distro. Start with your OS packages or
conda-forge if `pip install torch` does not provide a wheel for the host.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools
pip install torch torchaudio || echo "Install PyTorch from distro/conda/source for this architecture"
pip install f5-tts huggingface_hub
```

On small CPU-only machines, set the thread count explicitly:

```bash
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
```

## Download weights

```bash
huggingface-cli download AutomatedJanitor/vintage-voice \
  model.safetensors vocab.txt \
  --local-dir ./weights
```

## Run inference

Bring a consented 5-15 second reference WAV and the exact transcript of that
clip. Passing `--ref-text` avoids F5-TTS invoking Whisper to transcribe the
reference clip during inference.

```bash
python scripts/portable_infer.py \
  "Good evening, ladies and gentlemen, this is VintageVoice on real hardware." \
  --model ./weights/model.safetensors \
  --vocab ./weights/vocab.txt \
  --ref-audio ./refs/my_voice.wav \
  --ref-text "Exact words spoken in my voice reference." \
  --output portable_out.wav
```

Override device selection when needed:

```bash
python scripts/portable_infer.py "Testing MPS." --device mps ...
python scripts/portable_infer.py "Testing CPU." --device cpu --threads 2 ...
```

## Benchmark

The benchmark writes generated WAV files and prints JSON containing host
metadata, wall time, output audio duration, and real-time factor.

```bash
python scripts/benchmark_inference.py \
  --model ./weights/model.safetensors \
  --vocab ./weights/vocab.txt \
  --ref-audio ./refs/my_voice.wav \
  --ref-text "Exact words spoken in my voice reference." \
  --runs 3 \
  --device auto \
  --threads 4 \
  --json benchmarks/my-host.json
```

Example result shape:

```json
{
  "host": {"system": "Linux", "machine": "ppc64le", "python": "3.11.9"},
  "inference": {"device": "cpu", "threads": 4, "runs": 3},
  "summary": {"avg_wall_seconds": 86.4, "avg_audio_seconds": 5.2, "avg_real_time_factor": 16.6}
}
```

For reproducible submissions, include the JSON file, exact PyTorch/F5-TTS
versions, CPU model, RAM, OS version, and whether the run used CPU, MPS, or CUDA.

## Troubleshooting

- If `f5-tts` cannot install on PPC/POWER, install PyTorch from your distro or
  conda-forge first, then retry F5-TTS.
- If inference is much slower than expected, reduce `--threads`; small boards can
  be slower with too many BLAS threads.
- If reference transcription fails or takes a long time, pass the exact
  `--ref-text` transcript so Whisper is skipped.
- If Apple Silicon MPS errors on an operation, retry with `--device cpu`; PyTorch
  MPS coverage depends on the installed version.
