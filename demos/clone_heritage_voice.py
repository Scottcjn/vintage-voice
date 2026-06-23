import sys
import os
from pathlib import Path
import subprocess
import argparse
import time
import shutil  # For os.replace alternative on some systems

# --- Initial Dependency Checks ---
# Ensure core Python packages are installed or provide guidance
try:
    import torch
    import torchaudio
except ImportError:
    print("ERROR: torch and torchaudio not found. Please install them:")
    print("  pip install torch torchaudio")
    print("  (Consult pytorch.org for CUDA-specific installation instructions)", file=sys.stderr)
    sys.exit(1)

try:
    import whisper
except ImportError:
    print("ERROR: openai-whisper not found. Please install it:")
    print("  pip install 'whisper-openai>=1.1.0'", file=sys.stderr)
    sys.exit(1)

# --- Path Configuration (relative to this script) ---
# This script is located in vintage-voice/demos/
# So, VINTAGE_VOICE_BASE is two levels up from this script.
VINTAGE_VOICE_BASE = Path(__file__).resolve().parents[1]

COSYVOICE_REPO_PATH = VINTAGE_VOICE_BASE / "models" / "cosyvoice-repo"
CAJUN_MODEL_PATH = VINTAGE_VOICE_BASE / "models" / "CosyVoice2-cajun-ep2"  # Pre-trained Cajun model

# CosyVoice2 uses an internal reference audio and its transcript for general
# prosody/style mapping, separate from the speaker's timbre.
# This default reference is part of the CosyVoice2 standard setup.
DEFAULT_COSYVOICE_REF_AUDIO = VINTAGE_VOICE_BASE / "vintage-voice-samples-50ep" / "sophia_ref.wav"
DEFAULT_COSYVOICE_REF_TEXT = (
    "reporting from the Serengeti. Here at Elion Labs, we've successfully "
    "trapped and tagged approximately 15.5 of these majestic little creatures."
)

OUTPUT_DIR = VINTAGE_VOICE_BASE / "data" / "output" / "heritage_voice_clones"

# --- Add necessary paths to sys.path for module discovery ---
# This ensures that `cosyvoice` and `cajun_lexicon` modules can be found.
if str(COSYVOICE_REPO_PATH) not in sys.path:
    sys.path.insert(0, str(COSYVOICE_REPO_PATH))
if str(COSYVOICE_REPO_PATH / "third_party" / "Matcha-TTS") not in sys.path:
    sys.path.insert(0, str(COSYVOICE_REPO_PATH / "third_party" / "Matcha-TTS"))
if str(VINTAGE_VOICE_BASE / "scripts" / "cajun8h") not in sys.path:
    sys.path.insert(0, str(VINTAGE_VOICE_BASE / "scripts" / "cajun8h"))

# Import CosyVoice2 and cajun_lexicon after paths are set
try:
    from cosyvoice.cli.cosyvoice import CosyVoice2
    from cajun_lexicon import respell
except ImportError as e:
    print(
        f"ERROR: Could not import CosyVoice2 or cajun_lexicon. "
        f"Ensure '{COSYVOICE_REPO_PATH}' and '{VINTAGE_VOICE_BASE / 'scripts' / 'cajun8h'}' "
        f"are correctly set up and contain the necessary modules. (Details: {e})",
        file=sys.stderr,
    )
    sys.exit(1)


def transcribe_audio_with_whisper(audio_path, device):
    """Transcribe an audio file using OpenAI Whisper."""
    print(f"  [Whisper] Transcribing reference audio with Whisper-tiny on {device}...", flush=True)
    try:
        # Using "tiny" model for faster transcription and lower VRAM usage, suitable for 8GB GPU.
        whisper_model = whisper.load_model("tiny", device=device)
        result = whisper_model.transcribe(str(audio_path), fp16=(device != "cpu"), verbose=False)
        return result["text"].strip()
    except Exception as e:
        print(f"  [Whisper] Error during transcription: {e}", file=sys.stderr, flush=True)
        return ""


def apply_loudnorm(input_path, output_path):
    """Applies loudness normalization using ffmpeg."""
    print(f"  [FFmpeg] Applying loudness normalization to {input_path.name}...", flush=True)
    tmp_path = output_path.with_suffix(".loud.wav")
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-af",
        "loudnorm=I=-12:LRA=7:TP=-1.0",  # Target -12 LUFS, typical for speech
        "-ar",
        "24000",
        "-ac",
        "1",
        str(tmp_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        shutil.move(str(tmp_path), str(output_path))
        print(f"  [FFmpeg] Loudness normalization complete. Output: {output_path.name}", flush=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  [FFmpeg] Loudnorm failed or ffmpeg not found: {e}", file=sys.stderr, flush=True)
        print(f"  [FFmpeg] Copying original audio to {output_path.name} instead.", file=sys.stderr, flush=True)
        shutil.copy(str(input_path), str(output_path))
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Clone your heritage voice with a Cajun French accent using CosyVoice2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ref-audio",
        type=Path,
        required=True,
        help="Path to a clean WAV file (5-15 seconds) of the speaker's voice to clone."
    )
    parser.add_argument(
        "--ref-text",
        type=str,
        default=None,
        help="Transcript of the --ref-audio. If not provided, Whisper will automatically transcribe it. Providing it manually is recommended for best results."
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="The Cajun French text to generate. Will be processed by the Cajun lexicon for accurate pronunciation."
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="heritage_voice_clone",
        help="Base name for the output WAV file (e.g., 'my_cajun_voice')."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Compute device to use. 'auto' tries CUDA/GPU if available, then CPU."
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.95,
        help="Playback speed of the generated speech. 1.0 is normal. 0.95 is often preferred for a natural pace with CosyVoice2."
    )
    args = parser.parse_args()

    # --- Initial Checks & Setup ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not args.ref_audio.exists():
        print(f"ERROR: Reference audio file not found: {args.ref_audio}", file=sys.stderr)
        sys.exit(1)
    if not args.ref_audio.is_file():
        print(f"ERROR: Reference audio path is not a file: {args.ref_audio}", file=sys.stderr)
        sys.exit(1)

    if not CAJUN_MODEL_PATH.exists():
        print(f"ERROR: Cajun fine-tuned model not found at {CAJUN_MODEL_PATH}.", file=sys.stderr)
        print("Please ensure the 'CosyVoice2-cajun-ep2' directory is present in 'vintage-voice/models/'.", file=sys.stderr)
        print("This model needs to be downloaded or trained beforehand for the demo to run.", file=sys.stderr)
        sys.exit(1)

    if not (COSYVOICE_REPO_PATH / "cosyvoice" / "cli" / "cosyvoice.py").exists():
        print(f"ERROR: CosyVoice2 repository not found at {COSYVOICE_REPO_PATH}.", file=sys.stderr)
        print("Please ensure 'models/cosyvoice-repo' is cloned into your VintageVoice directory.", file=sys.stderr)
        sys.exit(1)
    
    if not DEFAULT_COSYVOICE_REF_AUDIO.exists():
        print(f"WARNING: CosyVoice2's internal reference audio not found at {DEFAULT_COSYVOICE_REF_AUDIO}.", file=sys.stderr)
        print(f"         This file is usually part of the CosyVoice2 model release. Using user's --ref-audio for this internal reference.", file=sys.stderr)
        print(f"         This might affect overall speech style/prosody consistency.", file=sys.stderr)
        # Fallback if the standard CosyVoice2 ref is missing
        global DEFAULT_COSYVOICE_REF_AUDIO
        DEFAULT_COSYVOICE_REF_AUDIO = args.ref_audio


    # --- Device Configuration ---
    if args.device == "auto":
        on_gpu = torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "")
        device_str = "cuda" if on_gpu else "cpu"
    else:
        on_gpu = (args.device == "cuda")
        device_str = args.device

    print(f"\n--- Starting Heritage Voice Cloning Demo ---", flush=True)
    print(f"  Using device: {device_str} (GPU available: {torch.cuda.is_available()})", flush=True)

    # --- Step 1: Transcribe User's Reference Audio if needed ---
    # This transcript is crucial for CosyVoice2 to correctly align the user's reference audio
    # with its internal representations for timbre cloning.
    ref_audio_transcript_for_model = args.ref_text
    if ref_audio_transcript_for_model is None:
        print(f"\n[Step 1/4] Automatically transcribing user's reference audio: {args.ref_audio.name}", flush=True)
        ref_audio_transcript_for_model = transcribe_audio_with_whisper(args.ref_audio, device_str)
        if not ref_audio_transcript_for_model:
            print("WARNING: Could not auto-transcribe reference audio. Using a placeholder transcript.", file=sys.stderr)
            print("         This may severely affect voice cloning quality. Please provide --ref-text manually for best results.", file=sys.stderr)
            ref_audio_transcript_for_model = "a clear voice speaks"  # Minimal placeholder
        print(f"  [Whisper] User's reference audio transcript: '{ref_audio_transcript_for_model}'", flush=True)
    else:
        print(f"\n[Step 1/4] Using provided transcript for user's reference audio: '{ref_audio_transcript_for_model}'", flush=True)

    # --- Step 2: Prepare Cajun French Text ---
    print(f"\n[Step 2/4] Preparing Cajun French text for synthesis:", flush=True)
    print(f"  Original text: '{args.text}'", flush=True)
    processed_text = respell(args.text)  # Apply Cajun lexicon for phonetic accuracy
    print(f"  Lexicon-processed text: '{processed_text}'", flush=True)

    # --- Step 3: Load CosyVoice2 Model ---
    print(f"\n[Step 3/4] Loading CosyVoice2-cajun-ep2 model from {CAJUN_MODEL_PATH}...", flush=True)
    t0 = time.time()
    try:
        model = CosyVoice2(
            str(CAJUN_MODEL_PATH),
            load_jit=False,
            load_trt=False,
            fp16=on_gpu,  # Use FP16 if on GPU for memory efficiency, especially on 8GB GPUs
        )
    except Exception as e:
        print(f"ERROR: Failed to load CosyVoice2 model from {CAJUN_MODEL_PATH}: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"  Model loaded in {time.time()-t0:.1f}s", flush=True)

    # --- Step 4: Perform Inference and Save Raw Audio ---
    print(f"\n[Step 4/4] Generating speech in your heritage voice...", flush=True)
    temp_raw_path = OUTPUT_DIR / f"{args.output_name}_raw.wav"
    try:
        # CosyVoice2 inference_zero_shot signature:
        # (text_to_synthesize, transcript_of_ref_audio_for_timbre, path_to_ref_audio_for_timbre, ...)
        for i, o in enumerate(model.inference_zero_shot(
            processed_text,
            ref_audio_transcript_for_model,  # Transcript of the user's reference audio
            str(args.ref_audio),             # Path to the user's reference audio (for timbre cloning)
            stream=False,
            speed=args.speed
        )):
            if i == 0:  # Only take the first output for simplicity
                torchaudio.save(str(temp_raw_path), o["tts_speech"], model.sample_rate)
                print(f"  Raw speech saved temporarily to {temp_raw_path.name}", flush=True)
                break
    except Exception as e:
        print(f"ERROR: Speech generation failed: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        del model  # Free up model resources
        if on_gpu:
            torch.cuda.empty_cache()  # Clear CUDA cache if on GPU
        import gc; gc.collect()  # Run garbage collection


    # --- Step 5: Apply Loudness Normalization to Final Output ---
    print(f"\n[Step 5/5] Post-processing generated audio...", flush=True)
    output_audio_path = OUTPUT_DIR / f"{args.output_name}.wav"
    if not apply_loudnorm(temp_raw_path, output_audio_path):
        print("WARNING: Loudness normalization failed. Output audio might have inconsistent volume.", file=sys.stderr)

    # Clean up raw temporary file
    if temp_raw_path.exists():
        os.remove(temp_raw_path)

    print(f"\n--- Demo Complete! ---", flush=True)
    print(f"Final generated audio saved to: {output_audio_path}", flush=True)


if __name__ == "__main__":
    main()