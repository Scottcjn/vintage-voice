#!/usr/bin/env python3
"""
VintageVoice Demo: Clone Your Heritage Voice in About an Hour

This script demonstrates an end-to-end process for generating speech in your
own voice (timbre) with a specific heritage accent (like Cajun French),
using a pre-finetuned CosyVoice2 model. This process leverages CosyVoice2's
zero-shot capabilities:
  - Your provided reference audio supplies the unique timbre of your voice.
  - The pre-finetuned model (e.g., CosyVoice2-cajun-ep2) supplies the learned
    heritage accent, prosody, and language-specific pronunciations.

This approach allows for "finetuned voice out" (a voice adapted to a heritage
style) within the "about an hour" and "8GB GPU" constraints, as it focuses
on efficient inference rather than full model retraining.

Instructions:
1.  **Environment Setup**:
    *   Ensure you have Python 3.8+ and `ffmpeg` installed.
    *   Clone the `vintage-voice` repository:
        `git clone https://github.com/Scottcjn/vintage-voice.git`
        `cd vintage-voice`
    *   Set up a Python virtual environment and install dependencies. The `vintage-voice`
        project typically uses `venv-cosy` for CosyVoice2:
        `python -m venv venv-cosy`
        `source venv-cosy/bin/activate`
        `pip install -r requirements-cosyvoice.txt` (if such a file exists, otherwise install manually:
        `pip install torch torchaudio numpy` and ensure the `cosyvoice-repo` is cloned and its
        dependencies are met).
        `pip install openai-whisper` (optional, for transcribing your reference audio).
    *   Ensure the pre-finetuned CosyVoice2 models are downloaded and placed
        in the `models/` directory (e.g., `models/CosyVoice2-cajun-ep2`).
        Refer to the `vintage-voice` README for model download instructions.

2.  **Prepare Your Reference Audio**:
    *   Record a clean 5-15 second audio clip of your own voice. Speak clearly
        and steadily, with no background noise.
    *   Example phrase: "Hello, my name is [Your Name], and this is my heritage voice."
    *   Save this audio as a 24kHz mono WAV file. If your recording is in another
        format or sample rate, use `ffmpeg` to convert it:
        `ffmpeg -i /path/to/your_recording.mp3 -ar 24000 -ac 1 ./demo_data/my_voice_ref.wav`
    *   Update `YOUR_REF_AUDIO` and `YOUR_REF_TEXT` in the script below.

3.  **Transcribe Your Reference Audio (Optional but Recommended)**:
    *   Providing an exact transcript (`YOUR_REF_TEXT`) for your reference audio
        helps CosyVoice2 avoid auto-transcription via Whisper, which can sometimes
        introduce minor artifacts or "bleed" from the reference speaker at the
        beginning of the generated output.
    *   You can use `openai-whisper` to get a transcript:
        `whisper --model tiny --language en ./demo_data/my_voice_ref.wav`
    *   Copy the output text into the `YOUR_REF_TEXT` variable.

4.  **Configure Script Variables**:
    *   Adjust `YOUR_REF_AUDIO`, `YOUR_REF_TEXT`, `CAJUN_TEXT_TO_SAY`,
        and `OUTPUT_BASENAME` in the `--- USER CONFIGURATION ---` section below.
    *   The `MODEL_PATH` defaults to the Cajun French model. You can change
        this to another finetuned model if available (e.g., Irish English).

5.  **Run the Script**:
    *   Execute this script from the `vintage-voice` root directory:
        `python demo_heritage_voice_clone.py`
    *   If you want to explicitly use a GPU (e.g., CUDA device 0), run:
        `CUDA_VISIBLE_DEVICES=0 python demo_heritage_voice_clone.py`
        (Ensure other GPU-intensive processes, like LLMs, are stopped first).

Output:
A `.wav` file will be saved in `data/output/heritage_demo/` containing your
voice speaking the specified Cajun French text with the learned accent.
"""

import sys
import os
import subprocess
from pathlib import Path
import torch
import torchaudio

# --- PATH CONFIGURATION ---
# Determine BASE_DIR assuming this script is in the project root (vintage-voice/)
SCRIPT_DIR = Path(__file__).parent.resolve()
BASE_DIR = SCRIPT_DIR

# Add necessary project paths to sys.path for module imports
sys.path.insert(0, str(BASE_DIR / "models" / "cosyvoice-repo"))
sys.path.insert(0, str(BASE_DIR / "models" / "cosyvoice-repo" / "third_party" / "Matcha-TTS"))
sys.path.insert(0, str(BASE_DIR / "scripts" / "cajun8h")) # For cajun_lexicon

# Import CosyVoice2 CLI and Cajun lexicon respelling
from cosyvoice.cli.cosyvoice import CosyVoice2
from cajun_lexicon import respell # Louisiana pronunciation lexicon

# --- USER CONFIGURATION ---
# IMPORTANT: Update these paths and texts for your setup!

# Directory for user-provided demo data. Will be created if it doesn't exist.
DEMO_DATA_DIR = BASE_DIR / "demo_data"
DEMO_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Path to your personal reference audio (5-15 seconds, 24kHz mono WAV)
# This file must exist for the demo to run.
# Example: DEMO_DATA_DIR / "my_voice_ref.wav"
YOUR_REF_AUDIO = DEMO_DATA_DIR / "my_voice_ref.wav"
# If you don't have your own yet, for initial testing, you could temporarily use:
# YOUR_REF_AUDIO = BASE_DIR / "vintage-voice-samples-50ep" / "sophia_ref.wav"

# Exact transcript of your reference audio. Highly recommended for quality.
# Example: "Hello, my name is AI Engineer, and this is my heritage voice."
YOUR_REF_TEXT = "Hello, my name is AI Engineer, and this is my heritage voice."
# If using Sophia's ref for testing:
# YOUR_REF_TEXT = ("reporting from the Serengeti. Here at Elion Labs, we've successfully "
#                  "trapped and tagged approximately 15.5 of these majestic little creatures.")


# The text you want to generate in Cajun French.
# This text will be processed by the Cajun lexicon for accurate pronunciation.
CAJUN_TEXT_TO_SAY = "Mais comment ça va, cher? Ça fait longtemps que je t'ai pas vu."
# Another example: "Laissez les bons temps rouler ! On va faire un bon gombo ce soir, cher."
# Another example: "On va passer par Opelousas et l'Atchafalaya."

# Desired output filename (e.g., "my_cajun_voice.wav")
OUTPUT_BASENAME = "my_cajun_voice"

# --- MODEL CONFIGURATION ---
# Path to the pre-finetuned CosyVoice2 model for the target accent.
# This model provides the "heritage accent" (Cajun French in this example).
# Ensure this model directory exists and contains the necessary model files.
MODEL_PATH = BASE_DIR / "models" / "CosyVoice2-cajun-ep2"
# If you have an Irish accent model, you could use:
# MODEL_PATH = BASE_DIR / "models" / "CosyVoice2-irish"

# Output directory for generated audio clips.
OUTPUT_DIR = BASE_DIR / "data" / "output" / "heritage_demo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --- DEMO SCRIPT START ---

def setup_environment():
    """Checks for prerequisites and provides setup instructions if needed."""
    print("--- Environment Setup Check ---")

    # Check for CosyVoice2 model directory
    if not MODEL_PATH.exists():
        print(f"ERROR: CosyVoice2 model directory not found: '{MODEL_PATH}'")
        print("  Please ensure the pre-finetuned CosyVoice2 model is downloaded and placed here.")
        print("  Refer to the 'vintage-voice' README for model download instructions.")
        sys.exit(1)

    # Check for user's reference audio file
    if not YOUR_REF_AUDIO.exists():
        print(f"ERROR: Your reference audio file not found: '{YOUR_REF_AUDIO}'")
        print("\nACTION REQUIRED: Please record a clean 5-15 second WAV file of your voice.")
        print(f"  - Recommended phrase: \"{YOUR_REF_TEXT}\"")
        print(f"  - Save it as a 24kHz mono WAV at the path specified above: '{YOUR_REF_AUDIO}'")
        print("  - Example ffmpeg command to convert/resample audio:")
        print(f"    ffmpeg -i /path/to/your_recording.mp3 -ar 24000 -ac 1 '{YOUR_REF_AUDIO}'")
        sys.exit(1)

    # Check for ffmpeg installation (used for loudnorm)
    try:
        subprocess.run(["ffmpeg", "-h"], capture_output=True, check=True)
    except FileNotFoundError:
        print("ERROR: `ffmpeg` is not installed or not in your system's PATH.")
        print("  Please install `ffmpeg` (e.g., `sudo apt install ffmpeg` on Debian/Ubuntu, `brew install ffmpeg` on macOS).")
        sys.exit(1)

    print("Environment check passed. Proceeding with voice generation.\n")


def apply_loudnorm(audio_path: Path):
    """Applies loudness normalization to the generated audio for consistent volume."""
    tmp_path = audio_path.parent / (audio_path.name + ".loud.wav")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(audio_path),
        "-af", "loudnorm=I=-12:LRA=7:TP=-1.0", "-ar", "24000", "-ac", "1", str(tmp_path)
    ]
    
    print(f"  Applying loudnorm to improve output volume...", end="", flush=True)
    if subprocess.run(cmd).returncode == 0:
        os.replace(str(tmp_path), str(audio_path))
        print(" Done.")
    else:
        print(" FAILED.")
        print(f"  WARNING: Loudnorm failed for {audio_path}. Output volume might be inconsistent.")


def main():
    """Main function to run the heritage voice cloning demo."""
    setup_environment()

    # Determine if a CUDA-enabled GPU is available
    on_gpu = torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "")
    device_info = 'GPU' if on_gpu else 'CPU'
    print(f"Loading CosyVoice2 model for '{MODEL_PATH.name}' accent on {device_info}...", flush=True)

    # Load the CosyVoice2 model. fp16 (half-precision) is enabled on GPU for performance.
    try:
        model = CosyVoice2(str(MODEL_PATH), load_jit=False, load_trt=False, fp16=on_gpu)
    except Exception as e:
        print(f"ERROR: Failed to load CosyVoice2 model from {MODEL_PATH}: {e}")
        print("  Please check the model path and ensure all CosyVoice2 dependencies are met.")
        print("  If on GPU, try setting CUDA_VISIBLE_DEVICES='' to force CPU mode, or free GPU memory.")
        sys.exit(1)

    # Apply the Cajun lexicon respelling to the input text.
    # This helps CosyVoice2 pronounce specific Cajun French words and place-names correctly.
    processed_text = respell(CAJUN_TEXT_TO_SAY)
    
    output_audio_path = OUTPUT_DIR / f"{OUTPUT_BASENAME}.wav"

    print(f"\n--- Generating Heritage Voice ---")
    print(f"  Your Reference Audio:    '{YOUR_REF_AUDIO.name}'")
    print(f"  Your Reference Text:     '{YOUR_REF_TEXT}'")
    print(f"  Input Text (Original):   '{CAJUN_TEXT_TO_SAY}'")
    print(f"  Input Text (Processed):  '{processed_text}'")
    print(f"  Output will be saved to: '{output_audio_path}'")

    # Perform zero-shot inference.
    # Your voice (from YOUR_REF_AUDIO) is cloned, and the text is spoken
    # with the accent/prosody learned by the MODEL_PATH (e.g., Cajun French).
    try:
        # CosyVoice2's inference_zero_shot can yield multiple outputs if stream=True.
        # Here, we typically expect one complete speech segment.
        for i, o in enumerate(model.inference_zero_shot(processed_text, YOUR_REF_TEXT, str(YOUR_REF_AUDIO), stream=False, speed=0.95)):
            torchaudio.save(str(output_audio_path), o["tts_speech"], model.sample_rate)
            break # Only save the first generated speech segment
    except Exception as e:
        print(f"ERROR: Voice generation failed: {e}")
        print("  Common issues:")
        print("  - CUDA out of memory: Try freeing GPU resources, reducing batch size (if applicable), or running on CPU (CUDA_VISIBLE_DEVICES='').")
        print("  - Invalid reference audio/text: Ensure `YOUR_REF_AUDIO` is a valid 24kHz mono WAV and `YOUR_REF_TEXT` is its accurate transcript.")
        print("  - Model loading issues: Check model path and file integrity.")
        sys.exit(1)

    # Apply loudness normalization for a more consistent listening experience.
    apply_loudnorm(output_audio_path)
    
    print(f"\n--- Generation Complete ---")
    print(f"Successfully generated your heritage voice clip: '{output_audio_path}'")
    print(f"You can now listen to it and share with community language workers!\n")

    # Clean up model from GPU/CPU memory to free resources.
    del model
    if on_gpu:
        torch.cuda.empty_cache()
    import gc; gc.collect()


if __name__ == "__main__":
    main()