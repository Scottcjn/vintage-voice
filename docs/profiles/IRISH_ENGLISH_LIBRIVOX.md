# Irish English LibriVox Profile

This is a small, rights-clear seed profile for an English-language Irish accent.
It adds a reference clip, a profile config, and a path for growing the seed into
an 8 hour finetune corpus following the Cajun 8h recipe.

This is **not** an Irish/Gaeilge language model and it does not use endangered
or Indigenous language data. Any future Irish-language or heritage-language work
must follow `docs/ENDANGERED_LANGUAGES.md` and be community-led.

## Source And License

| Field | Value |
|---|---|
| Collection | LibriVox Dialect and Accent Collection Vol. 1 |
| Archive item | https://archive.org/details/dialect_accent_0909_librivox |
| Source file | `dialectaccent_vol_01_02poh.mp3` |
| Source title | `Irish accent` |
| Reader metadata | `read by Padraig` |
| Source comment | `Recorded by Padraig in an Irish accent` |
| License | Public Domain |
| Included clip | `refs/irish_english_librivox_ref.wav` |

The included clip is the first 12 seconds of the source MP3, converted to 24 kHz
mono PCM WAV for use as an F5-TTS/CosyVoice reference.

```bash
curl -L -o /tmp/dialectaccent_vol_01_02poh.mp3 \
  https://archive.org/download/dialect_accent_0909_librivox/dialectaccent_vol_01_02poh.mp3

ffmpeg -y -ss 0 -t 12 -i /tmp/dialectaccent_vol_01_02poh.mp3 \
  -ac 1 -ar 24000 -sample_fmt s16 refs/irish_english_librivox_ref.wav
```

## Profile Files

- `configs/profiles/irish_english_librivox.yaml` records the source, license,
  clip parameters, rights gate, and target finetune plan.
- `configs/presets.yaml` registers the profile metadata.
- `scripts/generate.py` registers the preset as `irish_english`.

## 8 Hour Finetune Path

Follow the same shape as `docs/CAJUN_8H_FINETUNE.md`:

1. Gather only public-domain or explicitly consented Irish English recordings.
2. Keep a manifest row for every segment with source URL, license, speaker or
   reader, and consent/public-domain evidence.
3. Normalize source audio to 24 kHz mono WAV.
4. Split into 5-15 second segments and drop noisy or non-speech spans.
5. Transcribe and manually review the English transcript.
6. Train from `configs/cosyvoice2_8gb.yaml`, starting with the LLM and then the
   flow model once the corpus is clean.
7. Use `refs/irish_english_librivox_ref.wav` as the seed reference clip for
   smoke tests while a larger reference set is curated.

Smoke-test generation once a compatible checkpoint is available:

```bash
python scripts/generate.py \
  "Good evening from Dublin, and welcome to the programme." \
  --preset irish_english \
  --model path/to/model.pt \
  --vocab path/to/vocab.txt \
  --output data/output/irish_english_smoke.wav
```
