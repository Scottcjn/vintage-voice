#!/usr/bin/env bash
# VintageVoice — Edison Cylinder preset (1888-1920s)
#
# Post-process ffmpeg chain that transforms clean VintageVoice TTS output
# into audio that sounds as if it were recorded onto an Edison wax cylinder.
#
# Usage: ./edison.sh <input.wav> <output.wav>
#
# Rationale for each filter stage:
#
#   1. bandpass 300-3500 Hz
#        Edison cylinders had a narrow acoustic-horn pickup. The 300 Hz floor
#        comes from horn geometry; the 3500 Hz ceiling comes from the wax's
#        inability to hold fine groove detail.
#   2. acompressor threshold=-25 ratio=3
#        Acoustic-horn recording is inherently compressed — loud peaks carve
#        the wax deeper than it can hold, so peaks clip. Simulated here.
#   3. vibrato f=2.5 d=0.005
#        Cylinder rotation was hand-cranked or spring-driven with minor speed
#        variance. Wobble period ~0.4s (2.5 Hz), depth ~0.5% (d=0.005).
#   4. pink-noise mix-in at ~25% weight (bandpass-filtered)
#        Constant surface noise from the wax compound — a gentle hiss
#        that never stops, even during pauses.
#   5. tremolo f=0.3 d=0.1
#        Very slow dynamic ripple mimicking uneven cylinder rotation friction.
#
# References: IASA-TC 04 (Technical Committee on Preservation), Library of
# Congress National Audio-Visual Conservation Center technical notes on wax
# cylinder digitization artifacts.

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <input.wav> <output.wav>" >&2
  exit 1
fi

IN="$1"
OUT="$2"

# Duration of input (used to size the noise source)
DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$IN")

ffmpeg -y -i "$IN" \
  -f lavfi -t "$DUR" -i "anoisesrc=a=0.07:c=pink:d=${DUR}" \
  -f lavfi -t "$DUR" -i "anoisesrc=a=0.04:c=brown:d=${DUR}" \
  -filter_complex "
    [0:a]
      highpass=f=400:poles=2,
      lowpass=f=2800:poles=2,
      acompressor=threshold=-22dB:ratio=4:attack=5:release=80,
      vibrato=f=4.2:d=0.012,
      aemphasis=mode=production:type=75kf,
      atempo=0.98
    [voice];
    [1:a]
      highpass=f=500:poles=2,
      lowpass=f=2800:poles=2
    [hiss];
    [2:a]
      highpass=f=80:poles=2,
      lowpass=f=400:poles=2
    [rumble];
    [voice][hiss][rumble]amix=inputs=3:duration=first:weights='1.0 0.55 0.35',
      tremolo=f=0.6:d=0.18,
      acrusher=level_in=1:level_out=1:bits=8:mix=0.35:mode=lin:aa=1,
      alimiter=limit=0.85
    [out]
  " -map "[out]" -ar 24000 -ac 1 -c:a pcm_s16le "$OUT"

echo "Edison preset applied: $IN -> $OUT"
