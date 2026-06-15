#!/bin/bash
export PATH="/home/scott/audiocraft-venv/bin:$PATH"
cd /home/scott/vintage-voice
OUT="data/raw/youtube/cajun_fr"
ARCH="$OUT/.downloaded.txt"
C="-x --audio-format mp3 --audio-quality 5 --download-archive $ARCH -o $OUT/%(channel)s__%(title).80s__%(id)s.%(ext)s --restrict-filenames --ignore-errors --no-warnings"
echo "### Charrer-Veiller (all 8 eps, ~5.6h pure French)"
yt-dlp $C "https://www.youtube.com/channel/UCctqFlJdXDQmjqH3iN25xrw/videos" 2>&1 | grep -iE 'destination|already|ERROR' | tail -16
echo "### KVPI La Tasse de Café"
yt-dlp $C "ytsearch3:KVPI La Tasse De Cafe français" 2>&1 | grep -iE 'destination|already|ERROR' | tail -6
echo "### Télé-Louisiane (cap first 150 videos)"
yt-dlp $C --playlist-end 150 "https://www.youtube.com/channel/UCnha23C2RAYKRTEXfgdEPnQ/videos" 2>&1 | grep -iE 'destination|already|ERROR' | tail -30
echo "=== PULL2 DONE ==="
echo "total mp3s now: $(ls $OUT/*.mp3 2>/dev/null | wc -l)"; du -sh $OUT
