#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
VintageVoice — Louisiana Heritage Audio Downloader
Downloads public domain Cajun French, Creole, and Louisiana accent recordings.

CRITICALLY ENDANGERED: Cajun French has ~150,000 speakers, mostly elderly.
Louisiana Creole has ~10,000 speakers. These voices are disappearing NOW.
"""
import subprocess
import os
import sys
import json
import shutil
import urllib.request
import urllib.parse

# Override with VINTAGE_VOICE_DATA when the 18TB machine (.136) is offline
AUDIO_BASE = os.environ.get("VINTAGE_VOICE_DATA", "/mnt/18tb/louisiana_voice")

# Stop downloading if free space on the target drops below this (GB)
MIN_FREE_GB = int(os.environ.get("VINTAGE_VOICE_MIN_FREE_GB", "60"))

COLLECTIONS = {
    "cajun_french": {
        "query": 'subject:"cajun" AND mediatype:audio AND (subject:"french" OR subject:"louisiana")',
        "dir": "cajun_french",
        "limit": 200,
        "desc": "Cajun French speakers — severely endangered",
    },
    "louisiana_creole": {
        "query": 'subject:"creole" AND subject:"louisiana" AND mediatype:audio',
        "dir": "louisiana_creole",
        "limit": 100,
        "desc": "Louisiana Creole — critically endangered",
    },
    "cajun_music_spoken": {
        "query": 'subject:"cajun" AND subject:"music" AND mediatype:audio AND year:[1950 TO 1990]',
        "dir": "cajun_music",
        "limit": 200,
        "desc": "Cajun music with spoken intros/interviews",
    },
    "lomax_louisiana": {
        "query": 'creator:"Lomax" AND subject:"Louisiana" AND mediatype:audio',
        "dir": "lomax_louisiana",
        "limit": 100,
        "desc": "Alan Lomax field recordings — Louisiana",
    },
    "louisiana_folklife": {
        "query": 'subject:"Louisiana folklife" AND mediatype:audio',
        "dir": "louisiana_folklife",
        "limit": 100,
        "desc": "Louisiana folklife recordings",
    },
    "french_louisiana": {
        "query": 'subject:"French" AND subject:"Louisiana" AND mediatype:audio',
        "dir": "french_louisiana",
        "limit": 200,
        "desc": "French language recordings from Louisiana",
    },
    "new_orleans_oral": {
        "query": '(subject:"New Orleans" OR subject:"new orleans") AND mediatype:audio AND (subject:"oral history" OR subject:"interview")',
        "dir": "new_orleans_oral",
        "limit": 100,
        "desc": "New Orleans oral histories — Yat accent, Creole",
    },
    "zydeco": {
        "query": 'subject:"zydeco" AND mediatype:audio',
        "dir": "zydeco",
        "limit": 100,
        "desc": "Zydeco — Creole French musical tradition",
    },
    "acadian": {
        "query": 'subject:"Acadian" AND mediatype:audio',
        "dir": "acadian",
        "limit": 100,
        "desc": "Acadian heritage recordings",
    },
    "swamp_pop": {
        "query": 'subject:"swamp pop" AND mediatype:audio',
        "dir": "swamp_pop",
        "limit": 50,
        "desc": "Swamp pop — South Louisiana rock/R&B with Cajun influence",
    },
}


def search_archive(query, limit=50, retries=5):
    params = urllib.parse.urlencode({
        "q": query, "output": "json", "rows": limit,
        "fl[]": "identifier,title,year",
    })
    url = f"https://archive.org/advancedsearch.php?{params}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read())
                return data.get("response", {}).get("docs", [])
        except Exception as e:
            wait = 30 * (attempt + 1)
            print(f"  Search error (attempt {attempt+1}/{retries}): {e} — retrying in {wait}s")
            import time
            time.sleep(wait)
    print(f"  Search FAILED after {retries} attempts: {query}")
    return []


def download_item(identifier, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    cmd = [
        "wget", "-q", "-r", "-l1", "-nd",
        "-A", "*.mp3,*.ogg,*.wav,*.flac",
        "-P", dest_dir,
        f"https://archive.org/download/{identifier}/",
    ]
    try:
        subprocess.run(cmd, timeout=300, capture_output=True)
    except subprocess.TimeoutExpired:
        print(f"    Timeout: {identifier}")


def free_gb(path):
    return shutil.disk_usage(path).free / 1e9


def main():
    # Optional: limit to specific collections, e.g.
    #   download_louisiana.py cajun_french lomax_louisiana
    selected = sys.argv[1:] or list(COLLECTIONS.keys())
    unknown = [s for s in selected if s not in COLLECTIONS]
    if unknown:
        print(f"Unknown collections: {unknown}")
        print(f"Available: {', '.join(COLLECTIONS.keys())}")
        sys.exit(1)

    total = 0
    for name in selected:
        cfg = COLLECTIONS[name]
        dest = os.path.join(AUDIO_BASE, cfg["dir"])
        os.makedirs(dest, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Collection: {name} — {cfg['desc']}")

        items = search_archive(cfg["query"], cfg["limit"])
        print(f"Found {len(items)} items")

        for i, item in enumerate(items):
            if free_gb(AUDIO_BASE) < MIN_FREE_GB:
                print(f"\nSTOP: free space below {MIN_FREE_GB}GB on {AUDIO_BASE}")
                print(f"Total items downloaded: {total}")
                sys.exit(2)

            ident = item.get("identifier", "")
            title = item.get("title", "?")[:60]
            year = item.get("year", "?")

            item_dir = os.path.join(dest, ident)
            if os.path.exists(item_dir) and os.listdir(item_dir):
                continue

            print(f"  [{i+1}/{len(items)}] {title} ({year})")
            download_item(ident, item_dir)
            total += 1

    print(f"\n{'='*60}")
    print(f"Total items downloaded: {total}")
    os.system(f"du -sh {AUDIO_BASE}")


if __name__ == "__main__":
    main()
