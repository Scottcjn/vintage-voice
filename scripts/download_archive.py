#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
VintageVoice — Archive.org Public Domain Audio Downloader
Downloads historical recordings for TTS training data.
"""
import argparse
import subprocess
import os
import json
import urllib.request
import urllib.parse

COLLECTIONS = {
    "old_time_radio": {
        "query": "collection:oldtimeradio AND mediatype:audio AND year:[1930 TO 1955]",
        "desc": "Golden age radio dramas — peak transatlantic delivery",
    },
    "newsreels": {
        "query": "collection:prelinger AND mediatype:movies AND year:[1930 TO 1955]",
        "desc": "Prelinger Archive newsreels — narrator voice gold",
    },
    "fdr": {
        "query": "creator:Roosevelt AND mediatype:audio AND year:[1933 TO 1945]",
        "desc": "FDR Fireside Chats and speeches",
    },
    "ww2": {
        "query": 'subject:"world war" AND mediatype:audio AND year:[1939 TO 1945]',
        "desc": "WWII era speeches and broadcasts",
    },
    "edison": {
        "query": "collection:edison AND mediatype:audio",
        "desc": "Edison cylinder recordings — oldest recorded humans",
    },
    "radio_commercials": {
        "query": 'subject:"radio commercial" AND mediatype:audio AND year:[1930 TO 1960]',
        "desc": "Vintage radio ads — trained announcer voices",
    },
}


def search_archive(query, limit=50):
    params = urllib.parse.urlencode({
        "q": query, "output": "json", "rows": limit,
        "fl[]": "identifier,title,year",
    })
    url = f"https://archive.org/advancedsearch.php?{params}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read())
            return data.get("response", {}).get("docs", [])
    except Exception as e:
        print(f"  Search error: {e}")
        return []


def download_item(identifier, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    cmd = [
        "wget", "-q", "-r", "-l1", "-nd",
        "-A", "*.mp3,*.ogg,*.wav",
        "-P", dest_dir,
        f"https://archive.org/download/{identifier}/",
    ]
    try:
        subprocess.run(cmd, timeout=180, capture_output=True)
    except subprocess.TimeoutExpired:
        print(f"    Timeout: {identifier}")


def main():
    parser = argparse.ArgumentParser(description="Download vintage audio from Archive.org")
    parser.add_argument("--collection", choices=list(COLLECTIONS.keys()) + ["all"], default="all")
    parser.add_argument("--limit", type=int, default=50, help="Max items per collection")
    parser.add_argument("--output", default="data/raw", help="Output directory")
    args = parser.parse_args()

    collections = COLLECTIONS if args.collection == "all" else {args.collection: COLLECTIONS[args.collection]}
    total = 0

    for name, cfg in collections.items():
        dest = os.path.join(args.output, name)
        os.makedirs(dest, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Collection: {name} — {cfg['desc']}")

        items = search_archive(cfg["query"], args.limit)
        print(f"Found {len(items)} items")

        for i, item in enumerate(items):
            ident = item.get("identifier", "")
            title = item.get("title", "?")[:60]
            year = item.get("year", "?")

            item_dir = os.path.join(dest, ident)
            if os.path.exists(item_dir) and os.listdir(item_dir):
                continue

            print(f"  [{i+1}/{len(items)}] {title} ({year})")
            download_item(ident, item_dir)
            total += 1

    print(f"\nDone. {total} items downloaded.")


if __name__ == "__main__":
    main()
