#!/usr/bin/env python3
"""
ltx_lipsync.py — headless talking-head LIPSYNC from a still image + an external
audio clip, using the LTX-2.3 image+audio-to-video (ia2v) ComfyUI workflow.

The audio drives the mouth directly (LoadAudio -> TrimAudioDuration ->
LTXVAudioVAEEncode into the conditioned latent) — true lip-sync, not a mouth
pasted on afterward. Pairs with the Irish/Cajun CosyVoice2 voices in this repo:
generate a line with irish_say.py, then animate any portrait to speak it.

WHY A PATCHER (read this once)
------------------------------
The official LTX-2.3 ia2v workflow is a collapsed ComfyUI *subgraph*. ComfyUI's
frontend expands it to a flat API graph via app.graphToPrompt(), which is what
the server actually executes. Two practical snags this script handles:

  1) The workflow uses the custom node `ComfyMathExpression` to compute four
     derived constants (latent width = W/2, height = H/2, fps, and
     length = dur*fps+1). If that node pack isn't installed on the server, those
     nodes serialize as class_type=undefined with an "UNKNOWN" expression key.
     Since the inputs are constants, we resolve them to literals and delete the
     nodes — no custom-node install required.
  2) LTX latent video length must be of the form 8k+1; we snap it from the audio
     duration (keep clips < ~4 s — lip-sync drifts on longer takes).

ONE-TIME CAPTURE (produces the --src file)
------------------------------------------
Open ComfyUI in a browser, then in its devtools console (or via Playwright):

    const app = window.app;
    const wf = await (await fetch(
      '/api/userdata/workflows%2F<your_ia2v_workflow>.json')).json();
    await app.loadGraphData(wf, true, false, 'ia2v');
    const p = await app.graphToPrompt();
    copy(JSON.stringify(p.output));   // save as ia2v_graphToPrompt.json

That capture only needs redoing if the workflow itself changes. Node IDs below
are resolved structurally (by class_type / "UNKNOWN" key), not hard-coded, so a
re-capture with different IDs still works.

USAGE
-----
    python3 ltx_lipsync.py --image portrait.png --audio line.wav --audio-dur 3.46 \
        --prefix my_clip --prompt-file motion.txt
Then POST the emitted JSON to your ComfyUI /prompt endpoint (that POST is itself
the validator — HTTP 400 + node_errors means a bad graph, no GPU burned).
"""
import argparse, json, math

DEFAULT_PROMPT = (
    "A person speaks directly to the camera, lips moving naturally and clearly, "
    "with gentle expressive facial movements, small natural head motion, blinking "
    "and breathing. Static camera, head-and-shoulders portrait, photoreal cinematic "
    "lighting.\n\naction: speaking directly to camera with natural lip movement\n"
    "camera: fixed on character, head-and-shoulders framing"
)


def die(msg):
    raise SystemExit(f"[ltx_lipsync] ERROR: {msg}")


def snap_length(dur_s, fps):
    """LTX latent frame count: smallest 8k+1 >= dur*fps (>= audio, no last-word clip)."""
    return 8 * max(math.ceil((dur_s * fps - 1) / 8), 1) + 1


def main():
    ap = argparse.ArgumentParser(description="Build a headless LTX-2.3 ia2v lipsync prompt.")
    ap.add_argument("--src", default="ia2v_graphToPrompt.json",
                    help="graphToPrompt capture (a double-encoded JSON string)")
    ap.add_argument("--out", default="ltx_ia2v_prompt.json")
    ap.add_argument("--image", required=True, help="filename staged in ComfyUI/input/")
    ap.add_argument("--audio", required=True, help="wav filename staged in ComfyUI/input/")
    ap.add_argument("--audio-dur", type=float, required=True, help="seconds")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--prefix", default="lipsync")
    ap.add_argument("--prompt-file", default=None)
    a = ap.parse_args()

    prompt_text = open(a.prompt_file).read() if a.prompt_file else DEFAULT_PROMPT
    length = snap_length(a.audio_dur, a.fps)
    dur_s = round(length / a.fps, 3)

    g = json.loads(open(a.src).read())
    if isinstance(g, str):          # graphToPrompt output is a double-encoded JSON string
        g = json.loads(g)
    if not isinstance(g, dict):
        die(f"{a.src} did not decode to a node dict")

    # the missing-custom-node math nodes: class_type undefined + an "UNKNOWN" expression key
    math_ids = {nid for nid, n in g.items()
                if isinstance(n, dict)
                and n.get("class_type") in (None, "undefined")
                and "UNKNOWN" in (n.get("inputs") or {})}

    def const_of(ref):
        src = g.get(ref[0], {})
        if src.get("class_type") not in ("PrimitiveInt", "PrimitiveFloat", "PrimitiveBoolean"):
            die(f"math input {ref} is not a constant Primitive ({src.get('class_type')})")
        return src["inputs"]["value"]

    resolved = {}
    for mid in math_ids:
        ins = g[mid]["inputs"]
        expr = ins["UNKNOWN"].replace(" ", "")
        vals = {k.split(".")[-1]: const_of(v) for k, v in ins.items()
                if isinstance(v, list) and len(v) == 2}
        if expr == "a/2":
            resolved[mid] = int(vals["a"] // 2)
        elif expr == "a":
            resolved[mid] = int(vals["a"])
        elif expr == "a*b+1":
            resolved[mid] = int(length)
        else:
            die(f"unhandled math expr '{expr}' on {mid}")

    # inline resolved literals into every consumer, then delete the math nodes
    for n in g.values():
        if not isinstance(n, dict):
            continue
        for k, v in list((n.get("inputs") or {}).items()):
            if isinstance(v, list) and len(v) == 2 and v[0] in math_ids:
                n["inputs"][k] = resolved[v[0]]
    for mid in math_ids:
        g.pop(mid, None)

    # set the render inputs by node role (resolve IDs structurally, not hard-coded)
    def one(cls):
        ids = [nid for nid, n in g.items() if isinstance(n, dict) and n.get("class_type") == cls]
        if len(ids) != 1:
            die(f"expected exactly one {cls} node, found {len(ids)}")
        return ids[0]

    g[one("LoadImage")]["inputs"]["image"] = a.image
    la = g[one("LoadAudio")]["inputs"]; la["audio"] = a.audio; la["audioUI"] = ""
    g[one("PrimitiveStringMultiline")]["inputs"]["value"] = prompt_text
    g[one("TrimAudioDuration")]                              # presence check
    # the float feeding TrimAudioDuration.duration:
    trim = g[one("TrimAudioDuration")]["inputs"].get("duration")
    if isinstance(trim, list):
        g[trim[0]]["inputs"]["value"] = dur_s
    g[one("SaveVideo")]["inputs"]["filename_prefix"] = a.prefix

    # invariants (real exceptions, not asserts — survive python -O)
    for nid, n in g.items():
        if not isinstance(n, dict):
            continue
        if n.get("class_type") in (None, "undefined") or "UNKNOWN" in (n.get("inputs") or {}):
            die(f"node {nid} still broken after patch")
        for k, v in (n.get("inputs") or {}).items():
            if isinstance(v, list) and len(v) == 2 and v[0] in math_ids:
                die(f"dangling math ref {nid}.{k}")

    json.dump(g, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}: {len(g)} nodes | image={a.image} audio={a.audio} "
          f"length={length} ({dur_s}s) fps={a.fps} prefix={a.prefix}")


if __name__ == "__main__":
    main()
