#!/usr/bin/env python3
"""Build a self-contained, blinded native-speaker A/B rating sheet for ULL/CODOFIL.
Two comparisons per sentence: (1) ep2 vs prairie-ep1 (accent), (2) lexicon on vs off.
Clips copied locally so the folder is portable (zip + email). Ratings export to CSV."""
import csv, os, shutil, html, random
random.seed(21)
BASE = "/home/scott/vintage-voice"
OUT = f"{BASE}/eval/ab_sheet"; CLIPS = f"{OUT}/clips"
os.makedirs(CLIPS, exist_ok=True)

rows = list(csv.DictReader(open(f"{BASE}/eval/out/manifest.csv")))
by = {}
for r in rows:
    by[(r["tag"], r["id"])] = r
ids = [r["id"] for r in rows if r["tag"] == "ep2"]
texts = {r["id"]: r["orig_text"] for r in rows if r["tag"] == "ep2"}

def cp(tag, sid):
    src = by.get((tag, sid), {}).get("path", "")
    if src and os.path.exists(src):
        dst = f"{CLIPS}/{tag}__{sid}.wav"; shutil.copy(src, dst)
        return f"clips/{tag}__{sid}.wav"
    return ""

items = []  # each: id, text, two comparisons with blinded A/B + true mapping
for sid in ids:
    cmp1 = [("ep2", cp("ep2", sid)), ("prairie", cp("prairie", sid))]
    cmp2 = [("ep2", cp("ep2", sid)), ("ep2_noresp", cp("ep2_noresp", sid))]
    random.shuffle(cmp1); random.shuffle(cmp2)
    items.append({"id": sid, "text": texts[sid], "accent": cmp1, "lexicon": cmp2})

def block(item):
    sid = item["id"]; txt = html.escape(item["text"])
    def pair(kind, pair_):
        (la, pa), (lb, pb) = pair_
        return f'''
      <div class="cmp" data-item="{sid}" data-kind="{kind}" data-a="{la}" data-b="{lb}">
        <div class="q">{kind.upper()}: which sounds more <b>{"authentically Cajun / Opelousas" if kind=="accent" else "natural & correct"}</b>?</div>
        <div class="players">
          <div>A <audio controls src="{pa}"></audio></div>
          <div>B <audio controls src="{pb}"></audio></div>
        </div>
        <label><input type="radio" name="{sid}_{kind}" value="A">A better</label>
        <label><input type="radio" name="{sid}_{kind}" value="B">B better</label>
        <label><input type="radio" name="{sid}_{kind}" value="same">about the same</label>
      </div>'''
    return f'''
    <div class="item">
      <div class="txt">“{txt}”</div>
      {pair("accent", item["accent"])}
      {pair("lexicon", item["lexicon"])}
    </div>'''

rows_html = "\n".join(block(it) for it in items)
HTML = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Sophia Cajun French — Native-Speaker A/B</title>
<style>
 body{{font-family:system-ui,sans-serif;max-width:820px;margin:2rem auto;padding:0 1rem;line-height:1.5;color:#222}}
 h1{{color:#9a6a1a}} .item{{border-top:2px solid #e8d8b8;padding:1.2rem 0}}
 .txt{{font-size:1.15rem;font-style:italic;margin-bottom:.6rem}}
 .cmp{{background:#faf6ee;border-radius:8px;padding:.8rem 1rem;margin:.6rem 0}}
 .q{{font-size:.95rem;margin-bottom:.4rem}} .players{{display:flex;gap:1.5rem;margin:.4rem 0}}
 audio{{height:34px;vertical-align:middle}} label{{margin-right:1rem;font-size:.92rem}}
 button{{background:#9a6a1a;color:#fff;border:0;padding:.7rem 1.4rem;border-radius:6px;font-size:1rem;cursor:pointer}}
 .hdr{{background:#fdf8ee;border:1px solid #e8d8b8;border-radius:8px;padding:1rem 1.2rem}}
</style></head><body>
<h1>🎙️ Sophia — Cajun French listening test</h1>
<div class="hdr">
<p>Merci, cher! You're helping evaluate an AI Cajun French voice. For each line you'll
hear two short clips, <b>A</b> and <b>B</b>. Pick which sounds more authentically
Cajun (and, second question, more natural/correct). It's blinded on purpose —
there are no wrong answers, just your ear. Takes ~10 minutes.</p>
<p>Name/parish (optional): <input id="rater" style="padding:.3rem;width:16rem"></p>
</div>
{rows_html}
<div class="item"><button onclick="dl()">⬇️ Download my ratings (CSV)</button>
<p style="font-size:.85rem;color:#666">Then email the CSV back to scott@elyanlabs.ai. Merci beaucoup!</p></div>
<script>
function dl(){{
 var rater=document.getElementById('rater').value||'anon';
 var out=[['rater','sentence','kind','A_model','B_model','choice']];
 document.querySelectorAll('.cmp').forEach(function(c){{
   var id=c.dataset.item,k=c.dataset.kind;
   var sel=c.querySelector('input[name="'+id+'_'+k+'"]:checked');
   out.push([rater,id,k,c.dataset.a,c.dataset.b,sel?sel.value:'']);
 }});
 var csv=out.map(r=>r.map(x=>'"'+(''+x).replace(/"/g,'""')+'"').join(',')).join('\\n');
 var a=document.createElement('a');
 a.href='data:text/csv;charset=utf-8,'+encodeURIComponent(csv);
 a.download='cajun_ab_'+rater.replace(/\\W+/g,'_')+'.csv'; a.click();
}}
</script></body></html>'''
open(f"{OUT}/index.html", "w").write(HTML)
print(f"A/B sheet -> {OUT}/index.html  ({len(items)} sentences, {len(os.listdir(CLIPS))} clips)")
print("portable: zip the ab_sheet/ folder and send; blinding map is embedded in the CSV export (A_model/B_model).")
