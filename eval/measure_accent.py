#!/usr/bin/env python3
"""Accent shift: does prairie-ep1 sit acoustically CLOSER to the real St. Landry
prairie speakers than ep2? Both models clone the same Sophia reference, so the
speaker-timbre component is ~constant between them — the *difference* in distance
to the real-prairie corpus reflects the learned prosodic/phonetic prior (the accent).

Features per clip: f0 mean/std, F1/F2/F3 (Praat Burg), spectral centroid, MFCC(13).
Compares group centroids in z-scored space + per-clip distance distributions."""
import sys, os, csv, glob, random, warnings
warnings.filterwarnings("ignore")
import numpy as np, librosa, parselmouth
from scipy import stats
random.seed(13)
BASE = "/home/scott/vintage-voice"

def feats(path):
    try:
        y, sr = librosa.load(path, sr=16000)
        if len(y) < sr*0.4: return None
        snd = parselmouth.Sound(y, sampling_frequency=sr)
        pitch = snd.to_pitch(time_step=0.01)
        f0 = pitch.selected_array['frequency']; f0v = f0[f0 > 0]
        form = snd.to_formant_burg(time_step=0.01, max_number_of_formants=5, maximum_formant=5500)
        ts = form.ts()
        F = {1: [], 2: [], 3: []}
        for t in ts:
            if pitch.get_value_at_time(t) and pitch.get_value_at_time(t) > 0:
                for n in (1, 2, 3):
                    v = form.get_value_at_time(n, t)
                    if v and not np.isnan(v): F[n].append(v)
        cent = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
        vec = [float(np.mean(f0v)) if len(f0v) else 0.0,
               float(np.std(f0v)) if len(f0v) else 0.0,
               float(np.mean(F[1])) if F[1] else 0.0,
               float(np.mean(F[2])) if F[2] else 0.0,
               float(np.mean(F[3])) if F[3] else 0.0,
               cent, *[float(x) for x in mfcc]]
        return vec
    except Exception:
        return None

NAMES = ["f0_mean","f0_std","F1","F2","F3","centroid"] + [f"mfcc{i}" for i in range(1,14)]

def group(paths, label, limit=None):
    if limit and len(paths) > limit: paths = random.sample(paths, limit)
    X = [feats(p) for p in paths]; X = [v for v in X if v]
    print(f"  {label}: {len(X)} clips featured", flush=True)
    return np.array(X)

def main():
    rows = list(csv.DictReader(open(f"{BASE}/eval/out/manifest.csv")))
    ep2  = [r["path"] for r in rows if r["tag"]=="ep2"]
    prai = [r["path"] for r in rows if r["tag"]=="prairie"]
    real = glob.glob(f"{BASE}/data/processed/prairie/segments_clean/*.wav")
    print("extracting features...", flush=True)
    R = group(real, "real-prairie (target)", limit=150)
    E = group(ep2,  "ep2 output")
    P = group(prai, "prairie-ep1 output")
    # z-score on pooled
    allX = np.vstack([R, E, P]); mu = allX.mean(0); sd = allX.std(0); sd[sd==0]=1
    Rz, Ez, Pz = (R-mu)/sd, (E-mu)/sd, (P-mu)/sd
    Rc = Rz.mean(0)
    dE = np.linalg.norm(Ez - Rc, axis=1)   # per-clip distance ep2 -> real centroid
    dP = np.linalg.norm(Pz - Rc, axis=1)
    print("\n=== ACCENT DISTANCE TO REAL ST. LANDRY PRAIRIE (z-scored, lower=closer) ===")
    print(f"  ep2         mean dist {dE.mean():.3f}  (±{dE.std():.3f}, n={len(dE)})")
    print(f"  prairie-ep1 mean dist {dP.mean():.3f}  (±{dP.std():.3f}, n={len(dP)})")
    closer = "prairie-ep1 CLOSER" if dP.mean() < dE.mean() else "ep2 closer (no prairie gain)"
    print(f"  -> {closer}  (Δ={dE.mean()-dP.mean():+.3f})")
    try:
        u, p = stats.mannwhitneyu(dE, dP, alternative="two-sided")
        print(f"  Mann-Whitney p={p:.4f}  ({'significant' if p<0.05 else 'NOT significant'} at .05)")
    except Exception as e:
        print("  stat test failed:", e)
    # which features moved toward real (F1/F2 vowel space is the linguistic signal)
    print("\n  per-feature: ep2 vs prairie mean, and real target (z-scored)")
    print(f"  {'feat':9} {'ep2':>7} {'prairie':>8} {'real':>7} {'prairie→real?':>14}")
    for i,n in enumerate(NAMES):
        e,p_,r = Ez[:,i].mean(), Pz[:,i].mean(), Rc[i]
        toward = abs(p_-r) < abs(e-r)
        print(f"  {n:9} {e:7.2f} {p_:8.2f} {r:7.2f} {'  yes' if toward else '   no':>14}")

if __name__ == "__main__":
    main()
