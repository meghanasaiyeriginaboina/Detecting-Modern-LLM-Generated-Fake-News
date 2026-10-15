"""Detector backend: replicates the notebook pipeline (scrub -> TF-IDF / stylometric -> model)."""
import json, re
from pathlib import Path
import numpy as np, pandas as pd, joblib

OUT = Path(__file__).parent / "outputs"

# ---- scrub (same as notebook 1) ----
_pf = OUT / "data" / "scrub_patterns.json"
_pat = json.loads(_pf.read_text()) if _pf.exists() else []
LEAK_RE = re.compile("|".join(_pat), flags=re.I) if _pat else None

def scrub(s):
    return re.sub(r"\s+", " ", LEAK_RE.sub(" ", s)).strip() if LEAK_RE else s

# ---- stylometric features (same as notebook 2) ----
def stylometric(df, title_col="title", text_col="text"):
    t, x = df[text_col].astype(str), df[title_col].astype(str)
    nw = t.str.split().str.len().clip(lower=1); nc = t.str.len().clip(lower=1)
    f = pd.DataFrame(index=df.index)
    f["n_words"] = nw; f["n_chars"] = nc; f["avg_word_len"] = nc / nw
    f["n_sent"] = t.str.count(r"[.!?]+").clip(lower=1); f["avg_sent_len"] = nw / f["n_sent"]
    f["upper_ratio"] = t.str.count(r"[A-Z]") / nc; f["caps_word_ratio"] = t.str.count(r"\b[A-Z]{3,}\b") / nw
    f["exclam"] = t.str.count("!") / nw; f["question"] = t.str.count(r"\?") / nw
    f["quote"] = t.str.count(r"[\"“”']") / nw; f["digit_ratio"] = t.str.count(r"\d") / nc
    f["punct_ratio"] = t.str.count(r"[^\w\s]") / nc
    f["ttr"] = t.map(lambda s: len(set(s.lower().split())) / max(1, len(s.split())))
    f["title_words"] = x.str.split().str.len(); f["title_upper"] = x.str.count(r"[A-Z]") / x.str.len().clip(lower=1)
    f["title_excl"] = x.str.count("!")
    return f.astype(np.float32)

# ---- load models once ----
B = joblib.load(OUT / "models" / "phase2_models.joblib")
TFIDF, MODELS, KINDS = B["tfidf"], B["models"], B["kinds"]
MODEL_NAMES = list(MODELS)

def _fake_prob(model, X):
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(X)[0, 0])          # classes_ = [0 (fake), 1 (real)]
    return float(1 / (1 + np.exp(model.decision_function(X)[0])))  # LinearSVC: positive = real

def _features(kind, title, text):
    if kind == "style":
        return stylometric(pd.DataFrame({"title": [title], "text": [text]}))
    return TFIDF.transform([scrub((title.strip() + ". " if title else "") + text)])

def detect(text, title="", model="LogReg (TF-IDF)"):
    kind = KINDS[model]
    p = _fake_prob(MODELS[model], _features(kind, title, text))
    return {"model": model, "fake_probability": round(p, 4), "verdict": "FAKE" if p > 0.5 else "REAL"}

def explain(text, title="", model="LogReg (TF-IDF)", k=10):
    """Words pushing the score toward fake / real (linear TF-IDF models only)."""
    m = MODELS[model]
    if KINDS[model] != "tfidf" or not hasattr(m, "coef_"):
        return None
    X = _features("tfidf", title, text)
    contrib = X.multiply(m.coef_[0]).tocoo()      # coef > 0 -> REAL, < 0 -> FAKE
    names = TFIDF.get_feature_names_out()
    items = sorted(((names[j], float(v)) for j, v in zip(contrib.col, contrib.data)), key=lambda z: z[1])
    fake = [{"word": w, "weight": round(-v, 4)} for w, v in items if v < 0][:k]
    real = [{"word": w, "weight": round(v, 4)} for w, v in reversed(items) if v > 0][:k]
    return {"fake": fake, "real": real}
