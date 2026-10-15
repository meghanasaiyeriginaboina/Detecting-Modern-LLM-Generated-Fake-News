from pathlib import Path
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory
import detector as D

app = Flask(__name__)
OUT = D.OUT
MIN_WORDS = 80   # below this, the notebook's length-robustness results show lower reliability

def _clean(d):
    text = (d.get("text") or "").strip()
    title = (d.get("title") or "").strip()
    return title, text

@app.get("/")
def index():
    return render_template("index.html", models=D.MODEL_NAMES)

@app.post("/api/detect")
def api_detect():
    d = request.get_json(force=True)
    title, text = _clean(d)
    model = d.get("model", D.MODEL_NAMES[0])
    if len(text.split()) < 5:
        return jsonify(error="Please paste at least a few sentences of article text."), 400
    if model not in D.MODEL_NAMES:
        return jsonify(error="Unknown model."), 400
    res = D.detect(text, title, model)
    res["word_count"] = len(text.split())
    res["short_text_warning"] = res["word_count"] < MIN_WORDS
    res["explanation"] = D.explain(text, title, model)
    return jsonify(res)

@app.post("/api/compare")
def api_compare():
    title, text = _clean(request.get_json(force=True))
    if len(text.split()) < 5:
        return jsonify(error="Please paste at least a few sentences of article text."), 400
    results = [D.detect(text, title, m) for m in D.MODEL_NAMES]
    fake_votes = sum(r["verdict"] == "FAKE" for r in results)
    return jsonify(results=results, fake_votes=fake_votes, total=len(results),
                   word_count=len(text.split()), short_text_warning=len(text.split()) < MIN_WORDS)

@app.get("/api/results")
def api_results():
    tdir, fdir = OUT / "tables", OUT / "figures"
    tables = {}
    if tdir.exists():
        for f in sorted(tdir.glob("*.csv")):
            df = pd.read_csv(f).round(4)
            tables[f.stem] = {"columns": list(df.columns), "rows": df.astype(object).where(df.notna(), "").values.tolist()}
    figs = sorted(p.name for p in fdir.glob("*.png")) if fdir.exists() else []
    return jsonify(tables=tables, figures=figs)

@app.get("/outputs/figures/<path:name>")
def figure(name):
    return send_from_directory(OUT / "figures", name)

if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)
