const $ = s => document.querySelector(s);
const SAMPLES = {
  real: ["Senate passes budget bill after weeks of negotiation",
    "The Senate voted 62-38 on Thursday to approve a federal budget bill, ending weeks of negotiation between the two parties. Lawmakers said the measure funds government agencies through the end of the fiscal year and includes modest increases for infrastructure and education. The bill now moves to the House, where a vote is expected next week. Officials from both parties described the compromise as imperfect but necessary to avoid a shutdown."],
  fake: ["SHOCKING: Secret memo PROVES they are hiding the truth from you!!!",
    "You will NOT believe what insiders just leaked! The mainstream media is desperately trying to cover up the biggest scandal in history, and they do not want you to see this. Sources close to the story say the elites have been lying to all of us for years. Share this before it gets deleted! Wake up, people, the truth is finally coming out and nobody can stop it now!!!"]
};

document.querySelectorAll(".tab").forEach(b => b.onclick = () => {
  document.querySelectorAll(".tab").forEach(x => x.classList.toggle("active", x === b));
  document.querySelectorAll(".panel").forEach(p => p.classList.toggle("active", p.id === b.dataset.tab));
  if (b.dataset.tab === "results") loadResults();
});

const text = $("#text"), title = $("#title");
const wc = () => (text.value.trim().match(/\S+/g) || []).length;
text.oninput = () => $("#wc").textContent = wc() + " words";
$("#ex-real").onclick = () => load(SAMPLES.real);
$("#ex-fake").onclick = () => load(SAMPLES.fake);
$("#clear").onclick = () => { title.value = text.value = ""; text.oninput(); };
function load([t, x]) { title.value = t; text.value = x; text.oninput(); }

async function post(url, body) {
  const r = await fetch(url, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || "Request failed");
  return j;
}
const pct = p => Math.round(p * 100);
const col = p => p > .5 ? "var(--fake)" : "var(--real)";
function esc(s) { return String(s).replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c])); }

$("#go").onclick = async () => {
  const btn = $("#go"), err = $("#err"); err.hidden = true; btn.disabled = true; btn.textContent = "Analyzing…";
  try {
    const r = await post("/api/detect", {title: title.value, text: text.value, model: $("#model").value});
    $("#resPlaceholder").hidden = true; $("#resBody").hidden = false;
    const arc = $("#gArc"); arc.setAttribute("stroke-dasharray", `${r.fake_probability * 100} 100`); arc.style.stroke = col(r.fake_probability);
    $("#gPct").textContent = pct(r.fake_probability) + "%";
    const v = $("#verdict"); v.textContent = r.verdict; v.className = "verdict " + r.verdict.toLowerCase();
    const w = $("#warn"); w.hidden = !r.short_text_warning;
    w.textContent = `Only ${r.word_count} words: short texts are less reliable for these models.`;
    const ex = r.explanation; $("#expl").hidden = !ex;
    if (ex) {
      const li = a => a.length ? a.map(o => `<li><span>${esc(o.word)}</span><span class="muted">${o.weight.toFixed(2)}</span></li>`).join("") : "<li class='muted'>none</li>";
      $("#exFake").innerHTML = li(ex.fake); $("#exReal").innerHTML = li(ex.real);
    }
  } catch (e) { err.textContent = e.message; err.hidden = false; }
  btn.disabled = false; btn.textContent = "Analyze";
};

$("#goCmp").onclick = async () => {
  const err = $("#cmpErr"); err.hidden = true; $("#cmpBody").hidden = true;
  try {
    const r = await post("/api/compare", {title: title.value, text: text.value});
    $("#cmpSummary").textContent = `${r.fake_votes} of ${r.total} models say FAKE` + (r.short_text_warning ? " (short text: lower reliability)" : "");
    $("#cmpBars").innerHTML = r.results.map(m => `<div class="bar"><span>${esc(m.model)}</span>
      <div class="track"><div class="fill" style="width:${pct(m.fake_probability)}%;background:${col(m.fake_probability)}"></div></div>
      <span class="pill ${m.verdict.toLowerCase()}">${m.verdict} ${pct(m.fake_probability)}%</span></div>`).join("");
    $("#cmpBody").hidden = false;
  } catch (e) { err.textContent = e.message + " (enter text in the Detector tab first)"; err.hidden = false; }
};

let loaded = false;
async function loadResults() {
  if (loaded) return; loaded = true;
  const j = await (await fetch("/api/results")).json(), el = $("#resultsBody"); el.className = ""; el.innerHTML = "";
  for (const [name, t] of Object.entries(j.tables)) {
    el.insertAdjacentHTML("beforeend", `<h3>${esc(name)}</h3><div class="tw"><table><thead><tr>${t.columns.map(c => `<th>${esc(c)}</th>`).join("")}</tr></thead><tbody>${
      t.rows.map(r => `<tr>${r.map(c => `<td>${esc(c)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`);
  }
  if (j.figures.length) el.insertAdjacentHTML("beforeend", `<h3>Figures</h3><div class="figs">${j.figures.map(f => `<figure><img loading="lazy" src="/outputs/figures/${encodeURIComponent(f)}" alt="${esc(f)}"><figcaption>${esc(f)}</figcaption></figure>`).join("")}</div>`);
  if (!el.innerHTML) el.textContent = "No tables or figures found in outputs/.";
}
