(() => {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const h = (tag, attrs = {}, ...kids) => {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === false || v == null) continue;
      if (k === "class") el.className = v;
      else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v === true ? "" : v);
    }
    for (const kid of kids.flat()) if (kid != null && kid !== false) el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
    return el;
  };
  const store = {
    get(k) { try { return sessionStorage.getItem(k) || ""; } catch { return ""; } },
    set(k, v) { try { sessionStorage.setItem(k, v); } catch { /* storage unavailable: fine */ } },
  };

  const S = { status: null, job: null, result: null, selected: new Set(), dismissed: new Set(), timer: null, botsLoaded: false };

  async function api(path, opts = {}) {
    let res;
    try { res = await fetch(path, opts); } catch { throw new Error("Could not reach the server. Check that it is still running."); }
    let body = null;
    try { body = await res.json(); } catch { /* not json */ }
    if (!res.ok) throw new Error((body && body.error) || `The server answered ${res.status}.`);
    return body;
  }

  function banner(msg, kind) {
    const b = $("#banner");
    if (!msg) { b.hidden = true; return; }
    b.textContent = msg; b.className = "banner" + (kind ? " " + kind : ""); b.hidden = false;
  }

  /* ---------- formatting ---------- */
  const nice = (p) => String(p || "").replace(/_/g, " ");
  const yn = (v) => (v === true ? "Yes" : v === false ? "No" : String(v));
  const num = (v) => (typeof v === "number" ? String(Math.round(v * 100) / 100) : String(v));
  const withUnit = (v, unit) => [typeof v === "boolean" ? yn(v) : num(v), unit].filter(Boolean).join(" ");
  const OPS = { "<=": "at most", ">=": "at least", "==": "exactly" };

  function sentence(f) {
    const d = f.drift;
    if (!d.has_setting) return f.reason;
    const cur = d.current, unit = d.unit;
    if (d.operator === "contains") {
      const list = (Array.isArray(cur) ? cur : [cur]).join(", ") || "nothing";
      const has = (Array.isArray(cur) ? cur : [cur]).some((x) => same(x, d.required));
      return has ? `The circular allows ${d.required}, and this bot's list already includes it (${list}).`
                 : `The circular now allows ${d.required}. This bot's list has ${list} and does not include it.`;
    }
    if (typeof d.required === "boolean") {
      return `${nice(d.parameter)} is ${yn(cur)}. The circular requires ${yn(d.required)}.`;
    }
    return `${nice(d.parameter)} is ${withUnit(cur, unit)}. The circular requires ${OPS[d.operator] || d.operator} ${withUnit(d.required, unit)}.`;
  }
  function same(a, b) {
    const n = (x) => String(x).toLowerCase().replace(/[^a-z0-9 ]/g, " ").split(/\s+/).filter(Boolean).join(" ");
    const sg = (w) => (w.length > 3 && w.endsWith("s") && !w.endsWith("ss") ? w.slice(0, -1) : w);
    const A = n(a).split(" ").map(sg), B = n(b).split(" ").map(sg);
    if (A.join(" ") === B.join(" ")) return true;
    const stop = new Set(["of", "the", "and", "a", "an", "for", "in"]);
    const acro = (t) => t.filter((w) => !stop.has(w)).map((w) => w[0]).join("");
    return (A.length === 1 && B.length > 1 && A[0] === acro(B)) || (B.length === 1 && A.length > 1 && B[0] === acro(A));
  }
  const sevWord = (s) => (s === "OK" ? "Passed" : s);

  /* ---------- gauge: the setting drawn against the rule ---------- */
  function gauge(f) {
    const d = f.drift;
    if (!d.has_setting) return h("p", { class: "nosetting" }, "This bot has no matching setting. The verdict comes from its description, so check it by hand.");
    const g = h("div", { class: "gauge" });
    const bad = f.severity !== "OK";
    if (d.operator === "contains") {
      const list = Array.isArray(d.current) ? d.current : [d.current];
      const has = list.some((x) => same(x, d.required));
      g.append(h("div", { class: "pair" },
        ...list.map((x) => h("span", { class: "pill" }, x)),
        h("span", { class: "pill " + (has ? "hit" : "miss") }, has ? `${d.required} included` : `${d.required} missing`)));
      g.append(h("span", { class: "cap" }, "List in the bot, then the item the circular adds"));
      return g;
    }
    if (typeof d.required === "boolean" || typeof d.current === "boolean") {
      const mismatch = d.current !== d.required;
      g.append(h("div", { class: "pair" },
        h("span", { class: "pill" + (mismatch && bad ? " bad" : "") }, `Bot: ${yn(d.current)}`),
        h("span", { class: "pill" }, `Circular: ${yn(d.required)}`)));
      return g;
    }
    const cur = Number(d.current), req = Number(d.required);
    if (!isFinite(cur) || !isFinite(req)) {
      g.append(h("div", { class: "pair" }, h("span", { class: "pill" }, `Bot: ${d.current}`), h("span", { class: "pill" }, `Circular: ${d.required}`)));
      return g;
    }
    const max = Math.max(Math.abs(cur), Math.abs(req), 1) * 1.25;
    const pos = (v) => Math.max(0, Math.min(100, (v / max) * 100));
    const track = h("div", { class: "track" });
    if (d.operator === "<=") track.append(h("div", { class: "allowed", style: `left:0;width:${pos(req)}%` }));
    if (d.operator === ">=") track.append(h("div", { class: "allowed", style: `left:${pos(req)}%;right:0` }));
    const breach = bad && f.status === "violating";
    if (breach) {
      const a = pos(req), b = pos(cur), w = Math.abs(a - b);
      const gap = h("div", { class: "gap" });
      gap.style.setProperty("--w", w + "%");
      if (b >= a) gap.style.left = a + "%"; else gap.style.right = (100 - a) + "%";
      track.append(gap);
    }
    track.append(h("div", { class: "tick", style: `left:${pos(req)}%` }));
    track.append(h("div", { class: "dot", style: `left:${pos(cur)}%` }));
    g.append(track, h("div", { class: "legend-line" },
      h("span", {}, "Bot ", h("b", {}, withUnit(cur, d.unit))),
      h("span", {}, "Rule ", h("b", {}, `${OPS[d.operator] || d.operator} ${withUnit(req, d.unit)}`))));
    return g;
  }

  /* ---------- rows ---------- */
  function row(f, selectable) {
    const open = { v: false };
    const li = h("li", { class: "row", "data-sev": f.severity, "data-id": f.id });
    const pick = h("div", { class: "pick" });
    if (selectable) {
      const cb = h("input", { type: "checkbox", "aria-label": `Include ${f.bot}, ${f.clause} in the tickets`, onchange: () => {
        cb.checked ? S.selected.add(f.id) : S.selected.delete(f.id); updateBar();
      } });
      cb.checked = S.selected.has(f.id);
      pick.append(cb);
    }
    const who = h("div", { class: "who" }, h("span", { class: "sev" }, sevWord(f.severity)), h("span", { class: "bot" }, f.bot), h("span", { class: "plat" }, f.platform));
    const toggle = h("button", { class: "linkish more", type: "button", "aria-expanded": "false", onclick: () => {
      open.v = !open.v; li.classList.toggle("open", open.v); toggle.setAttribute("aria-expanded", String(open.v)); toggle.textContent = open.v ? "Hide details" : "Show details";
    } }, "Show details");
    const what = h("div", { class: "what" },
      h("span", { class: "clause" }, f.clause),
      h("p", { class: "prose" }, sentence(f)),
      f.in_jira ? h("p", { class: "note" }, "A ticket for this already exists in Jira.") : null,
      toggle);
    const g = gauge(f);

    const dl = h("dl", {},
      h("dt", {}, "Setting checked"), h("dd", {}, f.drift.has_setting ? nice(f.drift.parameter) : `${nice(f.drift.parameter)} (this bot has none)`),
      h("dt", {}, "Kind of rule"), h("dd", {}, f.drift.modality === "permissive" ? "Permitted, not required" : "Required"),
      h("dt", {}, "Confidence"), h("dd", {}, Math.round((f.confidence || 0) * 100) + "%"),
      f.ticket_summary ? [h("dt", {}, "Ticket title"), h("dd", {}, f.ticket_summary)] : null);
    const detail = h("div", { class: "detail" },
      h("blockquote", {}, f.clause_text.length > 700 ? f.clause_text.slice(0, 700) + "…" : f.clause_text), dl);
    if (selectable && S.status.writes) {
      detail.append(h("button", { class: "linkish", type: "button", onclick: async (e) => {
        try {
          await api(`/api/jobs/${S.job}/feedback`, { method: "POST", headers: authHeaders({ "Content-Type": "application/json" }), body: JSON.stringify({ id: f.id, decision: "reject" }) });
          li.classList.add("dismissed"); S.selected.delete(f.id); S.dismissed.add(f.id);
          const cb = $("input", li); if (cb) { cb.checked = false; cb.disabled = true; }
          e.target.textContent = "Marked as not a real issue"; e.target.disabled = true; updateBar();
        } catch (er) { banner(er.message); }
      } }, "This is not a real issue"));
    }
    li.append(pick, who, what, g, detail);
    return li;
  }

  /* ---------- results ---------- */
  function renderResults(r, seconds) {
    const act = r.findings.filter((f) => f.severity !== "OK");
    const pass = r.findings.filter((f) => f.severity === "OK");
    const count = (s) => r.findings.filter((f) => f.severity === s).length;
    const ref = r.circular_ref || S.source || "this circular";
    const highBots = new Set(r.findings.filter((f) => f.severity === "High").map((f) => f.bot)).size;
    let title, sub;
    if (!r.obligations) {
      title = `No checkable rules found in ${ref}`;
      sub = S.status.llm ? "The circular may state its rules in a way that has no number, deadline or yes/no condition to compare." : "Without a Groq key RegDrift only matches a few keywords. Add GROQ_API_KEY to read real circulars.";
    } else if (highBots) {
      title = `${highBots} ${highBots === 1 ? "bot is" : "bots are"} out of line with ${ref}`;
    } else if (act.length) {
      title = `No bot breaks ${ref}, but ${act.length} need${act.length === 1 ? "s" : ""} a look`;
    } else {
      title = `Every bot checked is in line with ${ref}`;
    }
    if (!sub) {
      const by = r.extractor === "lora" ? "the tuned local model" : r.extractor === "llm" ? r.model : "keyword matching";
      sub = `Checked ${r.obligations} ${r.obligations === 1 ? "rule" : "rules"} against ${r.bots_checked} bots in ${seconds} s. Rules were read by ${by}.` +
        (r.not_applicable ? ` ${r.not_applicable} bot and rule ${r.not_applicable === 1 ? "pair was" : "pairs were"} skipped as not related.` : "");
    }
    $("#r-title").textContent = title; $("#r-sub").textContent = sub;

    const total = r.findings.length;
    const split = $("#split"), legend = $("#legend");
    split.replaceChildren(); legend.replaceChildren();
    split.hidden = legend.hidden = !total;
    const cols = { High: "var(--high)", Medium: "var(--medium)", Low: "var(--low)", OK: "var(--ok)" };
    const names = { High: "High", Medium: "Medium", Low: "Low", OK: "Passed" };
    const parts = [];
    for (const s of ["High", "Medium", "Low", "OK"]) {
      const n = count(s); if (!n) continue;
      parts.push(`${n} ${names[s].toLowerCase()}`);
      const seg = h("span", { style: `background:${cols[s]};flex-grow:0` }); split.append(seg);
      requestAnimationFrame(() => requestAnimationFrame(() => { seg.style.flexGrow = String(n); }));
      legend.append(h("li", {}, h("i", { style: `background:${cols[s]}` }), `${n} ${names[s]}`));
    }
    split.setAttribute("aria-label", parts.join(", "));

    S.selected = new Set(act.filter((f) => !f.in_jira).map((f) => f.id)); S.dismissed = new Set();
    const ledger = $("#ledger"); ledger.replaceChildren(...act.map((f) => row(f, true)));
    $(".ledger-head").hidden = !act.length; ledger.hidden = !act.length;
    const clean = $("#clean");
    clean.hidden = !!act.length || !r.obligations;
    clean.textContent = "Nothing needs action. Passed checks are listed below.";
    const passed = $("#passed");
    passed.hidden = !pass.length;
    $("#passed-sum").textContent = `${pass.length} passed ${pass.length === 1 ? "check" : "checks"}`;
    $("#passed-list").replaceChildren(...pass.map((f) => row(f, false)));
    $("#intake").hidden = true; $("#progress").hidden = true; $("#results").hidden = false;
    updateBar();
    requestAnimationFrame(() => requestAnimationFrame(() => document.querySelectorAll(".gauge").forEach((g) => g.classList.add("in"))));
    window.scrollTo({ top: 0 });
  }

  /* ---------- action bar ---------- */
  function authHeaders(extra = {}) {
    const code = $("#code").value || store.get("rd_code");
    return code ? { ...extra, "X-RegDrift-Token": code } : extra;
  }
  function updateBar() {
    const act = S.result ? S.result.findings.filter((f) => f.severity !== "OK") : [];
    const bar = $("#bar");
    bar.hidden = !act.length;
    if (!act.length) return;
    const n = S.selected.size, live = S.status.jira && S.status.writes;
    $("#bar-text").textContent = `${n} of ${act.length} selected`;
    const btn = $("#file-btn");
    btn.disabled = n === 0;
    btn.textContent = live ? `Create ${n} Jira ${n === 1 ? "ticket" : "tickets"}` : `Preview ${n} ${n === 1 ? "ticket" : "tickets"}`;
    $("#code-wrap").hidden = !(live && S.status.needs_code);
  }
  async function fileTickets() {
    const btn = $("#file-btn"); btn.disabled = true;
    try {
      const code = $("#code").value; if (code) store.set("rd_code", code);
      const out = await api(`/api/jobs/${S.job}/tickets`, { method: "POST", headers: authHeaders({ "Content-Type": "application/json" }), body: JSON.stringify({ ids: [...S.selected] }) });
      const dlg = $("#dlg");
      $("#dlg-title").textContent = out.live ? "Jira tickets" : "Ticket preview";
      $("#dlg-note").textContent = out.live ? "These are the results from Jira." : "Jira is not connected here, so nothing was created. These are the tickets that would be filed.";
      const label = { created: "Created", exists: "Already in Jira, skipped", failed: "Failed", preview: "Would be created" };
      $("#dlg-list").replaceChildren(...out.tickets.map((t) => h("li", {}, h("span", { class: "st " + t.state }, label[t.state] + (t.key ? ` as ${t.key}` : "")), h("span", {}, t.summary), t.error ? h("span", { class: "note" }, t.error) : null)));
      dlg.showModal ? dlg.showModal() : alert(out.tickets.map((t) => t.summary).join("\n"));
      banner("");
    } catch (e) { banner(e.message); } finally { updateBar(); }
  }

  /* ---------- analysis flow ---------- */
  function showProgress(source) {
    S.source = source;
    $("#intake").hidden = true; $("#results").hidden = true; $("#progress").hidden = false; $("#bar").hidden = true;
    $("#progress-title").textContent = `Analysing ${source}`;
    paintStages(0, "queued");
  }
  function paintStages(stage, status) {
    $("#stages").replaceChildren(...S.status.stages.map((name, i) => {
      const st = i + 1 < stage ? "done" : i + 1 === stage ? "now" : "";
      return h("li", { class: st }, h("i"), name);
    }));
    if (status === "queued") $("#log").textContent = "Waiting for the analysis to start.";
  }
  async function begin(source, options) {
    banner("");
    showProgress(source);
    try {
      const { job } = await api("/api/analyze", options);
      S.job = job; poll();
    } catch (e) { banner(e.message); reset(); }
  }
  function poll() {
    clearInterval(S.timer);
    S.timer = setInterval(async () => {
      try {
        const j = await api(`/api/jobs/${S.job}`);
        if (j.status === "done") { clearInterval(S.timer); S.result = j.result; renderResults(j.result, j.seconds); }
        else if (j.status === "error") { clearInterval(S.timer); banner(j.error); reset(); }
        else { paintStages(j.stage + 1, j.status); if (j.log.length) $("#log").textContent = j.log[j.log.length - 1]; }
      } catch (e) { clearInterval(S.timer); banner(e.message); reset(); }
    }, 800);
  }
  function reset() {
    clearInterval(S.timer); S.result = null; S.job = null; $("#file").value = "";
    $("#intake").hidden = false; $("#progress").hidden = true; $("#results").hidden = true; $("#bar").hidden = true;
  }

  /* ---------- views ---------- */
  function renderBots(list) {
    $("#bots").replaceChildren(...list.map((b) => {
      const entries = Object.entries(b.parameters || {});
      return h("li", {}, h("div", {}, h("span", { class: "n" }, b.name), h("span", { class: "p" }, b.platform)),
        h("p", {}, b.description),
        entries.length ? h("dl", {}, ...entries.flatMap(([k, v]) => [h("dt", {}, nice(k)), h("dd", {}, Array.isArray(v) ? v.join(", ") : yn(v))]))
                       : h("span", { class: "none" }, "No settings recorded, so rules cannot be compared directly."));
    }));
  }
  async function route() {
    const bots = location.hash.startsWith("#/bots");
    $("#view-review").hidden = bots; $("#view-bots").hidden = !bots;
    if (bots) $("#bar").hidden = true; else updateBar();
    document.querySelectorAll("nav a").forEach((a) => a.toggleAttribute("aria-current", a.dataset.nav === (bots ? "bots" : "review")));
    document.querySelectorAll("nav a").forEach((a) => { if (a.hasAttribute("aria-current")) a.setAttribute("aria-current", "page"); });
    if (bots && !S.botsLoaded) {
      try { renderBots(await api("/api/bots")); S.botsLoaded = true; } catch (e) { banner(e.message); }
    }
  }

  async function init() {
    try { S.status = await api("/api/status"); } catch (e) { banner(e.message); return; }
    const st = S.status;
    $("#env").replaceChildren(
      st.llm ? h("span", {}, "Rules read by ", h("b", {}, st.model)) : h("span", {}, h("b", {}, "Keyword matching only")),
      " ", h("span", {}, st.jira ? "Jira connected." : "Jira not connected."));
    if (!st.llm) banner("No Groq key is set, so rules are found by keyword matching and real circulars will mostly show nothing. Set GROQ_API_KEY for full extraction.", "info");
    $("#code").value = store.get("rd_code");
    const samples = $("#samples");
    samples.replaceChildren(...(st.circulars.length ? st.circulars.map((n) => h("li", {}, h("span", { class: "name" }, n),
      h("button", { class: "ghost", type: "button", onclick: () => begin(n, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ circular: n }) }) }, "Analyse")))
      : [h("li", { class: "none" }, "No circulars in the circulars folder yet. Upload one on the left.")]));

    const drop = $("#drop"), input = $("#file");
    const send = (file) => {
      if (!file) return;
      const fd = new FormData(); fd.append("file", file);
      begin(file.name, { method: "POST", body: fd });
    };
    input.addEventListener("change", () => send(input.files[0]));
    ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
    ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
    drop.addEventListener("drop", (e) => send(e.dataTransfer.files[0]));
    $("#again").addEventListener("click", reset);
    $("#file-btn").addEventListener("click", fileTickets);
    window.addEventListener("hashchange", route);
    route();
  }
  init();
})();
