"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const S = { status: null, samples: [], bots: [], input: null, job: null, result: null,
              decisions: {}, tickets: {}, filter: "all", code: "", running: false };

  /* ---------- tiny DOM builder (text only, so nothing from the server is ever parsed as HTML) ---------- */
  function h(tag, props, ...kids) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(props || {})) {
      if (v == null || v === false) continue;
      if (k === "class") el.className = v;
      else if (k === "text") el.textContent = v;
      else if (k === "style") Object.assign(el.style, v);
      else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v === true ? "" : v);
    }
    for (const kid of kids.flat(Infinity)) if (kid != null && kid !== false) el.append(kid.nodeType ? kid : document.createTextNode(kid));
    return el;
  }
  const clear = (el) => { while (el.firstChild) el.removeChild(el.firstChild); return el; };
  const api = async (url, opts) => {
    const r = await fetch(url, opts);
    let body = null;
    try { body = await r.json(); } catch (_) { /* empty body */ }
    if (!r.ok) throw new Error((body && body.error) || "Something went wrong. Try again.");
    return body;
  };

  /* ---------- wording ---------- */
  const OPS = { "<=": "at most", ">=": "at least", "==": "exactly", contains: "includes" };
  const SEV = { violating: "High", at_risk: "Medium", optional_gap: "Low", compliant: "Compliant" };
  const unitText = (u, v) => (u ? " " + (Number(v) === 1 && u.length > 3 && /s$/.test(u) ? u.slice(0, -1) : u) : "");
  const fmt = (v) => (typeof v === "boolean" ? (v ? "On" : "Off") : Array.isArray(v) ? v.join(", ") : String(v));
  const ruleShort = (o) => `${OPS[o.operator] || o.operator} ${fmt(o.required_value)}${unitText(o.unit, o.required_value)}`;
  const findingRule = (f) => ruleShort({ operator: f.operator, required_value: f.required, unit: f.unit });

  /* ---------- the drift line ---------- */
  function labelAt(pct, cls, children) {
    const el = h("div", { class: "d-lab " + cls }, children);
    if (pct < 14) el.style.left = "0";
    else if (pct > 86) el.style.right = "0";
    else { el.style.left = pct + "%"; el.style.transform = "translateX(-50%)"; }
    return el;
  }

  function driftLine(f) {
    if (f.basis !== "setting") {
      return h("span", { class: "judged", text: "No matching setting. Judged from the bot's description." });
    }
    const op = f.operator, req = f.required, cur = f.current;
    if (op === "contains") {
      const list = Array.isArray(cur) ? cur : [cur];
      const present = f.status === "compliant";
      return h("div", { class: "chips" },
        h("span", { class: "lab", text: "Bot's list" }),
        list.map((v) => h("span", { class: "pill", text: String(v) })),
        h("span", { class: "pill " + (present ? "have" : "need"), text: String(req) }),
        h("span", { class: "lab", text: present ? "is in the list" : "is not in the list" }));
    }
    if (typeof req === "boolean" || typeof cur === "boolean") {
      const sw = (on, bot) => h("span", { class: "sw" + (on ? " on" : "") + (bot ? " is-bot" : "") });
      return h("div", { class: "toggles" },
        h("div", { class: "tg" }, sw(!!req, false), h("div", null, h("b", { text: "Rule: " + fmt(!!req) }), h("small", { text: "what the circular needs" }))),
        h("div", { class: "tg" }, sw(!!cur, true), h("div", null, h("b", { text: "Bot: " + fmt(!!cur) }), h("small", { text: "how it is set now" }))));
    }
    const c = Number(cur), r = Number(req);
    if (!isFinite(c) || !isFinite(r)) return h("span", { class: "judged", text: `Setting is ${fmt(cur)}; rule needs ${findingRule(f)}.` });
    const max = (Math.max(c, r) * 1.25) || 1;
    const pct = (v) => Math.max(0, Math.min(100, (v / max) * 100));
    const pr = pct(r), pc = pct(c);
    const node = h("div", { class: "drift", role: "img", "aria-label": `Rule needs ${findingRule(f)}. Bot is set to ${fmt(cur)}.` });
    node.append(h("div", { class: "d-track" }));
    if (op === "<=") node.append(h("div", { class: "d-zone", style: { left: "0", width: pr + "%" } }));
    if (op === ">=") node.append(h("div", { class: "d-zone", style: { left: pr + "%", width: (100 - pr) + "%" } }));
    if (f.status !== "compliant" && pc !== pr) node.append(h("div", { class: "d-gap", style: { left: Math.min(pr, pc) + "%", width: Math.abs(pc - pr) + "%" } }));
    node.append(h("div", { class: "d-tick", style: { left: pr + "%" } }), h("div", { class: "d-dot", style: { left: pc + "%" } }));
    node.append(labelAt(pr, "up", [h("b", { text: "Rule: " }), findingRule(f)]),
                labelAt(pc, "down", [h("b", { text: "Bot: " }), `${fmt(cur)}${unitText(f.unit, cur)}`]));
    return node;
  }

  /* ---------- findings ---------- */
  const isOpen = (f) => f.status !== "compliant";
  const decided = (v) => Object.values(S.decisions).filter((d) => d === v).length;

  function ticketNote(f) {
    const t = S.tickets[f.id];
    if (!t) return null;
    if (t.state === "created") {
      const base = (S.status && S.status.jira_base) || "";
      return h("span", { class: "ticket" }, "Jira ", base ? h("a", { href: `${base}/browse/${encodeURIComponent(t.key)}`, target: "_blank", rel: "noopener", text: t.key }) : t.key);
    }
    if (t.state === "duplicate") return h("span", { class: "ticket", text: "Already in Jira" });
    return h("span", { class: "ticket bad", text: "Could not create: " + (t.error || "try again") });
  }

  function findingEl(f) {
    const d = S.decisions[f.id];
    const el = h("article", { class: `finding s-${f.status}` + (d === "dismiss" ? " dismissed" : ""), "data-id": f.id });
    const small = f.status === "compliant" ? "" : f.basis !== "setting" ? "Needs a look" : f.modality === "permissive" ? "Optional rule" : "Mandatory rule";
    el.append(h("div", { class: "f-head" },
      h("h3", { class: "f-bot", text: f.bot }),
      h("span", { class: "f-where", text: f.platform }),
      h("span", { class: "f-where", text: f.clause }),
      h("span", { class: "sev" }, SEV[f.status] || f.status, small && h("small", { text: small }))));
    el.append(h("p", { class: "f-rule", text: f.rule }));
    el.append(driftLine(f));
    if (f.basis !== "setting") el.append(h("p", { class: "f-why", text: f.reason }));
    else el.append(h("p", { class: "f-why", text: `Checked on the setting ${f.parameter}.` }));
    if (isOpen(f)) {
      const mk = (val, label) => h("button", { type: "button", "aria-pressed": String(d === val), onclick: () => decide(f.id, val), text: label });
      el.append(h("div", { class: "f-foot" }, h("div", { class: "seg", role: "group", "aria-label": `Decision for ${f.bot}, ${f.clause}` }, mk("approve", "Approve"), mk("dismiss", "Dismiss")), ticketNote(f)));
    }
    return el;
  }

  function decide(id, val) {
    S.decisions[id] = S.decisions[id] === val ? undefined : val;
    if (!S.decisions[id]) delete S.decisions[id];
    renderList(); renderBar();
  }

  function renderList() {
    const box = $("list"); if (!box) return;
    clear(box);
    const all = S.result.findings;
    const open = all.filter(isOpen), ok = all.filter((f) => !isOpen(f));
    const want = { High: "violating", Medium: "at_risk", Low: "optional_gap", Compliant: "compliant" }[S.filter];
    const shown = S.filter === "all" ? open : all.filter((f) => f.status === want);
    if (!shown.length && S.filter !== "all") box.append(h("p", { class: "empty-note", text: "Nothing here." }));
    const ctxStatus = (S.result.context || {}).status;
    if (!open.length && S.filter === "all" && ctxStatus !== "withdrawn") box.append(h("p", { class: "empty-note", text: S.result.obligations.length === 0
      ? (S.status && S.status.llm ? "This text has no limits, deadlines or yes/no requirements to check. Titles, commencement and sign-off clauses are skipped on purpose."
         : "Nothing could be read from this circular with keyword rules alone. This server has no LLM key, so only a few built-in rule types are recognised. Try the samples, or run it with a GROQ_API_KEY.")
      : (S.result.findings.length + (S.result.skipped_not_applicable || 0)) === 0 ? "None of your bots handle these rules, so there was nothing to compare. Add or import the bot that does this work to have it checked."
      : S.result.findings.length === 0 ? "Your bots were compared with these rules, but none of the rules applied to any bot setting, so there is nothing to fix. Expand the list below to see the rules that were read."
      : "No drift found. Every bot that handles these rules is within them." }));
    shown.forEach((f) => box.append(findingEl(f)));
    if (S.filter === "all" && ok.length) {
      const det = h("details", { class: "fine" }, h("summary", { text: `Compliant (${ok.length})` }));
      ok.forEach((f) => det.append(findingEl(f)));
      box.append(det);
    }
  }

  function renderBar() {
    const bar = $("bar");
    if (!S.result || !S.result.findings.some(isOpen)) { bar.hidden = true; return; }
    bar.hidden = false;
    const open = S.result.findings.filter(isOpen).length, a = decided("approve"), x = decided("dismiss");
    const t = $("bar-text"); clear(t);
    t.append(`${a} approved`, h("span", { text: `${x} dismissed, ${open - a - x} not decided` }));
    const can = S.status && S.status.ticketing;
    const pending = S.result.findings.filter((f) => S.decisions[f.id] === "approve" && !S.tickets[f.id]).length;
    const b = $("tickets");
    b.disabled = !can || pending === 0;
    b.title = can ? "" : "Ticketing is turned off on this deployment. Download the CSV instead.";
    b.textContent = pending > 0 ? `Create ${pending} ticket${pending === 1 ? "" : "s"}` : "Create tickets";
  }

  function renderResult() {
    const r = S.result, box = clear($("results"));
    const open = r.findings.filter(isOpen);
    const bots = new Set(open.map((f) => f.bot_id)).size;
    const count = (st) => r.findings.filter((f) => f.status === st).length;
    const cx = r.context || {};
    const heading = cx.status === "withdrawn" ? "Circular withdrawn, nothing checked"
      : cx.status === "expired" ? "Temporary measure has ended"
      : r.obligations.length === 0 ? "No checkable rules found" : null;
    box.append(h("div", { class: "sum" },
      h("h1", { class: "h1 small", text: heading || (r.obligations.length === 0 ? "No checkable rules found" : bots === 0 ? (r.findings.length === 0 ? (r.skipped_not_applicable ? "No bot settings affected" : "No matching bots") : "No bots out of line") : `${bots} bot${bots === 1 ? "" : "s"} out of line`) }),
      h("p", { class: "meta", text: cx.status === "withdrawn" ? (r.circular_ref || r.source) : `Checked ${r.findings.length} rule and bot pairs from ${r.obligations.length} rule${r.obligations.length === 1 ? "" : "s"} in ${r.circular_ref || r.source}.` +
        (r.skipped_not_applicable ? ` ${r.skipped_not_applicable} pair${r.skipped_not_applicable === 1 ? " was" : "s were"} set aside as not applicable.` : "") })));
    const notes = (cx.warnings || []).slice();
    const cv = r.coverage || {};
    if (r.unchecked > 0) notes.push(`${r.unchecked} rule and bot pair${r.unchecked === 1 ? " was" : "s were"} not checked because the time limit was reached. Run it again in a minute, or try a shorter circular.`);
    if (cv.sections > 0 && cv.answered === 0) notes.push("The AI reader did not answer (it may be busy or rate limited), so only the basic keyword rules ran. Try again in a minute.");
    else if (cv.sections > cv.answered) notes.push(`Only ${cv.answered} of ${cv.sections} sections of this circular were read` +
      (cv.stopped ? " (it was taking too long, so reading stopped early)" : cv.sent < cv.sections ? " (it is very long)" : " (the AI service did not answer for the rest)") + ", so rules in the rest were not checked.");
    if (cx.status && cx.status !== "in_force") notes.unshift(cx.headline);
    if (notes.length) {
      const tone = cx.status === "withdrawn" || cx.status === "expired" ? "stop" : cx.status === "upcoming" ? "soon" : "warn";
      const nb = h("div", { class: `notice ${tone}`, role: "note" }, h("b", { text: { stop: "Not in force", soon: "Not yet in force", warn: "Check the date" }[tone] }));
      notes.forEach((t) => nb.append(h("p", { text: t })));
      box.append(nb);
    }
    const filters = [["all", "All", open.length], ["High", "High", count("violating")], ["Medium", "Medium", count("at_risk")],
                     ["Low", "Low", count("optional_gap")], ["Compliant", "Compliant", count("compliant")]];
    const fbox = h("div", { class: "filters", role: "group", "aria-label": "Filter findings" });
    filters.forEach(([key, label, n]) => fbox.append(h("button", { class: "chip", type: "button", "aria-pressed": String(S.filter === key),
      onclick: () => { S.filter = key; renderResult(); }}, label, h("i", { text: String(n) }))));
    if (r.findings.length) box.append(fbox);
    box.append(h("div", { id: "list" }));
    box.append(aboutEl());
    renderList(); renderBar();
  }

  function aboutEl() {
    const r = S.result;
    return h("details", { class: "fine about" },
      h("summary", { text: `How this was checked (${r.obligations.length} rule${r.obligations.length === 1 ? "" : "s"} found)` }),
      h("dl", null,
        h("dt", { text: "Circular" }), h("dd", { text: r.circular_ref || r.source }),
        h("dt", { text: "Read by" }), h("dd", { text: r.extractor === "llm" ? "LLM" : r.extractor === "lora" ? "LoRA model" : "Keyword rules" }),
        h("dt", { text: "Run at" }), h("dd", { text: new Date(r.run_at).toLocaleString() }),
        h("dt", { text: "SHA-256" }), h("dd", { text: r.sha256.slice(0, 16) + "…" })),
      r.obligations.length ? h("ul", { class: "rules" }, r.obligations.map((o) => h("li", null,
        h("span", { class: "rc", text: o.clause }), h("span", { class: "rv", text: ruleShort(o) }), h("p", { text: o.summary })))) : h("p", { class: "note", text: "No checkable rules were found in this text." }));
  }

  /* ---------- run flow ---------- */
  const STAGES = [["ingest", "Reading the circular"], ["extract", "Extracting rules"], ["retrieve", "Finding bots that handle them"], ["reason", "Comparing rules with settings"]];

  function show(which) {
    for (const id of ["empty", "running", "failed", "results"]) $(id).hidden = id !== which;
    if (which !== "results") $("bar").hidden = true;
  }
  function renderRunning(stage, progress) {
    const idx = Math.max(0, STAGES.findIndex(([k]) => k === stage));
    const box = clear($("running"));
    const sub = (i) => (i === idx && progress && progress.of > 1 && (stage === "extract" || stage === "reason"))
      ? ` (${stage === "extract" ? "section" : "pair"} ${progress.done} of ${progress.of})` : "";
    box.append(h("h2", { class: "h2", text: "Checking " + (S.input ? S.input.label : "the circular") }),
      h("ol", null, STAGES.map(([, label], i) => h("li", { class: i < idx ? "done" : i === idx ? "now" : "" }, label + sub(i)))),
      h("p", { class: "meta", text: "Long circulars are read in sections and can take a few minutes." }));
  }
  function fail(msg) {
    const box = clear($("failed"));
    box.append(h("h2", { class: "h2", text: "That check did not finish" }), h("p", { text: msg }),
      h("button", { class: "btn", type: "button", onclick: () => { show("empty"); }, text: "Back" }));
    show("failed");
  }

  async function run() {
    if (!S.input || S.running) return;
    S.running = true; $("run").disabled = true;
    S.decisions = {}; S.tickets = {}; S.filter = "all"; S.result = null;
    renderRunning("ingest"); show("running");
    try {
      let res;
      if (S.input.kind === "file") { const fd = new FormData(); fd.append("file", S.input.file); res = await api("/api/analyze", { method: "POST", body: fd }); }
      else if (S.input.kind === "sample") res = await api("/api/analyze", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ sample: S.input.file }) });
      else res = await api("/api/analyze", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ text: S.input.text }) });
      S.job = res.job_id;
      for (let i = 0; i < 800; i++) {
        const j = await api(`/api/jobs/${S.job}`);
        if (j.status === "done") { S.result = j.result; S.tickets = j.tickets || {}; renderResult(); show("results"); if (matchMedia("(max-width: 900px)").matches) $("results").scrollIntoView({ block: "start" }); return; }
        if (j.status === "error") throw new Error(j.error || "The check failed.");
        renderRunning(j.stage, j.progress);
        await new Promise((r) => setTimeout(r, 900));
      }
      throw new Error("The check took too long. Try a shorter circular.");
    } catch (e) { fail(e.message); }
    finally { S.running = false; updateRun(); }
  }

  /* ---------- input panel ---------- */
  function setInput(inp) { S.input = inp; renderPicked(); updateRun(); }
  function updateRun() {
    $("run").disabled = S.running || !S.input;
    $("input-note").textContent = S.input ? "" : "Pick a circular to begin.";
  }
  function renderPicked() {
    const p = $("picked");
    document.querySelectorAll(".sample").forEach((b) => b.setAttribute("aria-pressed", String(!!S.input && S.input.kind === "sample" && S.input.file === b.dataset.file)));
    if (!S.input || S.input.kind === "sample") { p.hidden = true; return; }
    clear(p); p.hidden = false;
    p.append(h("span", { text: S.input.label }), h("button", { class: "link", type: "button", text: "Remove", onclick: () => { $("file").value = ""; setInput(null); } }));
  }
  function pickFile(f) {
    if (!f) return;
    if (!/\.(pdf|txt)$/i.test(f.name)) { $("input-note").textContent = "Use a PDF or a .txt file."; return; }
    setInput({ kind: "file", file: f, label: f.name });
  }

  function renderSamples() {
    const ul = clear($("samples"));
    S.samples.forEach((s) => ul.append(h("li", null, h("button", { class: "sample", type: "button", "data-file": s.file, "aria-pressed": "false",
      onclick: () => { $("file").value = ""; $("paste-text").value = ""; setInput({ kind: "sample", file: s.file, label: s.label }); } },
      h("b", { text: s.label }), h("span", { text: s.ref || "" })))));
  }

  /* ---------- empty state ---------- */
  function renderEmpty() {
    const ex = { id: "x", status: "violating", basis: "setting", operator: "<=", required: 7, current: 12, unit: "days", parameter: "str_filing_days" };
    clear($("empty")).append(
      h("h1", { class: "h1", text: "Find the bots a new circular puts out of line" }),
      h("p", { class: "lede", text: "Add an RBI circular. RegDrift reads its rules, compares them with each bot's settings, and lists the bots that no longer comply, so you can decide what to ticket." }),
      h("div", { class: "example" }, h("p", { class: "tag", text: "How a finding looks: reports must be filed within 7 days, but the bot waits 12." }), driftLine(ex)));
  }

  /* ---------- bots ---------- */
  function renderBots() {
    const box = clear($("bot-list"));
    $("bot-count").textContent = S.bots.length ? String(S.bots.length) : "";
    S.bots.forEach((b) => {
      const params = Object.entries(b.parameters || {});
      box.append(h("article", { class: "bot" },
        h("div", null, h("h2", { text: b.name }), h("span", { class: "pf", text: b.platform })),
        h("div", null, h("p", { text: b.description }), h("p", { class: "tags", text: (b.tags || []).join(", ") })),
        params.length ? h("dl", null, params.map(([k, v]) => [h("dt", { text: k }), h("dd", { text: fmt(v) })])) : h("p", { class: "none", text: "No settings exposed. Judged from its description." })));
    });
  }

  /* ---------- CSV, tickets, unlock ---------- */
  const csvCell = (v) => { let s = v == null ? "" : String(v); if (/^[=+\-@\t\r]/.test(s)) s = "'" + s; return '"' + s.replace(/"/g, '""') + '"'; };
  function downloadCsv() {
    const a = decided("approve");
    const rows = S.result.findings.filter((f) => isOpen(f) && (a ? S.decisions[f.id] === "approve" : true));
    const cols = ["severity", "bot", "platform", "clause", "status", "rule", "reason", "parameter", "operator", "required", "current"];
    const cell = (f, c) => c === "severity" ? SEV[f.status] : c === "operator" ? (OPS[f.operator] || f.operator) : Array.isArray(f[c]) ? f[c].join("; ") : f[c];
    const text = [cols.join(",")].concat(rows.map((f) => cols.map((c) => csvCell(cell(f, c))).join(","))).join("\r\n");
    const url = URL.createObjectURL(new Blob([text], { type: "text/csv" }));
    const link = h("a", { href: url, download: "regdrift-findings.csv" }); document.body.append(link); link.click(); link.remove(); URL.revokeObjectURL(url);
  }

  async function createTickets() {
    if (!S.code) { openUnlock(createTickets); return; }
    const ids = S.result.findings.filter((f) => S.decisions[f.id] === "approve" && !S.tickets[f.id]).map((f) => f.id);
    if (!ids.length) return;
    const b = $("tickets"); b.disabled = true; b.textContent = "Creating…";
    try {
      const res = await api(`/api/jobs/${S.job}/tickets`, { method: "POST", headers: { "content-type": "application/json", "x-access-code": S.code }, body: JSON.stringify({ ids }) });
      Object.assign(S.tickets, res.tickets);
    } catch (e) { if (/access code/i.test(e.message)) S.code = ""; ids.forEach((id) => { S.tickets[id] = { state: "failed", error: e.message }; }); }
    renderList(); renderBar();
  }

  let afterUnlock = null;
  function openUnlock(then) { afterUnlock = then || null; $("unlock-err").textContent = ""; $("code").value = ""; $("unlock").showModal(); $("code").focus(); }

  /* ---------- boot ---------- */
  function route() {
    const bots = location.hash === "#bots";
    $("view-check").hidden = bots; $("view-bots").hidden = !bots;
    $("tab-check").setAttribute("aria-current", bots ? "false" : "page");
    $("tab-bots").setAttribute("aria-current", bots ? "page" : "false");
    document.title = bots ? "Bots - RegDrift" : "RegDrift";
    if (bots) $("bar").hidden = true; else if (S.result) renderBar();
  }

  async function boot() {
    renderEmpty(); updateRun();
    $("run").addEventListener("click", run);
    $("csv").addEventListener("click", downloadCsv);
    $("tickets").addEventListener("click", createTickets);
    $("file").addEventListener("change", (e) => pickFile(e.target.files[0]));
    $("paste-text").addEventListener("input", (e) => { const t = e.target.value.trim(); if (t) { $("file").value = ""; setInput({ kind: "text", text: t, label: "Pasted text" }); } else if (S.input && S.input.kind === "text") setInput(null); });
    const drop = $("drop");
    ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
    ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
    drop.addEventListener("drop", (e) => pickFile(e.dataTransfer.files[0]));
    $("unlock-btn").addEventListener("click", () => openUnlock());
    $("unlock-cancel").addEventListener("click", () => $("unlock").close());
    $("unlock-form").addEventListener("submit", async (e) => {
      e.preventDefault();
      const code = $("code").value;
      try { await api("/api/unlock", { method: "POST", headers: { "x-access-code": code } }); S.code = code; $("unlock").close(); $("unlock-btn").hidden = true; renderBar(); if (afterUnlock) afterUnlock(); }
      catch (err) { $("unlock-err").textContent = err.message; }
    });
    window.addEventListener("hashchange", route);
    route();
    try {
      const [st, sm, bt] = await Promise.all([api("/api/status"), api("/api/samples"), api("/api/bots")]);
      S.status = st; S.samples = sm; S.bots = bt;
      $("mode").textContent = `${st.llm ? "LLM on" : "Keyword rules only"}. ${st.ticketing ? "Ticketing on" : "Tickets off"}`;
      $("mode").classList.toggle("live", st.llm);
      $("unlock-btn").hidden = !st.ticketing;
    } catch (e) { $("input-note").textContent = "Could not reach the server. Reload the page."; }
    renderSamples(); renderBots();
  }
  boot();
})();
