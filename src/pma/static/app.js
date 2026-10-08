"use strict";
// All model output is rendered with textContent (never innerHTML): generated text is untrusted.

const $ = (id) => document.getElementById(id);

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function parseSpecs(raw) {
  const specs = {};
  raw.split("\n").forEach((line) => {
    const i = line.indexOf("=");
    if (i > 0) {
      let key = line.slice(0, i).trim();
      const value = line.slice(i + 1).trim();
      let n = 2;
      while (key in specs) key = `${line.slice(0, i).trim()} (${n++})`;
      if (key && value) specs[key] = value;
    }
  });
  return specs;
}

function readProduct() {
  const price = $("price").value;
  return {
    id: $("form").dataset.exampleId || null,
    name: $("name").value,
    category: $("category").value,
    price: price === "" ? null : Number(price),
    currency: $("currency").value || "EUR",
    audience: $("audience").value,
    specs: parseSpecs($("specs").value),
    notes: $("notes").value,
  };
}

function fillProduct(p) {
  $("form").dataset.exampleId = p.id || "";
  $("name").value = p.name;
  $("category").value = p.category;
  $("price").value = p.price ?? "";
  $("currency").value = p.currency || "EUR";
  $("audience").value = p.audience || "";
  $("specs").value = Object.entries(p.specs || {}).map(([k, v]) => `${k}=${v}`).join("\n");
  $("notes").value = p.notes || "";
}

function languages() {
  const out = [];
  if ($("lang-en").checked) out.push("en");
  if ($("lang-nl").checked) out.push("nl");
  return out;
}

function renderLang(code, content) {
  const wrap = el("div", "lang");
  wrap.appendChild(el("h3", null, code === "en" ? "English" : "Nederlands"));
  [["Description", content.description], ["Ad copy", content.ad_copy], ["Social post", content.social_post]]
    .forEach(([label, text]) => {
      const card = el("div", "card");
      card.appendChild(el("b", null, `${label} (${text.length})`));
      card.appendChild(el("span", null, text));
      wrap.appendChild(card);
    });
  const subjects = el("div", "card");
  subjects.appendChild(el("b", null, "Email subject lines"));
  const ul = el("ul");
  content.email_subjects.forEach((s) => ul.appendChild(el("li", null, `${s} (${s.length})`)));
  subjects.appendChild(ul);
  wrap.appendChild(subjects);
  return wrap;
}

function renderReport(result) {
  const box = el("div");
  box.appendChild(el("h3", null, result.status === "passed" ? "Why this passed" : "Why this was not approved"));
  box.appendChild(el("p", null, result.report.summary));
  const chips = el("div");
  result.report.checks_passed.forEach((c) => chips.appendChild(el("span", "chk ok", c)));
  result.report.checks_failed.forEach((c) => chips.appendChild(el("span", "chk bad", c)));
  box.appendChild(chips);
  if (result.report.judge_reasons.length) {
    box.appendChild(el("h3", null, "LLM judge"));
    const ul = el("ul");
    result.report.judge_reasons.forEach((r) => ul.appendChild(el("li", null, r)));
    box.appendChild(ul);
  }
  if (result.report.revisions.length) {
    box.appendChild(el("h3", null, "Revisions"));
    const ul = el("ul");
    result.report.revisions.forEach((rev) =>
      ul.appendChild(el("li", null, `Iteration ${rev.iteration}: ${rev.problems.join(" | ")}`)));
    box.appendChild(ul);
  }
  if (result.security.injection_detected) {
    box.appendChild(el("h3", "warn", "Security: suspected prompt injection in the input"));
    const ul = el("ul");
    result.security.findings.forEach((f) => ul.appendChild(el("li", null, `${f.field} - ${f.rule}: ${f.snippet}`)));
    box.appendChild(ul);
  }
  if (result.report.data_warnings.length) {
    box.appendChild(el("h3", null, "Data warnings"));
    const ul = el("ul");
    result.report.data_warnings.forEach((w) => ul.appendChild(el("li", null, w)));
    box.appendChild(ul);
  }
  if (result.report.guidelines_used.length) {
    box.appendChild(el("h3", null, "Guidelines retrieved"));
    box.appendChild(el("p", "muted", result.report.guidelines_used.join(", ")));
  }
  const u = result.usage;
  box.appendChild(el("p", "muted",
    `Run ${result.run_id} - ${result.iterations} iteration(s), ${u.llm_calls} LLM calls, ` +
    `${u.prompt_tokens + u.completion_tokens} tokens${u.tokens_estimated ? " (estimated)" : ""}, ` +
    `${Math.round(result.latency_ms)} ms`));
  return box;
}

function render(result) {
  const status = $("status");
  status.hidden = false;
  status.className = `status ${result.status}`;
  status.textContent = `Status: ${result.status}`;
  const out = $("output");
  out.replaceChildren();
  if (result.draft) {
    ["en", "nl"].forEach((code) => {
      if (result.draft[code]) out.appendChild(renderLang(code, result.draft[code]));
    });
  } else {
    out.appendChild(el("p", "muted", "No draft was produced."));
  }
  out.appendChild(renderReport(result));
}

async function run(path, button) {
  const langs = languages();
  if (!langs.length) { alert("Select at least one language."); return; }
  const buttons = document.querySelectorAll("button");
  buttons.forEach((b) => { b.disabled = true; });
  $("status").hidden = false;
  $("status").className = "status";
  $("status").textContent = "Running...";
  try {
    const response = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ product: readProduct(), languages: langs }),
    });
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(typeof detail.detail === "string" ? detail.detail : `HTTP ${response.status}`);
    }
    render(await response.json());
  } catch (err) {
    $("status").className = "status error";
    $("status").textContent = `Request failed: ${err.message}`;
  } finally {
    buttons.forEach((b) => { b.disabled = false; });
  }
}

async function init() {
  const config = await (await fetch("/api/config")).json();
  if (config.simulated) {
    const banner = $("banner");
    banner.hidden = false;
    banner.textContent =
      "Mock provider: the text is written by a rule-based simulator, not a language model. " +
      "Configure LLM_PROVIDER for real output.";
  }
  const examples = await (await fetch("/api/examples")).json();
  const select = $("example");
  examples.forEach((p, i) => select.appendChild(el("option", null, `${p.name} (${p.category})`)).value = String(i));
  select.addEventListener("change", () => { if (select.value !== "") fillProduct(examples[Number(select.value)]); });
  if (examples.length) { select.value = "0"; fillProduct(examples[0]); }
  $("form").addEventListener("submit", (e) => { e.preventDefault(); run("/api/generate"); });
  $("run-baseline").addEventListener("click", () => run("/api/baseline"));
}

init();
