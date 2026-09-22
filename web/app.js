/* The client over the API. No build step, no framework, one origin.
 *
 * Five pages, in the order work happens:
 *
 *   Video RAG    the extraction components, one at a time
 *   Aggregates   higher-level answers -- once video_rag has something to read
 *   Prompts      describe's questions, and what the aggregates ask a model
 *   Search       moments or videos
 *   Data         this video's aggregate files, and the rows in Postgres
 *
 * NOTHING ABOUT THE PIPELINE IS WRITTEN DOWN HERE. Every form is generated
 * from `GET /capabilities`: `parameters.<component>` names each setting with
 * its type and default, the registries supply the legal values, and
 * `aggregators` lists every aggregate -- code and definitions alike -- with its
 * tier, kind and default input. A restated list is a second copy to keep in
 * step, and when it drifts a form offers a parameter the component does not
 * take or hides one it does.
 *
 * A blank field is OMITTED rather than sent as null, so the component's own
 * default applies and this page holds no second copy of it.
 *
 * The widgets written by hand are the field builder -- a field is a name, a
 * type, a description and optionally a nested map, and a text box does not say
 * so -- and the answer viewer, which reads any payload generically: prose as
 * prose, lists of records as tables, the rest as key and value.
 */

const $ = (sel, root = document) => root.querySelector(sel);
const el = (tag, props = {}, kids = []) => {
  const node = Object.assign(document.createElement(tag), props);
  for (const kid of [].concat(kids)) {
    if (kid === null || kid === undefined) continue;
    node.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return node;
};

async function api(path, options) {
  const res = await fetch(path, options);
  const text = await res.text();
  let body;
  try { body = text ? JSON.parse(text) : null; } catch { body = { raw: text }; }
  if (!res.ok) {
    const detail = body && body.detail !== undefined ? body.detail : body;
    const err = new Error(
      typeof detail === "string" ? detail : JSON.stringify(detail, null, 2));
    err.status = res.status;
    err.detail = detail;
    throw err;
  }
  return body;
}

const postJSON = (path, payload) => api(path, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(payload),
});

const show = (node, text) => { node.textContent = text; };
const chip = (text, cls = "") => el("span", { className: `chip ${cls}` }, text);
const errorText = err => typeof err.detail === "object"
  ? JSON.stringify(err.detail, null, 2) : err.message;

let CAPS = null;      // GET /capabilities
let VIDEO = null;     // the selected video_id
let DETAIL = null;    // this video's row from GET /videos: artifacts, streams
let PICKED = null;    // the video_rag component whose form is showing
let PAGE = "rag";     // the page showing

/* ----------------------------------------------------------------- pages */

const OPEN = {
  aggregates: () => { renderAggregatePage(); loadAnswers(); },
  prompts: () => { loadPrompts(); loadAggregatePrompts(); },
  data: () => { loadAggregateFiles(); loadDbStatus(); },
};

function openPage(name) {
  PAGE = name;
  for (const button of document.querySelectorAll("#tabs button")) {
    const on = button.dataset.tab === name;
    button.setAttribute("aria-selected", String(on));
    $(`#tab-${button.dataset.tab}`).hidden = !on;
  }
  (OPEN[name] || (() => {}))();
}
for (const button of document.querySelectorAll("#tabs button")) {
  button.onclick = () => openPage(button.dataset.tab);
}

/* --------------------------------------------------------------- videos */

async function loadVideos(keep) {
  const { videos } = await api("/videos");
  const select = $("#video");
  select.replaceChildren(...videos.map(v =>
    el("option", { value: v.video_id, textContent: v.video_id })));
  select.append(el("option", { value: "__upload__", textContent: "+ upload a file..." }));
  if (keep && videos.some(v => v.video_id === keep)) select.value = keep;
  VIDEO = select.value === "__upload__" ? null : select.value;
  loadVideoDetail(videos);
  renderScope(videos.filter(v => v.artifacts.includes("embedded")));
  renderSteps();
  (OPEN[PAGE] || (() => {}))();
}

function loadVideoDetail(videos) {
  const facts = $("#video-facts");
  DETAIL = VIDEO ? (videos || []).find(v => v.video_id === VIDEO) || null : null;
  if (!DETAIL) { facts.textContent = ""; return; }
  const bits = [`${(DETAIL.duration_s || 0).toFixed(1)}s`];
  bits.push(DETAIL.has_video ? "video" : "no video");
  bits.push(DETAIL.has_audio ? "audio" : "no audio");
  if (DETAIL.chunks) bits.push(`${DETAIL.chunks} chunks · ${DETAIL.policy}`);
  facts.textContent = bits.join(" · ");
}

$("#video").onchange = async () => {
  if ($("#video").value === "__upload__") return upload();
  VIDEO = $("#video").value;
  await loadVideos(VIDEO);
};
$("#refresh").onclick = () => loadVideos(VIDEO);

/* `media` is the one component `/run/{component}` cannot reach: until it has
 * run there is no id to put in the URL. So uploading IS running component 1,
 * and `run=false` stops there rather than queueing the whole pipeline. */
async function upload() {
  const input = el("input", { type: "file", accept: "video/*,audio/*" });
  input.onchange = async () => {
    const file = input.files[0];
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    form.append("run", "false");
    form.append("sink", $("#video").dataset.sink || "file,supabase");
    openPage("rag");
    show($("#output"), `uploading ${file.name} ...`);
    try {
      const out = await api("/videos", { method: "POST", body: form });
      show($("#output"), JSON.stringify(out, null, 2));
      await loadVideos(out.video_id);
    } catch (err) {
      show($("#output"), `error ${err.status}\n${err.message}`);
      await loadVideos(VIDEO);
    }
  };
  input.click();
}

/* `stage` is what is RUNNING; `history` is what has finished. The workflow
 * announces a component twice for exactly this reason, so a poller does not
 * read the previous component's name through the longest stage of a run. */
async function pollJob(id, state, output, onDone) {
  for (;;) {
    const job = await api(`/jobs/${id}`);
    state.replaceChildren(
      chip(job.state, job.state === "failed" ? "bad" : "on"),
      el("span", { className: "hint", textContent:
        ` ${job.stage || ""} ${job.elapsed_s != null ? job.elapsed_s + "s" : ""}` }));
    show(output, JSON.stringify(job.error ? {
      state: job.state, error: job.error, traceback: (job.detail || {}).traceback,
    } : {
      state: job.state, stage: job.stage, elapsed_s: job.elapsed_s, detail: job.detail,
    }, null, 2));
    if (job.state === "done") { if (onDone) await onDone(job); return; }
    if (job.state === "failed") return;
    await new Promise(r => setTimeout(r, 1000));
  }
}

/* ----------------------------------------------------- page 1: video rag */

/* Which stages this file cannot support, decided AFTER `media` has read it.
 * Whether a file carries a soundtrack is a property of the file, not of the
 * request, so a stage needing a stream it has none of is marked n/a with the
 * reason rather than queued to fail. */
function unusable(name) {
  if (!DETAIL) return null;
  if (!DETAIL.has_audio && (name === "audio" || name === "cut"))
    return "the file carries no audio stream";
  if (!DETAIL.has_video && (name === "video" || name === "describe"))
    return "the file carries no video stream";
  return null;
}

/* Aggregates have their own page: they read what these wrote, never the video. */
const ragComponents = () => (CAPS.components || []).filter(c => c !== "aggregate");

function renderSteps() {
  if (!CAPS) return;
  const list = $("#steps");
  list.replaceChildren();
  const present = new Set((DETAIL && DETAIL.artifacts) || []);
  const wrote = { media: "media", audio: "raw_transcript", "boundaries.evidence": "cuts",
                  boundaries: "timeline", video: "manifest", cut: "transcript",
                  describe: "descriptions", embed: "embedded" };
  ragComponents().forEach((name, i) => {
    const why = unusable(name);
    const button = el("button", { type: "button" }, [
      el("span", { className: "n", textContent: String(i + 1) }),
      el("span", { textContent: name }),
    ]);
    const mark = name === "media" ? "upload" : why ? "n/a"
      : present.has(wrote[name]) ? "done" : "";
    if (mark) button.append(el("span", { className: "mark", textContent: mark }));
    button.setAttribute("aria-selected", String(name === PICKED));
    button.onclick = () => { PICKED = name; renderSteps(); renderParams(); };
    list.append(el("li", {}, button));
  });
  const why = PICKED && unusable(PICKED);
  $("#plan-note").textContent = why
    ? `${PICKED}: skipped — ${why}`
    : "Order is yours on this path; workflow.py is the reference. "
      + "When describe or cut has run, the Aggregates page opens up.";
}

/* Legal values, from the registries rather than from a list here. */
function choicesFor(param) {
  const c = CAPS;
  const map = {
    policy: c.policies,
    tier: c.tiers, transcriber: c.transcribers, diarizer: c.diarizers,
    store_scope: ["sampled", "decimated"],
    sink: ["file", "supabase", "file,supabase"],
    index_name: [...c.indexes, c.indexes.join(",")],
  };
  return map[param.name] || null;
}

/* Who can serve a role, which can run here, and what a blank field resolves
 * to -- all from /capabilities, so an endpoint added to data/providers.json
 * appears without touching this file. */
function providerNote(param) {
  const role = param === "embedder" ? "embed" : "chat";
  const listed = (((CAPS.models || {}).providers) || []).filter(p => p[role]).map(p => {
    const model = role === "embed" ? p.embed_model : p.chat_model;
    return `${p.name}${model ? ` (${model})` : ""}${p.configured ? "" : " ✗"}`;
  }).join(" · ");
  return `${param}: a provider, or provider/model. blank = ${(CAPS.defaults || {})[param]}; `
       + `✗ = no key set. ${listed}`;
}

const PROVIDER_PARAMS = ["describer", "embedder", "llm"];

function renderParams() {
  const box = $("#params");
  box.replaceChildren();
  $("#params-title").textContent = PICKED ? `${PICKED} — parameters` : "Parameters";
  $("#params-note").textContent = "";

  if (!PICKED) { box.append(el("p", { className: "note", textContent: "Pick a component." })); return; }

  if (PICKED === "media") {
    box.append(
      el("p", { className: "note", textContent:
        "media establishes the video id, so it is the one component the run "
        + "route cannot address. Use the video selector — “+ upload a file…” — "
        + "which posts run=false: it runs media and stops." }),
      el("button", { className: "go", textContent: "Upload a file", onclick: upload }));
    $("#run").disabled = true;
    return;
  }
  $("#run").disabled = false;

  const params = (CAPS.parameters || {})[PICKED] || [];
  for (const p of params) {
    const id = `arg-${p.name}`;
    const field = el("div", { className: "field" });
    const label = el("label", { htmlFor: id }, [p.name]);
    label.append(el("span", { className: "hint", textContent:
      p.required ? "required" : `default ${JSON.stringify(p.default)}` }));
    field.append(label);

    const choices = choicesFor(p);
    const kind = kindOf(p.type);
    let input;
    if (kind === "bool") {
      input = el("select", { id });
      input.append(el("option", { value: "", textContent: `— default (${p.default})` }),
                   el("option", { value: "true", textContent: "true" }),
                   el("option", { value: "false", textContent: "false" }));
    } else if (choices) {
      input = el("select", { id });
      input.append(el("option", { value: "", textContent: "— default" }));
      for (const c of choices) input.append(el("option", { value: c, textContent: c }));
    } else if (kind === "int" || kind === "float") {
      input = el("input", { type: "number", id, step: kind === "float" ? "any" : "1",
                            placeholder: p.default === null ? "" : String(p.default) });
    } else {
      input = el("input", { type: "text", id,
                            placeholder: p.default === null ? "" : String(p.default) });
    }
    input.dataset.kind = kind;
    input.dataset.name = p.name;
    if (PROVIDER_PARAMS.includes(p.name)) input.placeholder = (CAPS.defaults || {})[p.name] || "";
    field.append(input);

    if (p.name === "sampler") {
      field.append(el("p", { className: "note", textContent:
        `samplers: ${CAPS.samplers.join(", ")} · questions: ${CAPS.prompts.join(", ")}`
        + " — pair as name:question, or name:[q1,q2] for one pass answering two." }));
    }
    if (PROVIDER_PARAMS.includes(p.name)) {
      field.append(el("p", { className: "note", textContent: providerNote(p.name) }));
    }
    field.condition = p.when;
    box.append(field);
  }
  if (!params.length) box.append(el("p", { className: "note", textContent: "No parameters." }));
  box.oninput = box.onchange = () => applyConditions(params);
  applyConditions(params);
}

/* The published type is the annotation as written -- `Optional[float]`,
 * `str | Sequence[str]` -- because `/capabilities` restates nothing about a
 * component and the annotation is what the component takes. Picking a widget
 * needs the *shape*, not the wording, so it is read here rather than matched
 * exactly, which is what this used to do: every `Optional[...]` field fell
 * through to a text box and was sent as a string, so `threshold` reached a
 * sampler as "0.9" and raised on its range check, and `vocabulary` arrived as
 * a string that `list()` turned into one entry per letter. A union takes its
 * first member, which is why `str | Sequence[str]` stays one text box holding
 * a spec the component parses itself. */
function kindOf(type) {
  let t = String(type || "").trim().replace(/^Optional\[(.*)\]$/, "$1");
  t = t.split("|").map(s => s.trim()).filter(s => s && s !== "None")[0] || t;
  if (/^(Sequence|list|List)\[/.test(t)) return "list";
  return ["bool", "int", "float"].includes(t) ? t : "str";
}

/* A setting that only one policy, sampler or tier reads is shown only when
 * that choice is made. The conditions come from /capabilities
 * (`parameters.<component>[].when`), so this holds none of them: a blank field
 * is read as its parameter's default, and a sampler spec by the names in it. */
function currentValue(params, name) {
  const input = document.querySelector(`#params [data-name="${name}"]`);
  const raw = input ? input.value.trim() : "";
  if (raw !== "") return input.dataset.kind === "bool" ? raw === "true" : raw;
  const p = params.find(q => q.name === name);
  return p ? p.default : null;
}

function samplerNames(spec) {
  const list = Array.isArray(spec) ? spec.join(",") : String(spec || "");
  return list.replace(/\[[^\]]*\]/g, "").split(",")
    .map(s => s.split(":")[0].trim()).filter(Boolean);
}

function applies(params, when) {
  if (!when) return true;
  const value = currentValue(params, when.param);
  if (when.names) return samplerNames(value).some(n => when.names.includes(n));
  return when.in.includes(value);
}

function applyConditions(params) {
  const hidden = [];
  for (const field of document.querySelectorAll("#params .field")) {
    if (!field.condition) continue;
    field.hidden = !applies(params, field.condition);
    if (field.hidden) hidden.push(field.querySelector("[data-name]").dataset.name);
  }
  const why = [...new Set(params.filter(p => p.when && hidden.includes(p.name))
    .map(p => {
      const value = currentValue(params, p.when.param);
      return value === null ? `no ${p.when.param} yet` : `${p.when.param} = ${JSON.stringify(value)}`;
    }))];
  $("#params-note").textContent = hidden.length
    ? `Hidden (${why.join("; ")}): ${hidden.join(", ")}.` : "";
}

/* A blank field is omitted, never sent as null: the component's own default
 * then applies, and this page holds no second copy of it. */
function collectParams() {
  const out = {};
  for (const input of document.querySelectorAll("#params [data-name]")) {
    if (input.closest(".field").hidden) continue;      // a value that does not apply is not sent
    const raw = input.value.trim();
    if (raw === "") continue;
    const kind = input.dataset.kind;
    if (kind === "bool") out[input.dataset.name] = raw === "true";
    else if (kind === "int") out[input.dataset.name] = parseInt(raw, 10);
    else if (kind === "float") out[input.dataset.name] = parseFloat(raw);
    // Every sequence, not `samplers` alone: `vocabulary` and `languages` are
    // the same shape and were reaching the component as one long string.
    else if (kind === "list")
      out[input.dataset.name] = raw.split(",").map(s => s.trim()).filter(Boolean);
    else out[input.dataset.name] = raw;
  }
  return out;
}

$("#run").onclick = async () => {
  if (!VIDEO || !PICKED || PICKED === "media") return;
  const why = unusable(PICKED);
  if (why && !confirm(`${PICKED} needs a stream this file has none of (${why}).\nRun anyway?`)) return;

  const params = collectParams();
  const state = $("#run-state");
  state.replaceChildren(chip("submitting", "on"));
  show($("#output"), JSON.stringify({ component: PICKED, params }, null, 2));
  $("#artifacts").replaceChildren();
  $("#run").disabled = true;

  try {
    const { job } = await postJSON(
      `/videos/${encodeURIComponent(VIDEO)}/run/${PICKED}`, { params });
    await pollJob(job.id, state, $("#output"), showArtifacts);
  } catch (err) {
    state.replaceChildren(chip(`http ${err.status || "error"}`, "bad"));
    show($("#output"), errorText(err));
  } finally {
    $("#run").disabled = false;
    loadVideos(VIDEO);
  }
};

/* Only the artifacts that exist are offered. A link that 404s reads as
 * breakage rather than as a stage that never ran. */
async function showArtifacts() {
  const box = $("#artifacts");
  box.replaceChildren();
  const detail = await api(`/videos/${encodeURIComponent(VIDEO)}`);

  if (detail.documents.length) {
    box.append(el("h3", {}, "Artifacts"));
    const row = el("div", { className: "row" });
    for (const d of detail.documents) {
      row.append(el("button", { className: "small", textContent: d.name, title: d.about,
        onclick: async () => show($("#output"),
          JSON.stringify(await api(d.url), null, 2).slice(0, 200000)) }));
    }
    box.append(row);
  }
  if (detail.frames) {
    box.append(el("h3", {}, "Frames"));
    box.append(el("button", { className: "small", textContent: "show sampled frames",
      onclick: showFrames }));
    box.append(el("div", { className: "frames", id: "frame-strip" }));
  }
}

/* Frames come off disk by the READ INDEX the manifest names -- not by second,
 * which is a lossy rendering of a pts. */
async function showFrames() {
  const strip = $("#frame-strip");
  strip.replaceChildren("loading...");
  const manifest = await api(`/videos/${encodeURIComponent(VIDEO)}/artifacts/manifest`);
  const frames = [];
  for (const chunk of manifest.chunks || []) {
    for (const [sid, block] of Object.entries(chunk.samplers || {})) {
      for (const f of block.frames || []) frames.push({ ...f, sid, chunk: chunk.chunk_id });
    }
  }
  const seen = new Set();
  strip.replaceChildren(...frames.filter(f => {
    if (seen.has(f.index)) return false;   // one file however many samplers chose it
    seen.add(f.index); return true;
  }).slice(0, 120).map(f => el("figure", {}, [
    el("img", { src: `/videos/${encodeURIComponent(VIDEO)}/frames/${f.index}`, loading: "lazy" }),
    el("figcaption", {}, `#${f.index} · c${f.chunk} · ${f.media_ts}s`),
  ])));
  if (!frames.length) strip.replaceChildren("no frames in this manifest");
}

/* ---------------------------------------------------- page 2: aggregates */

/* Aggregates read what video_rag wrote -- descriptions, a transcript -- and
 * never the video, so until one of those exists there is nothing to ask. */
function extractedFor() {
  const present = new Set((DETAIL && DETAIL.artifacts) || []);
  return ["descriptions", "transcript"].filter(a => present.has(a));
}

const definitionOf = answer => answer.split("~")[0];
const tierRank = tier => (CAPS.tiers || []).indexOf(tier);

/* Selections survive a re-render: switching video must not wipe what was ticked. */
const PICKS = { chosen: null, inputs: {} };

function renderAggregatePage() {
  if (!CAPS) return;
  const gate = $("#a-gate");
  const have = extractedFor();
  if (!VIDEO) {
    gate.hidden = false;
    gate.textContent = "Pick a video.";
  } else if (!have.length) {
    gate.hidden = false;
    gate.textContent = `${VIDEO} has nothing to aggregate yet. Aggregates read what video_rag `
      + "wrote — run describe (the picture) or cut (the soundtrack) on the Video RAG page first.";
  } else {
    gate.hidden = true;
  }
  $("#a-run").disabled = !VIDEO || !have.length;

  const aggregators = CAPS.aggregators || {};
  if (PICKS.chosen === null) {
    PICKS.chosen = new Set(Object.keys(aggregators).filter(n => aggregators[n].tier === "free"));
  }

  const table = el("table", { className: "pick" });
  table.append(el("thead", {}, el("tr", {}, [
    el("th", {}, ""), el("th", {}, "aggregator"), el("th", {}, "kind"),
    el("th", {}, "reads"),
  ])));
  const body = el("tbody");
  for (const tier of CAPS.tiers || []) {
    const names = Object.keys(aggregators).filter(n => aggregators[n].tier === tier);
    if (!names.length) continue;
    const cost = { free: "arithmetic", local: "GPU models", llm: "paid calls" }[tier] || "";
    body.append(el("tr", { className: "tier" }, el("td", { colSpan: 4 },
      `${tier} — ${cost}`)));
    for (const name of names) {
      const a = aggregators[name];
      const box = el("input", { type: "checkbox", checked: PICKS.chosen.has(name) });
      box.onchange = () => {
        if (box.checked) PICKS.chosen.add(name); else PICKS.chosen.delete(name);
        applyAggregateConditions();
      };
      let reads;
      if (a.reads === null) {
        reads = el("span", { className: "hint", textContent: "everything extraction wrote" });
      } else {
        reads = el("input", { type: "text", placeholder: a.reads,
                              value: PICKS.inputs[name] || "" });
        reads.oninput = () => { PICKS.inputs[name] = reads.value; };
        reads.title = "blank = the default shown; see “What an input may say”";
      }
      body.append(el("tr", {}, [
        el("td", {}, box),
        el("td", { className: "wrap" }, [name, el("div", { className: "hint",
          style: "margin:0", textContent: a.about })]),
        el("td", {}, a.kind || "code"),
        el("td", {}, reads),
      ]));
    }
  }
  table.append(body);
  $("#a-list").replaceChildren(table, el("p", { className: "note", textContent:
    "Tick what to run; the run's tier is the dearest one ticked. A blank input reads the "
    + "default; comma-separated inputs make one answer each, stored as name~label." }));

  const grammar = (CAPS.aggregate_inputs || {}).grammar || [];
  const g = el("table");
  g.append(el("tbody", {}, grammar.map(x =>
    el("tr", {}, [el("td", {}, el("code", {}, x.syntax)), el("td", {}, x.reads)]))));
  $("#a-grammar-body").replaceChildren(g);

  const sink = $("#a-sink");
  if (!sink.children.length) {
    sink.append(...["file", "supabase", "file,supabase"].map(s =>
      el("option", { value: s, textContent: s })));
    $("#a-index").append(el("option", { value: "", textContent: "— no video vector" }),
      ...CAPS.indexes.map(i => el("option", { value: i, textContent: i })));
  }
  $("#a-llm").placeholder = (CAPS.defaults || {}).llm || "";
  $("#a-embedder").placeholder = (CAPS.defaults || {}).embedder || "";
  $("#a-providers").textContent = `${providerNote("llm")}\n${providerNote("embedder")}`;
  applyAggregateConditions();
}

/* The run's tier is the dearest one ticked. */
function pickedTier() {
  const tiers = [...PICKS.chosen].filter(n => (CAPS.aggregators || {})[n])
    .map(n => CAPS.aggregators[n].tier);
  return tiers.sort((a, b) => tierRank(b) - tierRank(a))[0] || null;
}

/* llm, embedder and index mean nothing below the llm tier; the condition is
 * the one /capabilities publishes for the aggregate component's parameters. */
function applyAggregateConditions() {
  const when = Object.fromEntries(((CAPS.parameters || {}).aggregate || [])
    .map(p => [p.name, p.when]));
  const tier = pickedTier();
  const hidden = [];
  for (const [id, key] of [["#a-llm", "llm"], ["#a-embedder", "embedder"], ["#a-index", "index"]]) {
    const c = when[key];
    const off = !!c && c.param === "tier" && !c.in.includes(tier);
    $(id).closest(".field").hidden = off;
    if (off) hidden.push(key);
  }
  $("#a-providers").hidden = hidden.includes("llm") && hidden.includes("embedder");
}

$("#a-run").onclick = async () => {
  const state = $("#a-state");
  const output = $("#a-output");
  const only = [...PICKS.chosen].filter(n => (CAPS.aggregators || {})[n]);
  if (!only.length) { state.replaceChildren(chip("tick at least one", "bad")); return; }

  const tier = pickedTier();
  const params = { tier, only, sink: $("#a-sink").value };
  const inputs = {};
  for (const name of only) {
    const text = (PICKS.inputs[name] || "").trim();
    if (text) inputs[name] = text;
  }
  if (Object.keys(inputs).length) params.inputs = inputs;
  for (const [id, key] of [["#a-llm", "llm"], ["#a-embedder", "embedder"], ["#a-index", "index"]]) {
    if ($(id).closest(".field").hidden) continue;       // does not apply to this tier
    const v = $(id).value.trim();
    if (v) params[key] = v;
  }
  if ($("#a-force").checked) params.force = true;

  output.hidden = false;
  show(output, JSON.stringify({ component: "aggregate", params }, null, 2));
  state.replaceChildren(chip("submitting", "on"));
  $("#a-run").disabled = true;
  try {
    const { job } = await postJSON(
      `/videos/${encodeURIComponent(VIDEO)}/run/aggregate`, { params });
    await pollJob(job.id, state, output, loadAnswers);
  } catch (err) {
    /* A typo in an input is a 422 before anything is queued, naming what the
     * question does have. */
    state.replaceChildren(chip(`http ${err.status || "error"}`, "bad"));
    show(output, errorText(err));
  } finally {
    $("#a-run").disabled = false;
  }
};

let VIEWING = null;

async function loadAnswers() {
  const box = $("#a-answers");
  if (!VIDEO) { box.replaceChildren(); $("#a-view").replaceChildren(); return; }
  const detail = await api(`/videos/${encodeURIComponent(VIDEO)}`);
  const answers = detail.aggregates || [];
  if (!answers.length) {
    box.replaceChildren(el("p", { className: "note", textContent: "No answers yet." }));
    $("#a-view").replaceChildren();
    return;
  }
  const row = el("div", { className: "answers" });
  for (const a of answers) {
    const button = el("button", { className: "small", textContent: a.name, title: a.about });
    button.setAttribute("aria-selected", String(a.name === VIEWING));
    button.onclick = () => { VIEWING = a.name; loadAnswers(); };
    row.append(button);
  }
  box.replaceChildren(row);
  const picked = answers.find(a => a.name === VIEWING) || answers[0];
  VIEWING = picked.name;
  for (const b of row.children) b.setAttribute("aria-selected", String(b.textContent === VIEWING));
  await viewAnswer(picked);
}

async function viewAnswer(a) {
  const view = $("#a-view");
  view.replaceChildren("loading...");
  const doc = await api(a.url);
  const stats = doc.stats || {};
  const meta = el("div", { className: "meta" }, [
    chip(doc.tier),
    stats.model ? chip(stats.model) : null,
    stats.inputs ? chip(`reads ${stats.inputs}`) : null,
    stats.version ? chip(`v ${stats.version}`) : null,
    stats.read_chars ? chip(`${stats.read_chars} chars`) : null,
  ]);
  view.replaceChildren(
    el("h3", { style: "margin-top:0" }, doc.aggregate_id),
    el("p", { className: "note", style: "margin-top:0" }, stats.about || a.about || ""),
    meta,
    renderPayload(doc.payload || {}),
    el("details", {}, [el("summary", {}, "raw JSON"),
      el("pre", { className: "out" }, JSON.stringify(doc, null, 2))]));
}

/* Any payload, generically: long text as prose, lists of records as tables,
 * nested records collapsed, the rest as key and value. Nothing here knows
 * which aggregator wrote it, so a custom prompt's answer reads like a built-in's. */
function renderPayload(payload) {
  const box = el("div");
  const scalars = [];
  const later = [];
  for (const [key, value] of Object.entries(payload)) {
    if (typeof value === "string" && value.length > 90) {
      box.append(el("h3", {}, key), el("p", { className: "prose" }, value));
    } else if (Array.isArray(value) && value.length && typeof value[0] === "object" && value[0] !== null) {
      later.push([key, value]);
    } else if (value !== null && typeof value === "object" && !Array.isArray(value)) {
      later.push([key, value]);
    } else {
      scalars.push([key, value]);
    }
  }
  if (scalars.length) {
    const t = el("table", { className: "kv" });
    t.append(el("tbody", {}, scalars.map(([k, v]) => el("tr", {}, [
      el("th", {}, k),
      el("td", { className: "wrap" }, Array.isArray(v) ? (v.length ? v.join(" · ") : "—")
        : v === null ? "—" : String(v)),
    ]))));
    box.append(t);
  }
  for (const [key, value] of later) {
    if (Array.isArray(value)) {
      const d = el("details", { open: key !== "layers" && value.length <= 60 });
      d.append(el("summary", {}, `${key} (${value.length})`), recordTable(value));
      box.append(d);
    } else {
      const d = el("details");
      d.append(el("summary", {}, key), el("pre", { className: "out" }, JSON.stringify(value, null, 2)));
      box.append(d);
    }
  }
  return box;
}

function cell(key, value) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number" && /(_ts|_s)$/.test(key)) return `${value.toFixed(1)}s`;
  if (key === "chunk_ids" && Array.isArray(value)) {
    return value.length > 1 && value[value.length - 1] - value[0] === value.length - 1
      ? `${value[0]}–${value[value.length - 1]}` : value.join(",");
  }
  if (Array.isArray(value) && value.every(v => typeof v !== "object")) return value.join(" · ");
  if (typeof value === "object") {
    const text = JSON.stringify(value);
    return text.length > 300 ? text.slice(0, 300) + "…" : text;
  }
  return String(value);
}

function recordTable(rows) {
  const cols = [];
  for (const row of rows.slice(0, 50)) {
    for (const key of Object.keys(row)) if (!cols.includes(key)) cols.push(key);
  }
  const t = el("table");
  t.append(el("thead", {}, el("tr", {}, cols.map(c => el("th", {}, c)))));
  t.append(el("tbody", {}, rows.slice(0, 500).map(row =>
    el("tr", {}, cols.map(c => el("td", { className: "wrap" }, cell(c, row[c])))))));
  return el("div", { className: "scroll" }, t);
}

/* ------------------------------------------------------- page 3: prompts */

async function loadPrompts() {
  const data = await api("/prompts");
  const box = $("#prompt-list");
  const table = el("table");
  table.append(el("thead", {}, el("tr", {}, [
    el("th", {}, "name"), el("th", {}, "shape"), el("th", {}, "fields"), el("th", {}, ""),
  ])));
  const body = el("tbody");
  for (const p of data.prompts) {
    const remove = el("button", { className: "small", textContent: "delete", disabled: p.builtin,
      title: p.builtin ? "built-ins ship in the package — 409" : "",
      onclick: async () => {
        if (!confirm(`Delete "${p.name}"? Descriptions already written are untouched.`)) return;
        try { await api(`/prompts/${p.name}`, { method: "DELETE" }); }
        catch (err) { alert(`${err.status}: ${err.message}`); }
        loadPrompts(); refreshCaps();
      } });
    body.append(el("tr", {}, [
      el("td", {}, [p.name, p.builtin ? "" : " *"]),
      el("td", {}, p.shape || ""),
      el("td", { className: "wrap" }, (p.fields || []).join(", ") || "—"),
      el("td", {}, remove),
    ]));
  }
  table.append(body);
  box.replaceChildren(table, el("p", { className: "note", textContent:
    "* custom. Built-ins ship in the package so every deployment's `yolo` means "
    + "the same thing; editing or deleting one is a 409." }));

  const shape = $("#p-shape");
  shape.replaceChildren(...Object.entries(data.shapes).map(([name, s]) =>
    el("option", { value: name, textContent:
      `${name}${s.builtin ? "" : " (custom)"} — ${s.fields.join(", ") || "no fields"}` })));
  describeShape(data.shapes);
  shape.onchange = () => describeShape(data.shapes);
}

function describeShape(shapes) {
  const s = shapes[$("#p-shape").value];
  $("#p-shape-fields").textContent = s
    ? `summary: ${s.summary} · fields: ${s.fields.join(", ") || "none"}` : "";
}

for (const radio of document.querySelectorAll('input[name="p-mode"]')) {
  radio.onchange = () => {
    const custom = radio.value === "custom" && radio.checked;
    $("#p-shape-wrap").hidden = custom;
    $("#p-custom-wrap").hidden = !custom;
    if (custom && !$("#p-fields").children.length) addFieldRow("#p-fields");
  };
}

/* One builder for both forms: a question's shape and an aggregate's answer are
 * the same builder on the server, so they are the same widget here. */
function addFieldRow(container, name = "", spec = {}) {
  const row = el("div", { className: "fieldrow" });
  const key = el("input", { type: "text", placeholder: "hazards", value: name });
  const type = el("select");
  type.append(el("option", { value: "list", textContent: "list" }),
              el("option", { value: "text", textContent: "text" }));
  type.value = spec.type || "list";
  const about = el("input", { type: "text", placeholder: "what to put in it",
                              value: spec.about || "" });
  const extra = el("input", { type: "text", placeholder: "of: key: desc, ... | one_of: a, b",
    value: spec.of ? Object.entries(spec.of).map(([k, d]) => `${k}: ${d}`).join(", ")
         : spec.one_of ? spec.one_of.join(", ") : "" });
  const drop = el("button", { className: "small", textContent: "×", type: "button",
    onclick: () => row.remove() });
  row.append(key, type, about, extra, drop);
  row.dataset.row = "1";
  $(container).append(row);
}
$("#p-add-field").onclick = () => addFieldRow("#p-fields");
$("#ap-add-field").onclick = () => addFieldRow("#ap-fields");

/* The builder, not raw JSON Schema. The call goes out with `strict: true`,
 * whose subset is narrow -- a schema the API refuses would fail with the
 * request about to be paid for. */
function collectFields(container) {
  const fields = {};
  for (const row of document.querySelectorAll(`${container} [data-row]`)) {
    const [name, type, about, extra] = row.children;
    const key = name.value.trim();
    if (!key) continue;
    const spec = { type: type.value, about: about.value.trim() };
    const raw = extra.value.trim();
    if (raw) {
      if (raw.includes(":")) {
        spec.of = {};
        for (const pair of raw.split(",")) {
          const at = pair.indexOf(":");
          if (at < 0) continue;
          const k = pair.slice(0, at).trim();
          if (k) spec.of[k] = pair.slice(at + 1).trim();
        }
      } else {
        spec.one_of = raw.split(",").map(s => s.trim()).filter(Boolean);
      }
    }
    fields[key] = spec;
  }
  return fields;
}

$("#p-save").onclick = async () => {
  const custom = document.querySelector('input[name="p-mode"]:checked').value === "custom";
  const payload = {
    name: $("#p-name").value.trim(),
    instruction: $("#p-instruction").value.trim(),
    about: $("#p-about").value.trim(),
  };
  if (custom) {
    payload.fields = collectFields("#p-fields");
    payload.summary = $("#p-summary").value;
  } else {
    payload.shape = $("#p-shape").value;
  }
  const state = $("#p-state");
  const out = $("#p-out");
  state.replaceChildren(chip("saving", "on"));
  try {
    const made = await postJSON("/prompts", payload);
    state.replaceChildren(chip("201 created", "on"));
    out.hidden = false;
    show(out, JSON.stringify(made, null, 2));
    loadPrompts(); refreshCaps();
  } catch (err) {
    state.replaceChildren(chip(`http ${err.status}`, "bad"));
    out.hidden = false;
    show(out, errorText(err));
  }
};

/* Aggregate prompts: the same shape of page. A kind is how a prompt is asked --
 * the kinds are the only code -- and everything the model is told is the entry. */
let DEFINITIONS = null;

const KIND_NOTES = {
  fold: "Batches the chunks, summarises each batch, then summarises the summaries. "
      + "The answer is your fields, once, for the whole video.",
  spans: "Contiguous ranges covering the video, each citing its first and last chunk. "
       + "Each span carries your fields; the list is stored under the list key.",
  items: "Discrete things, each citing the chunk it happened in. Each item carries your "
       + "fields; the list is stored under the list key.",
};

async function loadAggregatePrompts() {
  const data = DEFINITIONS = await api("/aggregate-definitions");
  const table = el("table");
  table.append(el("thead", {}, el("tr", {}, [
    el("th", {}, "name"), el("th", {}, "kind"), el("th", {}, "reads"),
    el("th", {}, "fields"), el("th", {}, ""),
  ])));
  const body = el("tbody");
  for (const p of data.prompts) {
    const actions = el("div", { className: "row", style: "gap:4px;flex-wrap:nowrap" }, [
      el("button", { className: "small", textContent: "copy", title: "start a new prompt from this one",
        onclick: () => fillAggregateForm(p) }),
      el("button", { className: "small", textContent: "delete", disabled: p.builtin,
        title: p.builtin ? "built-ins ship in the package — 409" : "",
        onclick: async () => {
          if (!confirm(`Delete "${p.name}"? Answers it already wrote are untouched.`)) return;
          try { await api(`/aggregate-prompts/${p.name}`, { method: "DELETE" }); }
          catch (err) { alert(`${err.status}: ${err.message}`); }
          loadAggregatePrompts(); refreshCaps();
        } }),
    ]);
    body.append(el("tr", {}, [
      el("td", { className: "wrap" }, [p.name, p.builtin ? "" : " *",
        el("div", { className: "hint", style: "margin:0", textContent: p.about || "" })]),
      el("td", {}, p.kind),
      el("td", {}, el("code", {}, p.inputs || data.inputs.default)),
      el("td", { className: "wrap" }, Object.keys(p.fields || {}).join(", ")),
      el("td", {}, actions),
    ]));
  }
  table.append(body);
  const problems = Object.entries(data.problems || {});
  $("#ap-list").replaceChildren(...[table,
    el("p", { className: "note", textContent:
      `* custom, in ${data.custom_file}. Link profiles are not edited here.` }),
    problems.length ? el("pre", { className: "out err" },
      problems.map(([k, v]) => `${k}: ${v.join("; ")}`).join("\n")) : null,
  ].filter(Boolean));

  const template = $("#ap-template");
  const keep = template.value;
  template.replaceChildren(el("option", { value: "", textContent: "— blank" }),
    ...data.prompts.map(p => el("option", { value: p.name,
      textContent: `${p.name} (${p.kind}${p.builtin ? "" : ", custom"})` })));
  template.value = keep;
  template.onchange = () => {
    const p = data.prompts.find(x => x.name === template.value);
    fillAggregateForm(p || null);
  };

  const kind = $("#ap-kind");
  if (!kind.children.length) {
    kind.append(...data.kinds.map(k => el("option", { value: k, textContent: k })));
    kind.onchange = kindChanged;
    kindChanged();
  }
  $("#ap-inputs").placeholder = `blank = ${data.inputs.default}`;
  if (!$("#ap-fields").children.length) addFieldRow("#ap-fields");
}

function kindChanged() {
  const kind = $("#ap-kind").value;
  $("#ap-kind-note").textContent = KIND_NOTES[kind] || "";
  $("#ap-fold-wrap").hidden = kind !== "fold";
  $("#ap-key-wrap").hidden = kind === "fold";
  $("#ap-fields-hint").textContent = kind === "fold" ? "the whole answer"
    : kind === "spans" ? "per span; first_chunk and last_chunk are added"
    : "per item; chunk_id is added";
}

/* Copying a built-in is how one is "edited": built-ins cannot be replaced, so
 * the copy is named apart and runs beside the original. */
function fillAggregateForm(p) {
  $("#ap-fields").replaceChildren();
  if (!p) {
    for (const id of ["#ap-name", "#ap-instruction", "#ap-fold", "#ap-inputs", "#ap-key", "#ap-about"]) $(id).value = "";
    addFieldRow("#ap-fields");
    kindChanged();
    return;
  }
  $("#ap-template").value = p.name;
  $("#ap-name").value = p.builtin ? `my_${p.name}` : p.name;
  $("#ap-kind").value = p.kind;
  $("#ap-instruction").value = p.instruction || "";
  $("#ap-fold").value = p.fold_instruction || "";
  $("#ap-inputs").value = p.inputs || "";
  $("#ap-key").value = p.key || "";
  $("#ap-about").value = p.about || "";
  for (const [name, spec] of Object.entries(p.fields || {})) addFieldRow("#ap-fields", name, spec);
  kindChanged();
}

$("#ap-save").onclick = async () => {
  const kind = $("#ap-kind").value;
  const payload = {
    name: $("#ap-name").value.trim(),
    kind,
    instruction: $("#ap-instruction").value.trim(),
    about: $("#ap-about").value.trim(),
    fields: collectFields("#ap-fields"),
  };
  const optional = { inputs: "#ap-inputs",
                     key: kind === "fold" ? null : "#ap-key",
                     fold_instruction: kind === "fold" ? "#ap-fold" : null };
  for (const [key, id] of Object.entries(optional)) {
    if (id && $(id).value.trim()) payload[key] = $(id).value.trim();
  }
  const state = $("#ap-state");
  const out = $("#ap-out");
  state.replaceChildren(chip("saving", "on"));
  try {
    const made = await postJSON("/aggregate-prompts", payload);
    state.replaceChildren(chip("201 created", "on"));
    out.hidden = false;
    show(out, JSON.stringify(made, null, 2));
    await refreshCaps();
    loadAggregatePrompts();
  } catch (err) {
    state.replaceChildren(chip(`http ${err.status}`, "bad"));
    out.hidden = false;
    show(out, errorText(err));
  }
};

/* -------------------------------------------------------- page 4: search */

/* Every structured field the shapes gave a vocabulary to. A free-text field is
 * filterable in the mechanical sense and useless in practice -- one video
 * produced `cashier`, `customer` and `cashier or customer near checkout` -- so
 * only `one_of` fields are offered, and the list comes from /capabilities. */
function renderStructuredFilters() {
  const box = $("#s-structured");
  const fields = ((CAPS.search || {}).structured_fields) || {};
  box.replaceChildren();
  const names = Object.keys(fields).sort();
  if (!names.length) {
    box.append(el("p", { className: "note", textContent:
      "no shape fixes a vocabulary yet — add one_of to a custom question" }));
    return;
  }
  for (const name of names) {
    const wrap = el("div", { className: "field", style: "flex:1;min-width:150px" });
    wrap.append(el("label", { htmlFor: `sf-${name}`, textContent: name }));
    const select = el("select", { id: `sf-${name}` });
    select.dataset.field = name;
    select.append(el("option", { value: "", textContent: "— any" }));
    for (const v of fields[name]) select.append(el("option", { value: v, textContent: v }));
    wrap.append(select);
    box.append(wrap);
  }
}

/* Scope is a set of videos, so the control is a set of checkboxes. None ticked
 * means every video -- not "nothing", because a scope of none is not a
 * question anyone asks. */
function renderScope(videos, keep) {
  const box = $("#s-scope");
  if (!box) return;
  const chosen = new Set(keep || scopeIds());
  box.replaceChildren(...videos.map(v => {
    const id = `sv-${v.video_id}`;
    const input = el("input", { type: "checkbox", id });
    input.dataset.video = v.video_id;
    input.checked = chosen.has(v.video_id);
    return el("label", { htmlFor: id, style: "font-size:12px" }, [input, " " + v.video_id]);
  }));
}

function scopeIds() {
  return [...document.querySelectorAll("#s-scope [data-video]")]
    .filter(c => c.checked).map(c => c.dataset.video);
}

function searchPayload() {
  const payload = {
    query: $("#s-query").value,
    level: $("#s-level").value,
    moments: parseInt($("#s-moments").value, 10) || 5,
    candidates: parseInt($("#s-candidates").value, 10) || 20,
    index: $("#s-index").value,
  };
  /* Omitted, not sent empty: no `video_ids` means every video, and an empty
   * array would read as "no videos" to anything strict about it. */
  const scope = scopeIds();
  if (scope.length) payload.video_ids = scope;
  const text = (id, key) => {
    const v = $(id).value.trim();
    if (v) payload[key] = v;
  };
  text("#s-question", "question");
  text("#s-strategy", "strategy");
  text("#s-sampler", "sampler");
  /* Blank means the server resolves it exactly as `embed` did -- the one
   * embedder guaranteed to match an index this deployment built by default. */
  text("#s-embedder", "embedder");

  const chunks = $("#s-chunks").value.trim();
  if (chunks) {
    payload.chunk_ids = chunks.split(",").map(c => parseInt(c, 10))
                              .filter(n => !Number.isNaN(n));
  }
  const widen = parseInt($("#s-window").value, 10);
  if (widen) payload.window = widen;
  /* A time window is not a field on a vector: the server resolves seconds to
   * chunk ids through the grid, which is the one place a span is stored. */
  if ($("#s-after").value !== "") payload.after = parseFloat($("#s-after").value);
  if ($("#s-before").value !== "") payload.before = parseFloat($("#s-before").value);

  const structured = {};
  for (const select of document.querySelectorAll("#s-structured [data-field]")) {
    if (select.value) structured[select.dataset.field] = select.value;
  }
  if (Object.keys(structured).length) payload.structured = structured;
  return payload;
}

$("#s-level").onchange = () => {
  /* The moment filters narrow inside a video, so they have nothing to say
   * about which video. Hidden rather than silently ignored. */
  $("#s-filters").hidden = $("#s-level").value === "video";
};

$("#s-clear").onclick = () => {
  for (const id of ["#s-sampler", "#s-chunks", "#s-after", "#s-before"]) $(id).value = "";
  $("#s-question").value = ""; $("#s-strategy").value = "";
  $("#s-window").value = "0"; $("#s-candidates").value = "20";
  for (const select of document.querySelectorAll("#s-structured [data-field]")) select.value = "";
};

$("#s-go").onclick = async () => {
  const state = $("#s-state");
  const box = $("#s-results");
  state.replaceChildren(chip("searching", "on"));
  box.replaceChildren();
  try {
    const out = await postJSON("/search", searchPayload());

    if (out.level === "video") {
      const found = out.videos || [];
      state.replaceChildren(chip(`${found.length} videos`, "on"));
      if (out.ignored) box.append(el("p", { className: "note" }, out.ignored));
      if (!found.length) {
        box.append(el("p", { className: "note" }, out.note || "nothing")); return;
      }
      const table = el("table");
      table.append(el("thead", {}, el("tr", {}, [
        el("th", {}, "video"), el("th", {}, "similarity"), el("th", {}, ""),
      ])));
      const body = el("tbody");
      for (const v of found) {
        body.append(el("tr", {}, [
          el("td", {}, v.video_id),
          el("td", {}, v.similarity.toFixed(4)),
          el("td", {}, el("button", { className: "small", textContent: "moments inside",
            onclick: () => {
              /* The two-step: which video, then which moment inside it --
               * the same endpoint, a narrower scope and the other level. */
              for (const c of document.querySelectorAll("#s-scope [data-video]")) {
                c.checked = c.dataset.video === v.video_id;
              }
              $("#s-level").value = "moment";
              $("#s-level").onchange();
              $("#s-go").click();
            } })),
        ]));
      }
      table.append(body);
      box.append(el("div", { className: "scroll" }, table));
      return;
    }

    const moments = out.moments || [];
    state.replaceChildren(chip(`${moments.length} moments`, "on"));
    if (!moments.length) {
      /* The server's own words: "nothing matched" names the embedder, and an
       * embedder that never indexed these videos is empty in exactly this way. */
      const said = (out.notes || []).filter(n => !n.startsWith("scope is"));
      box.append(el("p", { className: "note" },
        said.length ? said.join(" · ") : "nothing matched those filters")); return;
    }
    const note = (moments[0].notes || []).find(n => n.startsWith("scope is"));
    if (note) box.append(el("p", { className: "note" }, note));

    const found = [...new Set(moments.map(m => m.chunk_id))];
    box.append(el("div", { className: "row", style: "margin-bottom:10px" }, [
      el("button", { className: "small", textContent:
        `search within these ${found.length} chunks`,
        onclick: () => { $("#s-chunks").value = found.join(","); $("#s-go").click(); } }),
      el("button", { className: "small", textContent: "…and their neighbours",
        onclick: () => { $("#s-chunks").value = found.join(",");
                         $("#s-window").value = "1"; $("#s-go").click(); } }),
    ]));

    for (const m of moments) {
      const card = el("div", { className: "panel", style: "margin-bottom:10px" });
      card.append(el("div", { className: "row" }, [
        /* The video is named on every moment: with a scope of several, a chunk
         * id alone does not identify anything. */
        chip(`${m.video_id} · chunk ${m.chunk_id}`),
        el("span", { className: "hint", textContent:
          `${m.start_ts.toFixed(1)}–${m.end_ts.toFixed(1)}s · score ${m.score.toFixed(4)} · ${m.samplers.length} account(s)` }),
      ]));
      for (const [sid, text] of Object.entries(m.descriptions)) {
        const r = (m.ranks || {})[sid] || {};
        card.append(el("h3", {}, [
          sid,
          el("span", { className: "hint", textContent:
            `  dense ${r.dense ?? "–"} · text ${r.text ?? "–"}` }),
        ]));
        card.append(el("p", { className: "note", textContent: text }));
        const st = (m.structured || {})[sid];
        if (st && Object.keys(st).length) {
          card.append(el("pre", { className: "out", style: "margin-top:4px" },
            JSON.stringify(st)));
        }
      }
      box.append(card);
    }
  } catch (err) {
    state.replaceChildren(chip(`http ${err.status || "err"}`, "bad"));
    box.append(el("pre", { className: "out err" }, errorText(err)));
  }
};

/* ---------------------------------------------------------- page 5: data */

/* The file half: every answer this video has on disk, with what it read and
 * who wrote it. Files are the primary store; Postgres is the second copy. */
async function loadAggregateFiles() {
  const box = $("#d-files");
  if (!VIDEO) { box.replaceChildren(el("p", { className: "note", textContent: "Pick a video." })); return; }
  const detail = await api(`/videos/${encodeURIComponent(VIDEO)}`);
  const answers = detail.aggregates || [];
  if (!answers.length) {
    box.replaceChildren(el("p", { className: "note", textContent: "No aggregates written yet." }));
    return;
  }
  const docs = await Promise.all(answers.map(a => api(a.url).then(doc => [a, doc])));
  const t = el("table");
  t.append(el("thead", {}, el("tr", {}, ["answer", "tier", "kind", "reads", "model",
    "version", "fingerprint", ""].map(h => el("th", {}, h)))));
  t.append(el("tbody", {}, docs.map(([a, doc]) => {
    const s = doc.stats || {};
    /* An answer can outlive its definition: a deleted custom prompt leaves its
     * files, and so does an id the aggregators no longer use. */
    const known = (CAPS.aggregators || {})[definitionOf(a.name)];
    const kind = known ? known.kind || "code" : "no longer defined";
    return el("tr", {}, [
      el("td", { className: "wrap" }, a.name),
      el("td", {}, doc.tier),
      el("td", { className: "wrap" }, kind),
      el("td", { className: "wrap" }, s.inputs || "—"),
      el("td", { className: "wrap" }, s.model || "—"),
      el("td", {}, s.version || "—"),
      el("td", {}, (doc.inputs_fingerprint || "").slice(0, 10)),
      el("td", {}, el("button", { className: "small", textContent: "view",
        onclick: () => { const v = $("#d-file-view"); v.hidden = false;
                         show(v, JSON.stringify(doc, null, 2)); } })),
    ]);
  })));
  box.replaceChildren(t);
}

let TABLES = [];
let OPS = [];

/* Quick filters for the tables an aggregate run fills. Each is an ordinary
 * filter row, so it can be edited before querying. */
const PRESETS = {
  aggregates: [["aggregate_id", "eq", "summary"], ["tier", "eq", "llm"]],
  entities: [["aggregate_id", "eq", "entities:people"], ["appearances", "gte", "2"]],
  entity_mentions: [["chunk_id", "eq", "0"], ["doubt", "is", "not.null"]],
  aggregate_definitions: [["kind", "eq", "fold"], ["name", "eq", "summary"]],
};

async function loadDbStatus() {
  const box = $("#db-status");
  box.replaceChildren("checking...");
  let s;
  try { s = await api("/db/status"); }
  catch (err) { box.replaceChildren(el("pre", { className: "out err" }, errorText(err))); return; }
  if (!s.reachable) {
    box.replaceChildren(el("pre", { className: "out err" },
      `not reachable\nschema ${s.schema}\n${s.error || ""}`));
  } else {
    const table = el("table");
    table.append(el("thead", {}, el("tr", {}, [el("th", {}, "table"), el("th", {}, "rows")])));
    table.append(el("tbody", {}, Object.entries(s.counts).map(([name, n]) =>
      el("tr", {}, [el("td", {}, name), el("td", {}, String(n))]))));
    box.replaceChildren(el("div", { className: "row" },
      [chip(`schema ${s.schema}`), el("span", { className: "hint",
        textContent: " read under the publishable key — what a reader with the read grants sees" })]),
      el("div", { className: "scroll", style: "margin-top:8px" }, table));
  }

  const listed = await api("/db/tables");
  TABLES = listed.tables;
  OPS = listed.ops;
  const select = $("#d-table");
  const keep = select.value;
  select.replaceChildren(...TABLES.map(t =>
    el("option", { value: t.name, textContent: t.name, title: t.about })));
  if (keep) select.value = keep;
  select.onchange = tableChanged;
  tableChanged();
}

function tableChanged() {
  const t = TABLES.find(x => x.name === $("#d-table").value);
  $("#d-about").textContent = t ? `${t.about}${t.order.length ? ` · ordered by ${t.order.join(", ")}` : ""}` : "";
  $("#d-columns").replaceChildren(...((t && t.columns) || []).map(c => el("option", { value: c })));
  $("#d-filters").replaceChildren();
  const presets = PRESETS[t && t.name] || [];
  $("#d-presets").replaceChildren(...presets.map(([c, op, v]) =>
    el("button", { className: "small", type: "button", textContent: `${c} ${op} ${v}`,
      onclick: () => addFilterRow(c, op, v) })));
}

function addFilterRow(column = "", op = "eq", value = "") {
  const row = el("div", { className: "filterrow" });
  const col = el("input", { type: "text", placeholder: "column", value: column });
  col.setAttribute("list", "d-columns");
  const opSelect = el("select");
  opSelect.append(...(OPS.length ? OPS : ["eq"]).map(o => el("option", { value: o, textContent: o })));
  opSelect.value = op;
  const val = el("input", { type: "text", placeholder: "value; comma-separated for in", value });
  const drop = el("button", { className: "small", textContent: "×", type: "button",
    onclick: () => row.remove() });
  row.append(col, opSelect, val, drop);
  row.dataset.filter = "1";
  $("#d-filters").append(row);
}
$("#d-add-filter").onclick = () => addFilterRow();

$("#d-go").onclick = async () => {
  const box = $("#d-results");
  box.replaceChildren("querying...");
  const table = $("#d-table").value;
  const meta = TABLES.find(x => x.name === table);
  const payload = {
    table,
    limit: parseInt($("#d-limit").value, 10) || 25,
    include_heavy: $("#d-heavy").checked,
    filters: [],
  };
  if ($("#d-order").value.trim()) payload.order = $("#d-order").value.trim();
  for (const row of document.querySelectorAll("#d-filters [data-filter]")) {
    const [col, op, val] = row.children;
    if (col.value.trim()) payload.filters.push({ column: col.value.trim(), op: op.value, value: val.value });
  }
  /* `prompts` and `aggregate_definitions` are keyed by (name, version), not by
   * a video, so a video filter would 42703 them. The deployed column list says
   * which tables have one; the names cover a deployment it could not read. */
  const keyed = meta && meta.columns && meta.columns.length
    ? meta.columns.includes("video_id")
    : !["prompts", "aggregate_definitions"].includes(table);
  if ($("#d-thisvideo").checked && VIDEO && keyed) {
    payload.filters.push({ column: "video_id", op: "eq", value: VIDEO });
  }
  try {
    const out = await postJSON("/db/query", payload);
    const cols = out.rows.length ? Object.keys(out.rows[0]) : [];
    const t = el("table");
    t.append(el("thead", {}, el("tr", {}, cols.map(c => el("th", {}, c)))));
    t.append(el("tbody", {}, out.rows.map(row => el("tr", {}, cols.map(c => {
      const v = row[c];
      const text = v === null ? "—" : typeof v === "object" ? JSON.stringify(v) : String(v);
      return el("td", { className: "wrap" }, text.length > 400 ? text.slice(0, 400) + "…" : text);
    })))));
    box.replaceChildren(
      el("p", { className: "note", textContent:
        `${out.rows.length} of ${out.count} rows · select: ${out.select}` }),
      el("div", { className: "scroll" }, t));
  } catch (err) {
    box.replaceChildren(el("pre", { className: "out err" }, errorText(err)));
  }
};

/* ----------------------------------------------------------------- boot */

async function refreshCaps() {
  CAPS = await api("/capabilities");
  $("#s-index").replaceChildren(...CAPS.indexes.map(i =>
    el("option", { value: i, textContent: i })));
  $("#s-embedder").placeholder = (CAPS.defaults || {}).embedder || "";
  $("#s-embedders").replaceChildren(...CAPS.embedders.map(e => el("option", { value: e })));
  $("#s-question").replaceChildren(
    el("option", { value: "", textContent: "— any question" }),
    ...CAPS.prompts.map(q => el("option", { value: q, textContent: q })));
  $("#s-strategy").replaceChildren(
    el("option", { value: "", textContent: "— any sampler" }),
    ...CAPS.samplers.map(x => el("option", { value: x, textContent: x })),
    el("option", { value: "transcript", textContent: "transcript" }));
  renderStructuredFilters();
  if (PICKED) renderParams();
  renderSteps();
  if (PAGE === "aggregates") renderAggregatePage();
}

(async function boot() {
  try {
    await refreshCaps();
    await loadVideos();
    PICKED = ragComponents().find(c => c !== "media") || null;
    renderSteps();
    renderParams();
  } catch (err) {
    show($("#output"), `could not reach the API: ${err.message}`);
  }
})();
