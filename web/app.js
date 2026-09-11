/* One page over the API. No build step, no framework, one origin.
 *
 * NOTHING ABOUT THE PIPELINE IS WRITTEN DOWN HERE. Every form is generated
 * from `GET /capabilities`: `parameters.<component>` names each setting with
 * its type and default, and the registries supply the legal values. A restated
 * list is a second copy to keep in step, and when it drifts a form offers a
 * parameter the component does not take or hides one it does.
 *
 * A blank field is OMITTED rather than sent as null, so the component's own
 * default applies and this page holds no second copy of it.
 *
 * The one widget written by hand is the custom-shape builder, because a field
 * is a name, a type, a description and optionally a nested map -- and a text
 * box does not say so.
 */

const $ = (sel, root = document) => root.querySelector(sel);
const el = (tag, props = {}, kids = []) => {
  const node = Object.assign(document.createElement(tag), props);
  for (const kid of [].concat(kids)) {
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

let CAPS = null;      // GET /capabilities
let VIDEO = null;     // the selected video_id
let DETAIL = null;    // GET /videos/{id}, for has_audio / has_video
let PICKED = null;    // the component whose form is showing

/* ------------------------------------------------------------------ tabs */

for (const button of document.querySelectorAll("#tabs button")) {
  button.onclick = () => {
    for (const other of document.querySelectorAll("#tabs button")) {
      other.setAttribute("aria-selected", String(other === button));
      $(`#tab-${other.dataset.tab}`).hidden = other !== button;
    }
    if (button.dataset.tab === "data") loadDbStatus();
    if (button.dataset.tab === "prompts") loadPrompts();
  };
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
  await loadVideoDetail(videos);
  renderScope(videos.filter(v => v.artifacts.includes("embedded")));
  renderSteps();
}

async function loadVideoDetail(videos) {
  const facts = $("#video-facts");
  if (!VIDEO) { facts.textContent = ""; DETAIL = null; return; }
  const found = (videos || []).find(v => v.video_id === VIDEO);
  DETAIL = found || null;
  if (!found) { facts.textContent = ""; return; }
  const bits = [`${(found.duration_s || 0).toFixed(1)}s`];
  bits.push(found.has_video ? "video" : "no video");
  bits.push(found.has_audio ? "audio" : "no audio");
  if (found.chunks) bits.push(`${found.chunks} chunks · ${found.policy}`);
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

/* ------------------------------------------------------------ components */

/* Which stages this file cannot support, decided AFTER `media` has read it.
 * Whether a file carries a soundtrack is a property of the file, not of the
 * request, so a stage needing a stream it has none of is marked skipped with
 * the reason rather than queued to fail. */
function unusable(name) {
  if (!DETAIL) return null;
  if (!DETAIL.has_audio && (name === "audio" || name === "cut"))
    return "the file carries no audio stream";
  if (!DETAIL.has_video && (name === "video" || name === "describe"))
    return "the file carries no video stream";
  return null;
}

function renderSteps() {
  const list = $("#steps");
  list.replaceChildren();
  (CAPS.components || []).forEach((name, i) => {
    const why = unusable(name);
    const isMedia = name === "media";
    const button = el("button", { type: "button" }, [
      el("span", { className: "n", textContent: String(i + 1) }),
      el("span", { textContent: name }),
    ]);
    if (isMedia) button.append(el("span", { className: "mark", textContent: "upload" }));
    else if (why) button.append(el("span", { className: "mark", textContent: "n/a" }));
    button.setAttribute("aria-selected", String(name === PICKED));
    button.onclick = () => { PICKED = name; renderSteps(); renderParams(); };
    list.append(el("li", {}, button));
  });
  const why = PICKED && unusable(PICKED);
  $("#plan-note").textContent = why
    ? `${PICKED}: skipped — ${why}`
    : "Order is yours on this path. workflow.py is the reference.";
}

/* Legal values, from the registries rather than from a list here. */
function choicesFor(param) {
  const c = CAPS;
  const map = {
    policy: c.policies, describer: c.describers, embedder: c.embedders,
    tier: c.tiers, transcriber: c.transcribers, diarizer: c.diarizers,
    store_scope: ["sampled", "decimated"],
    sink: ["file", "supabase", "file,supabase"],
    index_name: [...c.indexes, c.indexes.join(",")],
  };
  return map[param.name] || null;
}

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
    let input;
    if (p.type === "bool") {
      input = el("select", { id });
      input.append(el("option", { value: "", textContent: `— default (${p.default})` }),
                   el("option", { value: "true", textContent: "true" }),
                   el("option", { value: "false", textContent: "false" }));
    } else if (choices) {
      input = el("select", { id });
      input.append(el("option", { value: "", textContent: "— default" }));
      for (const c of choices) input.append(el("option", { value: c, textContent: c }));
      if (p.required && p.default === null) input.value = "";
    } else if (p.type === "int" || p.type === "float") {
      input = el("input", { type: "number", id, step: p.type === "float" ? "any" : "1",
                            placeholder: p.default === null ? "" : String(p.default) });
    } else {
      input = el("input", { type: "text", id,
                            placeholder: p.default === null ? "" : String(p.default) });
    }
    input.dataset.kind = p.type;
    input.dataset.name = p.name;
    field.append(input);

    if (p.name === "sampler") {
      field.append(el("p", { className: "note", textContent:
        `samplers: ${CAPS.samplers.join(", ")} · questions: ${CAPS.prompts.join(", ")}`
        + " — pair as name:question, or name:[q1,q2] for one pass answering two." }));
    }
    box.append(field);
  }
  if (!params.length) box.append(el("p", { className: "note", textContent: "No parameters." }));
}

/* A blank field is omitted, never sent as null: the component's own default
 * then applies, and this page holds no second copy of it. */
function collectParams() {
  const out = {};
  for (const input of document.querySelectorAll("#params [data-name]")) {
    const raw = input.value.trim();
    if (raw === "") continue;
    const kind = input.dataset.kind;
    if (kind === "bool") out[input.dataset.name] = raw === "true";
    else if (kind === "int") out[input.dataset.name] = parseInt(raw, 10);
    else if (kind === "float") out[input.dataset.name] = parseFloat(raw);
    else if (input.dataset.name === "samplers")
      out.samplers = raw.split(",").map(s => s.trim()).filter(Boolean);
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
    await pollJob(job.id, state);
  } catch (err) {
    state.replaceChildren(chip(`http ${err.status || "error"}`, "bad"));
    show($("#output"), typeof err.detail === "object"
      ? JSON.stringify(err.detail, null, 2) : err.message);
  } finally {
    $("#run").disabled = false;
    loadVideos(VIDEO);
  }
};

/* `stage` is what is RUNNING; `history` is what has finished. The workflow
 * announces a component twice for exactly this reason, so a poller does not
 * read the previous component's name through the longest stage of a run. */
async function pollJob(id, state) {
  for (;;) {
    const job = await api(`/jobs/${id}`);
    state.replaceChildren(
      chip(job.state, job.state === "failed" ? "bad" : "on"),
      el("span", { className: "hint", textContent:
        ` ${job.stage || ""} ${job.elapsed_s != null ? job.elapsed_s + "s" : ""}` }));
    show($("#output"), JSON.stringify(job.error ? {
      state: job.state, error: job.error, traceback: (job.detail || {}).traceback,
    } : {
      state: job.state, stage: job.stage, elapsed_s: job.elapsed_s,
      detail: job.detail,
    }, null, 2));
    if (job.state === "done") { await showArtifacts(); return; }
    if (job.state === "failed") return;
    await new Promise(r => setTimeout(r, 1000));
  }
}

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
  if (detail.aggregates.length) {
    box.append(el("h3", {}, "Aggregates"));
    const row = el("div", { className: "row" });
    for (const a of detail.aggregates) {
      row.append(el("button", { className: "small", textContent: a.name, title: a.about,
        onclick: async () => show($("#output"), JSON.stringify(await api(a.url), null, 2)) }));
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

/* -------------------------------------------------------------- prompts */

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
    if (custom && !$("#p-fields").children.length) addFieldRow();
  };
}

function addFieldRow() {
  const row = el("div", { className: "fieldrow" });
  const name = el("input", { type: "text", placeholder: "hazards" });
  const type = el("select");
  type.append(el("option", { value: "list", textContent: "list" }),
              el("option", { value: "text", textContent: "text" }));
  const about = el("input", { type: "text", placeholder: "what to put in it" });
  const extra = el("input", { type: "text", placeholder: "of: key: desc, ... | one_of: a, b" });
  const drop = el("button", { className: "small", textContent: "×", type: "button",
    onclick: () => row.remove() });
  row.append(name, type, about, extra, drop);
  row.dataset.row = "1";
  $("#p-fields").append(row);
}
$("#p-add-field").onclick = addFieldRow;

/* The builder, not raw JSON Schema. The call goes out with `strict: true`,
 * whose subset is narrow -- a schema the API refuses would fail after the
 * frames are read, with the request about to be paid for. */
function collectFields() {
  const fields = {};
  for (const row of document.querySelectorAll("#p-fields [data-row]")) {
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
    payload.fields = collectFields();
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
    show(out, typeof err.detail === "object"
      ? JSON.stringify(err.detail, null, 2) : err.message);
  }
};

/* --------------------------------------------------------------- search */

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
      "no shape fixes a vocabulary yet \u2014 add one_of to a custom question" }));
    return;
  }
  for (const name of names) {
    const wrap = el("div", { className: "field", style: "flex:1;min-width:150px" });
    wrap.append(el("label", { htmlFor: `sf-${name}`, textContent: name }));
    const select = el("select", { id: `sf-${name}` });
    select.dataset.field = name;
    select.append(el("option", { value: "", textContent: "\u2014 any" }));
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
      box.append(el("p", { className: "note" },
        "nothing matched those filters")); return;
    }
    const note = (moments[0].notes || []).find(n => n.startsWith("scope is"));
    if (note) box.append(el("p", { className: "note" }, note));

    const found = [...new Set(moments.map(m => m.chunk_id))];
    box.append(el("div", { className: "row", style: "margin-bottom:10px" }, [
      el("button", { className: "small", textContent:
        `search within these ${found.length} chunks`,
        onclick: () => { $("#s-chunks").value = found.join(","); $("#s-go").click(); } }),
      el("button", { className: "small", textContent: "\u2026and their neighbours",
        onclick: () => { $("#s-chunks").value = found.join(",");
                         $("#s-window").value = "1"; $("#s-go").click(); } }),
    ]));

    for (const m of moments) {
      const card = el("div", { className: "panel", style: "margin-bottom:10px" });
      card.append(el("div", { className: "row" }, [
        /* The video is named on every moment: with a scope of several, a chunk
         * id alone does not identify anything. */
        chip(`${m.video_id} \u00b7 chunk ${m.chunk_id}`),
        el("span", { className: "hint", textContent:
          `${m.start_ts.toFixed(1)}\u2013${m.end_ts.toFixed(1)}s \u00b7 score ${m.score.toFixed(4)} \u00b7 ${m.samplers.length} account(s)` }),
      ]));
      for (const [sid, text] of Object.entries(m.descriptions)) {
        const r = (m.ranks || {})[sid] || {};
        card.append(el("h3", {}, [
          sid,
          el("span", { className: "hint", textContent:
            `  dense ${r.dense ?? "\u2013"} \u00b7 text ${r.text ?? "\u2013"}` }),
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
    box.append(el("pre", { className: "out err" },
      typeof err.detail === "object" ? JSON.stringify(err.detail, null, 2) : err.message));
  }
};

/* ----------------------------------------------------------------- data */

async function loadDbStatus() {
  const box = $("#db-status");
  box.replaceChildren("checking...");
  const s = await api("/db/status");
  if (!s.reachable) {
    box.replaceChildren(el("pre", { className: "out err" },
      `not reachable\nschema ${s.schema}\n${s.error || ""}`));
    return;
  }
  const table = el("table");
  table.append(el("thead", {}, el("tr", {}, [el("th", {}, "table"), el("th", {}, "rows")])));
  const body = el("tbody");
  for (const [name, n] of Object.entries(s.counts)) {
    body.append(el("tr", {}, [el("td", {}, name), el("td", {}, String(n))]));
  }
  table.append(body);
  box.replaceChildren(el("div", { className: "row" },
    [chip(`schema ${s.schema}`), el("span", { className: "hint",
      textContent: " read under the publishable key — what a reader with the read grants sees" })]),
    el("div", { className: "scroll", style: "margin-top:8px" }, table));

  const { tables } = await api("/db/tables");
  const select = $("#d-table");
  const keep = select.value;
  select.replaceChildren(...tables.map(t =>
    el("option", { value: t.name, textContent: t.name, title: t.about })));
  if (keep) select.value = keep;
  select.onchange = () => {
    const t = tables.find(x => x.name === select.value);
    $("#d-about").textContent = t ? t.about : "";
  };
  select.onchange();
}

$("#d-go").onclick = async () => {
  const box = $("#d-results");
  box.replaceChildren("querying...");
  const table = $("#d-table").value;
  const payload = {
    table,
    limit: parseInt($("#d-limit").value, 10) || 25,
    include_heavy: $("#d-heavy").checked,
    filters: [],
  };
  /* `prompts` is the one table not keyed by a video -- it is keyed by
   * (name, version) -- so a video filter would 42703 it. */
  if ($("#d-thisvideo").checked && VIDEO && table !== "prompts") {
    payload.filters.push({ column: "video_id", op: "eq", value: VIDEO });
  }
  try {
    const out = await postJSON("/db/query", payload);
    const cols = out.rows.length ? Object.keys(out.rows[0]) : [];
    const t = el("table");
    t.append(el("thead", {}, el("tr", {}, cols.map(c => el("th", {}, c)))));
    const body = el("tbody");
    for (const row of out.rows) {
      body.append(el("tr", {}, cols.map(c => {
        const v = row[c];
        const text = v === null ? "—"
          : typeof v === "object" ? JSON.stringify(v) : String(v);
        return el("td", { className: "wrap" }, text.length > 400 ? text.slice(0, 400) + "…" : text);
      })));
    }
    t.append(body);
    box.replaceChildren(
      el("p", { className: "note", textContent:
        `${out.rows.length} of ${out.count} rows · select: ${out.select}` }),
      el("div", { className: "scroll" }, t));
  } catch (err) {
    box.replaceChildren(el("pre", { className: "out err" },
      typeof err.detail === "object" ? JSON.stringify(err.detail, null, 2) : err.message));
  }
};

/* ----------------------------------------------------------------- boot */

async function refreshCaps() {
  CAPS = await api("/capabilities");
  $("#s-index").replaceChildren(...CAPS.indexes.map(i =>
    el("option", { value: i, textContent: i })));
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
}

(async function boot() {
  try {
    await refreshCaps();
    await loadVideos();
    PICKED = CAPS.components.find(c => c !== "media") || null;
    renderSteps();
    renderParams();
  } catch (err) {
    show($("#output"), `could not reach the API: ${err.message}`);
  }
})();
