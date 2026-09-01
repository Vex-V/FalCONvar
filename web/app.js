/* FalCONvar - a thin browser client over the API.
 *
 * Deliberately no framework and no build step: it exists to exercise the API
 * and to make the pipeline's behaviour visible, so being readable next to the
 * routes it calls matters more than anything a bundler buys.
 *
 * Everything the form can offer comes from GET /capabilities, so a sampler
 * registered in ver2 shows up here without this file being edited.
 */

const $ = (s) => document.querySelector(s);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};

/* Why each sampler exists, in the words the project uses. Shown under the
 * name because "yolo" does not tell you it is asking about people. */
const WHY = {
  clip: "has the scene changed?",
  yolo: "have the people changed?",
  objects: "have things moved or appeared?",
  text: "has the text changed?",
  uniform: "every N seconds, on a clock",
  overview: "every N seconds, prose only",
  "uniform:text": "read the screen on a clock",
};
const GRID = {
  uniform: "arithmetic; neither modality decides",
  scene: "the video pass, on frame content",
  vad: "the audio pass, in the gaps between speech",
  speaker: "the audio pass, where the voice changes",
};

let CAPS = null;

async function api(path, options) {
  const res = await fetch(path, options);
  const body = res.status === 204 ? null : await res.json().catch(() => null);
  if (!res.ok) {
    const detail = body && body.detail !== undefined ? body.detail : res.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return body;
}

/* ------------------------------------------------------------------ chrome */
function showTab(name) {
  document.querySelectorAll("#tabs button").forEach((x) =>
    x.classList.toggle("on", x.dataset.tab === name));
  document.querySelectorAll(".tab").forEach((x) =>
    x.classList.toggle("on", x.id === name));
}

document.querySelectorAll("#tabs button").forEach((b) => {
  b.onclick = () => {
    showTab(b.dataset.tab);
    if (b.dataset.tab === "library") loadVideos();
    if (b.dataset.tab === "jobs") loadJobs();
  };
});

async function health() {
  try {
    const h = await api("/health");
    const pill = $("#health");
    pill.textContent = h.queued ? `${h.queued} queued` : "ready";
    pill.className = "pill ok";
  } catch {
    $("#health").textContent = "api unreachable";
    $("#health").className = "pill bad";
  }
}

/* ------------------------------------------------------------ capabilities */
async function boot() {
  await health();
  CAPS = await api("/capabilities");

  const chips = $("#samplers");
  const offered = [...CAPS.samplers, "uniform:text"];
  offered.forEach((name) => {
    const chip = el("label", "chip");
    const box = el("input");
    box.type = "checkbox";
    box.value = name;
    box.checked = name === "clip";
    box.onchange = () => chip.classList.toggle("on", box.checked);
    chip.classList.toggle("on", box.checked);
    chip.append(box, el("span", null, name), el("span", "why", WHY[name] || ""));
    chips.append(chip);
  });

  const grid = $("#chunking");
  CAPS.chunking.forEach((name, i) => {
    const chip = el("label", "chip" + (i === 0 ? " on" : ""));
    const radio = el("input");
    radio.type = "radio";
    radio.name = "chunking";
    radio.value = name;
    radio.checked = i === 0;
    radio.onchange = () =>
      document.querySelectorAll("#chunking .chip").forEach((c) =>
        c.classList.toggle("on", c.querySelector("input").checked));
    chip.append(radio, el("span", null, name), el("span", "why", GRID[name] || ""));
    grid.append(chip);
  });

  const sampler = $("#sampler");
  ["transcript", ...CAPS.samplers].forEach((name) => {
    const o = el("option", null, name === "transcript" ? "transcript (what was said)" : name);
    o.value = name;
    sampler.append(o);
  });

  await refreshScopes();
}

async function refreshScopes() {
  const { videos } = await api("/videos");
  const scope = $("#scope");
  const chosen = scope.value;
  scope.innerHTML = "";
  scope.append(Object.assign(el("option", null, "every video"), { value: "" }));
  videos.forEach((v) =>
    scope.append(Object.assign(el("option", null, v.video_id), { value: v.video_id })));
  scope.value = chosen;
  return videos;
}

/* ---------------------------------------------------------------- search */
$("#search-form").onsubmit = async (e) => {
  e.preventDefault();
  const out = $("#results");
  out.innerHTML = "";
  out.append(el("p", "note", "searching…"));
  try {
    const body = await api("/search", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        query: $("#q").value,
        video_id: $("#scope").value || null,
        sampler: $("#sampler").value || null,
        moments: 5,
      }),
    });
    renderMoments(body.moments, out);
  } catch (err) {
    out.innerHTML = "";
    out.append(el("div", "error", "search failed: " + err.message));
  }
};

function renderMoments(moments, out) {
  out.innerHTML = "";
  if (!moments.length) {
    out.append(el("p", "note", "no matches. Has this video been embedded?"));
    return;
  }
  moments.forEach((m, i) => {
    const card = el("div", "moment");
    const head = el("h3");
    head.append(
      el("span", "rank", "#" + (i + 1)),
      el("span", null, m.video_id),
      el("span", "span", `${m.start_ts.toFixed(1)}–${m.end_ts.toFixed(1)}s`),
      el("span", "sam", m.samplers.join(" + ")),
      el("span", "score", "score " + m.score.toFixed(4)),
    );
    card.append(head);

    /* One block per account of this window. Two independent accounts both
     * matching is the thing the per-sampler split exists to make visible. */
    Object.entries(m.descriptions).forEach(([who, text]) => {
      const block = el("div", "acct");
      block.append(el("div", "who", who), el("p", null, text));
      card.append(block);
    });

    if (m.frame_indexes.length) {
      const strip = el("div", "frames");
      m.frame_indexes.slice(0, 12).forEach((idx) => {
        const img = el("img");
        img.src = `/videos/${m.video_id}/frames/${idx}`;
        img.alt = `frame ${idx}`;
        img.loading = "lazy";
        img.onclick = () => {
          $("#lightbox-img").src = img.src;
          $("#lightbox").hidden = false;
        };
        strip.append(img);
      });
      card.append(strip);
    }
    out.append(card);
  });
}

$("#lightbox").onclick = () => ($("#lightbox").hidden = true);

/* ---------------------------------------------------------------- upload */
$("#file").onchange = (e) => {
  const f = e.target.files[0];
  $("#file-name").textContent = f
    ? `${f.name} — ${(f.size / 1e6).toFixed(1)} MB`
    : "choose a media file…";
};

$("#upload-form").onsubmit = async (e) => {
  e.preventDefault();
  const status = $("#upload-status");
  const file = $("#file").files[0];
  if (!file) return;

  const chosen = [...document.querySelectorAll("#samplers input:checked")]
    .map((i) => i.value);
  if (!chosen.length) {
    status.innerHTML = "";
    status.append(el("div", "error", "pick at least one sampler"));
    return;
  }

  const form = new FormData();
  form.append("file", file);
  form.append("samplers", chosen.join(","));
  form.append("chunking", document.querySelector("#chunking input:checked").value);
  form.append("chunk_duration", $("#chunk_duration").value);
  form.append("every_seconds", $("#every_seconds").value);
  form.append("vocabulary", $("#vocabulary").value);
  form.append("use_audio", $("#use_audio").checked);
  form.append("frame_store", $("#frame_store").checked);
  form.append("sinks", $("#sink_supabase").checked ? "file,supabase" : "file");

  $("#go").disabled = true;
  status.innerHTML = "";
  try {
    const body = await api("/videos", { method: "POST", body: form });
    await follow(body.job.id, status, "processing " + body.video_id);
    await refreshScopes();
  } catch (err) {
    status.append(el("div", "error", err.message));
  } finally {
    $("#go").disabled = false;
  }
};

/* Poll one job to completion, rendering each stage as it arrives. This is the
 * client half of `on_progress`: the CLI prints these lines, the browser draws
 * them, and the pipeline knows about neither. */
async function follow(jobId, host, label) {
  const box = el("div", "job");
  host.prepend(box);
  for (;;) {
    const job = await api("/jobs/" + jobId);
    box.innerHTML = "";
    const top = el("div", "top");
    top.append(
      el("span", "state " + job.state, job.state),
      el("strong", null, label),
      el("span", "id", job.id),
      el("span", "id", job.elapsed_s != null ? job.elapsed_s + "s" : ""),
    );
    box.append(top);
    if (job.history.length) {
      box.append(el("pre", null,
        job.history.map((h) => {
          const { stage, at, ...rest } = h;
          return `${stage.padEnd(8)} ${JSON.stringify(rest)}`;
        }).join("\n")));
    }
    if (job.state === "failed") {
      box.append(el("div", "error", job.error || "failed"));
      return job;
    }
    if (job.state === "done") {
      box.append(el("pre", null, JSON.stringify(job.result, null, 2)));
      return job;
    }
    await new Promise((r) => setTimeout(r, 1200));
    health();
  }
}

/* --------------------------------------------------------------- library */
async function loadVideos() {
  const host = $("#videos");
  host.innerHTML = "";
  const videos = await refreshScopes();
  if (!videos.length) {
    host.append(el("p", "note", "nothing processed yet."));
    return;
  }
  videos.forEach((v) => {
    const card = el("div", "card");
    const left = el("div");
    left.append(el("h3", null, v.video_id));
    left.append(el("div", "meta",
      `${v.chunks ?? "?"} chunks · ${v.policy ?? "?"} grid · ` +
      `${v.frames} frames · ${v.timeline_fingerprint ?? ""}`));
    const has = el("div", "has");
    Object.entries(v.has).forEach(([name, yes]) =>
      has.append(el("span", yes ? "yes" : "", name)));
    left.append(has);

    const actions = el("div", "actions");
    const describe = el("button", null, "Describe");
    describe.onclick = () => queue("/describe", { video_id: v.video_id }, v.video_id);
    const embed = el("button", null, "Embed");
    embed.onclick = () => queue("/embed", { video_id: v.video_id }, v.video_id);
    actions.append(describe, embed);

    card.append(left, actions);
    host.append(card);
  });
}

async function queue(path, payload, label) {
  // Switch tabs, then AWAIT the list render before prepending the live box.
  // Clicking the tab instead fires loadJobs() unawaited, and its
  // `innerHTML = ""` detaches the box follow() has already inserted -- which
  // then keeps polling and updating a node no longer in the document, leaving
  // the stale snapshot on screen reading "queued" forever.
  showTab("jobs");
  await loadJobs();
  const host = $("#job-list");
  try {
    const body = await api(path, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
    });
    await follow(body.job.id, host, `${path.slice(1)} ${label}`);
    loadVideos();
  } catch (err) {
    host.prepend(el("div", "error", err.message));
  }
}

/* ------------------------------------------------------------------ jobs */
async function loadJobs() {
  const body = await api("/jobs");
  $("#jobs-note").textContent = body.note;
  const host = $("#job-list");
  host.innerHTML = "";
  if (!body.jobs.length) {
    host.append(el("p", "note", "no jobs this session."));
    return;
  }
  body.jobs.forEach((job) => {
    const box = el("div", "job");
    const top = el("div", "top");
    top.append(
      el("span", "state " + job.state, job.state),
      el("strong", null, `${job.kind} ${job.video_id}`),
      el("span", "id", job.id),
      el("span", "id", job.elapsed_s != null ? job.elapsed_s + "s" : ""),
    );
    box.append(top);
    if (job.error) box.append(el("div", "error", job.error));
    else if (job.result) box.append(el("pre", null, JSON.stringify(job.result, null, 2)));
    host.append(box);
  });
}

boot().catch((err) => {
  $("#health").textContent = "boot failed";
  $("#health").className = "pill bad";
  $("#results").append(el("div", "error", err.message));
});
setInterval(health, 5000);
