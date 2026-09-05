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
  uniform: "every Nth decimated frame",
  overview: "every Nth frame, prose only",
  "uniform:text": "read the screen on a stride",
  "uniform:overview": "a few sentences, on a stride",
};
const GRID = {
  uniform: "arithmetic; neither modality decides",
  scene: "the video pass, on frame content",
  vad: "the audio pass, in the gaps between speech",
  speaker: "the audio pass, where the voice changes",
};
/* Which stream a policy's boundaries come from. `uniform` names none: the
 * arithmetic is identical on both sides, so it is the one policy available
 * whatever is switched on. */
const GRID_NEEDS = { scene: "video", vad: "audio", speaker: "audio" };

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
    if (b.dataset.tab === "digest") fillDigestVideos();
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
  // Strategies, then the pairings the server offers. `uniform:text` used to be
  // hard-coded here; the list moved to /capabilities so a pairing can be added
  // without editing JavaScript -- the same reason the sampler chips are read
  // from a registry rather than written out.
  const offered = [...CAPS.samplers, ...(CAPS.pairings || [])];
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
  CAPS.chunking.forEach((name) => {
    const chip = el("label", "chip" + (name === CAPS.defaults.chunking ? " on" : ""));
    // Which stream has to be running for this policy to be derivable. The
    // same rule `orchestrate.validate` enforces, stated once more here so the
    // form cannot offer a combination the server will reject.
    chip.dataset.needs = GRID_NEEDS[name] || "";
    const radio = el("input");
    radio.type = "radio";
    radio.name = "chunking";
    radio.value = name;
    radio.checked = name === CAPS.defaults.chunking;
    radio.onchange = () =>
      document.querySelectorAll("#chunking .chip").forEach((c) =>
        c.classList.toggle("on", c.querySelector("input").checked));
    chip.append(radio, el("span", null, name), el("span", "why", GRID[name] || ""));
    grid.append(chip);
  });

  const transcribers = $("#transcriber");
  CAPS.transcribers.forEach((name) => {
    const o = el("option", null,
      name === "stub" ? "stub — fake output, for testing the wiring" : name);
    o.value = name;
    // The registry is alphabetical, so the first entry is `stub`. Selecting
    // the server's declared default instead is the difference between
    // transcribing a video and filling its transcript with placeholders --
    // which runs, succeeds, and reports nothing.
    o.selected = name === CAPS.defaults.transcriber;
    transcribers.append(o);
  });

  $("#use_video").onchange = modes;
  $("#use_audio").onchange = modes;
  $("#diarize").onchange = modes;
  modes();

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
  const unit = $("#unit").value;
  try {
    if (unit === "videos") {
      const body = await api("/search/videos", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ query: $("#q").value, limit: 8 }),
      });
      renderVideos(body.videos, out);
      return;
    }
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

/* Whole videos rather than moments. A different unit, so a different card:
 * there is no span to play and no frame to show, only what the video is. */
function renderVideos(videos, out) {
  out.innerHTML = "";
  if (!videos.length) {
    out.append(el("p", "note",
      "no videos matched. Has a summary been built and embedded? " +
      "Digest -> tier llm."));
    return;
  }
  videos.forEach((v, i) => {
    const card = el("div", "moment");
    const head = el("h3");
    const ranks = [v.vector_rank ? "v" + v.vector_rank : null,
                   v.text_rank ? "t" + v.text_rank : null].filter(Boolean);
    head.append(el("span", "rank", "#" + (i + 1)), el("span", null, v.video_id),
                el("span", "sam", ranks.join(",")),
                el("span", "score", "score " + v.score.toFixed(4)));
    card.append(head, el("p", null, v.summary));
    const topics = (v.structured || {}).topics || [];
    if (topics.length) {
      const strip = el("div", "chips");
      topics.forEach((t) => strip.append(el("span", "chip", t)));
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

/* Show only what the chosen streams actually read, and disable the grid
 * policies that stream cannot produce.
 *
 * This is the form half of `orchestrate.validate`, and duplicating the rule is
 * deliberate: the server still enforces it, because a client is not a
 * guarantee, but a user should not have to upload a file to be told that
 * `scene` boundaries are found by a pass they turned off. */
function modes() {
  const video = $("#use_video").checked;
  const audio = $("#use_audio").checked;
  const diarize = $("#diarize").checked;

  document.querySelectorAll("[data-needs]").forEach((node) => {
    const needs = node.dataset.needs;
    if (!needs) return;
    const ok = (needs === "video" && video) || (needs === "audio" && audio);
    if (node.classList.contains("chip")) {
      // A grid chip is disabled rather than hidden: the policy still exists,
      // and saying why it is unavailable is more useful than removing it.
      const input = node.querySelector("input");
      input.disabled = !ok || (needs === "audio" && !diarize && input.value === "speaker");
      node.classList.toggle("off", input.disabled);
      if (input.disabled && input.checked) {
        const fallback = document.querySelector("#chunking input:not(:disabled)");
        if (fallback) { fallback.checked = true; fallback.onchange(); }
      }
    } else {
      node.hidden = !ok;
    }
  });

  const note = $("#what-note");
  if (!video && !audio) {
    note.textContent = "nothing selected — pick at least one.";
    note.className = "hint error";
  } else {
    note.className = "hint";
    note.textContent = video && audio
      ? "Both, onto one shared chunk grid: chunk_id means the same thing in the manifest and the transcript."
      : video
      ? "The picture only. No transcript, and vad/speaker grids are unavailable — those boundaries live in the waveform."
      : "The soundtrack only. No manifest and no frames; the transcript is already text, so there is no describe stage to run.";
  }
  $("#go").disabled = !video && !audio;
}

$("#upload-form").onsubmit = async (e) => {
  e.preventDefault();
  const status = $("#upload-status");
  const file = $("#file").files[0];
  if (!file) return;

  const video = $("#use_video").checked;
  const audio = $("#use_audio").checked;
  const chosen = [...document.querySelectorAll("#samplers input:checked")]
    .map((i) => i.value);
  if (video && !chosen.length) {
    status.innerHTML = "";
    status.append(el("div", "error", "pick at least one sampler, or turn the "
                                     + "picture off and process the sound alone"));
    return;
  }

  const form = new FormData();
  form.append("file", file);
  form.append("use_video", video);
  form.append("use_audio", audio);
  form.append("samplers", chosen.join(","));
  form.append("chunking", document.querySelector("#chunking input:checked").value);
  form.append("chunk_duration", $("#chunk_duration").value);
  // Only when given. Blank means "each sampler keeps its own stride", which
  // is what the route's absent field means -- sending the box's value anyway
  // would impose one number on both of them.
  if ($("#every_frames").value.trim())
    form.append("every_frames", $("#every_frames").value.trim());
  form.append("vocabulary", $("#vocabulary").value);
  form.append("frame_store", video && $("#frame_store").checked);
  if (audio) {
    form.append("diarizer", $("#diarize").checked ? "pyannote" : "none");
    form.append("transcriber", $("#transcriber").value);
    if ($("#language").value.trim()) form.append("language", $("#language").value.trim());
  }
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
  for (const v of videos) {
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
    left.append(await exportLinks(v.video_id));

    const actions = el("div", "actions");
    // Describe reads a manifest, so it is offered only where one exists --
    // an audio-only video has nothing for a VLM to look at, and a transcript
    // is already the text that stage would produce.
    if (v.has.manifest) {
      const describe = el("button", null, "Describe");
      describe.onclick = () => queue("/describe", { video_id: v.video_id }, v.video_id);
      actions.append(describe);
    }
    const embed = el("button", null, "Embed");
    embed.onclick = () => queue("/embed", { video_id: v.video_id }, v.video_id);
    actions.append(embed);

    card.append(left, actions);
    host.append(card);
  }
}

/* Every document this video can hand to another service, as links.
 *
 * Read from `/exports` rather than assembled here, so a video that produced no
 * manifest offers no manifest link -- the alternative is a row of links, some
 * of which 404, which reads as breakage rather than as an audio-only run. */
async function exportLinks(videoId) {
  const row = el("div", "exports");
  let found;
  try {
    found = await api(`/videos/${videoId}/exports`);
  } catch {
    return row;
  }
  const all = el("a", "all", "all ↓");
  all.href = found.bundle + "?download=1";
  row.append(all);
  found.exports.forEach((x) => {
    const link = el("a", null, x.name);
    link.href = x.url + "?download=1";
    link.title = x.about + ` · ${(x.bytes / 1024).toFixed(1)} kB`;
    row.append(link);
  });
  return row;
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

/* ---------------------------------------------------------------- digest */
async function fillDigestVideos() {
  const select = $("#digest-video");
  const chosen = select.value;
  const { videos } = await api("/videos");
  select.innerHTML = "";
  videos.forEach((v) =>
    select.append(Object.assign(el("option", null, v.video_id), { value: v.video_id })));
  if (chosen) select.value = chosen;
  if (select.value) loadDigest(select.value);
  select.onchange = () => loadDigest(select.value);
}

$("#digest-form").onsubmit = async (e) => {
  e.preventDefault();
  const video = $("#digest-video").value;
  if (!video) return;
  const host = $("#digest-job");
  host.innerHTML = "";
  try {
    const body = await api("/aggregate", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        video_id: video,
        tier: $("#digest-tier").value,
        force: $("#digest-force").checked,
      }),
    });
    await follow(body.job.id, host, `aggregate ${video}`);
    loadDigest(video);
  } catch (err) {
    host.append(el("div", "error", err.message));
  }
};

async function loadDigest(video) {
  const out = $("#digest-out");
  out.innerHTML = "";
  const { aggregates } = await api(`/videos/${video}/aggregates`);
  const names = Object.keys(aggregates);
  if (!names.length) {
    out.append(el("p", "note", "nothing built yet for this video."));
    return;
  }
  const get = (n) => (aggregates[n] || {}).payload;

  const summary = get("summary");
  if (summary) {
    const card = section("Summary", `${summary.based_on.join(" + ")} · ` +
      `${summary.chunks} chunks · ${summary.reduction_levels} reduction levels`);
    card.append(el("p", null, summary.summary));
    if (summary.key_points?.length) {
      const ul = el("ul");
      summary.key_points.forEach((k) => ul.append(el("li", null, k)));
      card.append(ul);
    }
    if (summary.topics?.length) {
      const strip = el("div", "chips");
      summary.topics.forEach((t) => strip.append(el("span", "chip", t)));
      card.append(strip);
    }
    // The intermediate summaries, if the video was long enough to need any.
    // Folded away by default because the final summary is the answer -- but
    // kept reachable, because each leaf covers a real span and is the only
    // account of the video at that granularity.
    (summary.layers || []).forEach((layer) => {
      const fold = el("details", "layer");
      fold.append(el("summary", null,
        `level ${layer.level} · ${layer.kind} · ${layer.parts.length} parts`));
      layer.parts.forEach((part) => {
        const row = el("div", "layer-part");
        row.append(el("span", "ts",
          `${part.start_ts.toFixed(1)}–${part.end_ts.toFixed(1)}s  ` +
          `chunks ${part.chunk_ids[0]}–${part.chunk_ids[1]}`));
        row.append(el("p", null, part.text));
        fold.append(row);
      });
      card.append(fold);
    });
    out.append(card);
  }

  const chapters = get("chapters");
  if (chapters) {
    const card = section("Chapters",
      `${chapters.chapters.length} · covers the whole video: ${chapters.covers_whole_video}`);
    chapters.chapters.forEach((c) => {
      const row = el("div", "acct");
      row.append(el("div", "who", `${c.start_ts.toFixed(1)}–${c.end_ts.toFixed(1)}s   chunks ${c.first_chunk}–${c.last_chunk}`));
      row.append(el("strong", null, c.title), el("p", null, c.summary));
      card.append(row);
    });
    out.append(card);
  }

  const events = get("events");
  if (events) {
    const card = section("Events",
      `${events.count} · ` + Object.entries(events.categories)
        .map(([k, v]) => `${k} ${v}`).join(", "));
    events.events.forEach((ev) => {
      const row = el("div", "acct");
      row.append(el("div", "who", `${ev.start_ts.toFixed(1)}s   ${ev.category}`));
      row.append(el("p", null, ev.event + (ev.actor ? `  — ${ev.actor}` : "")));
      card.append(row);
    });
    out.append(card);
  }

  const stats = get("stats");
  if (stats) {
    const card = section("Statistics", "what embeddings cannot count");
    const grid = el("div", "stat-grid");
    const add = (label, value) => {
      const box = el("div", "stat");
      box.append(el("div", "stat-v", String(value)), el("div", "stat-k", label));
      grid.append(box);
    };
    add("duration", `${stats.duration_s.toFixed(0)}s`);
    add("chunks", stats.chunks);
    if (stats.people) {
      add("people, most", stats.people.max);
      add("people, mean", stats.people.mean);
      add("busiest", `${stats.people.busiest.start_ts.toFixed(0)}s`);
    }
    if (stats.speech) {
      add("speech", `${(stats.speech.speech_ratio * 100).toFixed(0)}%`);
      add("words", stats.speech.words);
      add("words/min", stats.speech.words_per_minute);
    }
    if (stats.objects) add("distinct objects", stats.objects.distinct);
    card.append(grid);
    out.append(card);
  }

  const speakers = get("speakers");
  if (speakers) {
    const card = section("Speakers",
      `${speakers.speaker_count} · ${speakers.handovers} handovers` +
      (speakers.monologue ? " · monologue" : ""));
    speakers.speakers.forEach((sp) => {
      const row = el("div", "bar-row");
      row.append(el("span", "bar-name", sp.speaker));
      const track = el("div", "bar");
      const fill = el("div", "bar-fill");
      fill.style.width = `${Math.max(sp.share * 100, 1)}%`;
      track.append(fill);
      row.append(track, el("span", "bar-val",
        `${(sp.share * 100).toFixed(0)}%  ${sp.seconds.toFixed(0)}s  ${sp.words}w`));
      card.append(row);
    });
    out.append(card);
  }

  const novelty = get("novelty");
  if (novelty) {
    const card = section("Least like the rest",
      `${novelty.outliers.length} beyond 2σ · ` +
      Object.entries(novelty.bases).map(([k, v]) => `${k} n=${v.points}`).join(", "));
    novelty.ranked.slice(0, 5).forEach((r) => {
      const row = el("div", "acct");
      row.append(el("div", "who",
        `${r.start_ts.toFixed(1)}–${r.end_ts.toFixed(1)}s   ${r.sampler}   novelty ${r.novelty.toFixed(3)}`));
      row.append(el("p", null, r.text));
      card.append(row);
    });
    out.append(card);
  }

  const ner = get("ner");
  if (ner) {
    const card = section("Entities", `${ner.count} across ${ner.based_on.join(" + ")}`);
    Object.entries(ner.by_label).forEach(([label, items]) => {
      const row = el("div", "acct");
      row.append(el("div", "who", label));
      const strip = el("div", "chips");
      items.forEach((t) => strip.append(el("span", "chip", t)));
      row.append(strip);
      card.append(row);
    });
    out.append(card);
  }

  const sentiment = get("sentiment");
  if (sentiment) {
    const o = sentiment.overall;
    const card = section("Language",
      `${o.turns} turns · dominant ${o.dominant} · ${sentiment.measures}`);
    const strip = el("div", "chips");
    Object.entries(o.shares).forEach(([k, v]) =>
      strip.append(el("span", "chip", `${k} ${(v * 100).toFixed(0)}%`)));
    card.append(strip);
    out.append(card);
  }
}

function section(title, note) {
  const card = el("div", "moment");
  const head = el("h3");
  head.append(el("span", null, title), el("span", "score", note || ""));
  card.append(head);
  return card;
}
