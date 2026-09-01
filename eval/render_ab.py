"""Measure candidate renderings of the structured half against each other.

Only `render` varies. The summaries, the queries, the embedder and the ranking
are identical across arms, so any difference is the rendering.

  values-insertion   what ships today: values joined, entity key order as read.
                     Source-dependent -- jsonb reorders object keys -- so this
                     arm uses the file's order, which is its best case.
  values-sorted      the minimal fix: same text, entity keys sorted, so the
                     rendering is a function of content rather than transport.
  labelled-sorted    keys sorted AND named, so a field boundary is visible.
                     `", "` currently separates fields and also occurs inside
                     them, which makes the boundaries invisible to the model.

Ranking follows the shipped path: cosine over units, then `to_moments` folds
them into chunks by RRF. Ground truth is the chunk the query was written from.
Five chunks, so random scores 0.200 top-1 and 0.457 MRR.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from ver2.embed.index.base import Hit
from ver2.embed.units import Unit
from ver2.retrieve.search import to_moments


def render_values_insertion(structured):
    lines = []
    for key, value in sorted((structured or {}).items()):
        if isinstance(value, str) and value.strip():
            lines.append(f"{key}: {value}")
        elif isinstance(value, list) and value:
            items = []
            for item in value:
                if isinstance(item, dict):
                    items.append(", ".join(str(v) for v in item.values() if v))
                else:
                    items.append(str(item))
            lines.append(f"{key}: " + "; ".join(items))
    return ". ".join(lines)


def render_values_sorted(structured):
    lines = []
    for key, value in sorted((structured or {}).items()):
        if isinstance(value, str) and value.strip():
            lines.append(f"{key}: {value}")
        elif isinstance(value, list) and value:
            items = []
            for item in value:
                if isinstance(item, dict):
                    items.append(", ".join(str(item[k]) for k in sorted(item) if item[k]))
                else:
                    items.append(str(item))
            lines.append(f"{key}: " + "; ".join(items))
    return ". ".join(lines)


def render_labelled_sorted(structured):
    lines = []
    for key, value in sorted((structured or {}).items()):
        if isinstance(value, str) and value.strip():
            lines.append(f"{key}: {value}")
        elif isinstance(value, list) and value:
            items = []
            for item in value:
                if isinstance(item, dict):
                    items.append("; ".join(f"{k} {item[k]}" for k in sorted(item) if item[k]))
                else:
                    items.append(str(item))
            lines.append(f"{key}: " + " | ".join(items))
    return ". ".join(lines)


ARMS = {
    "values-insertion": render_values_insertion,
    "values-sorted": render_values_sorted,
    "labelled-sorted": render_labelled_sorted,
}


def cosine(a, b):
    return (sum(x * y for x, y in zip(a, b))
            / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))))


def units_of(document):
    out = []
    for chunk in document["chunks"]:
        for sampler, block in chunk["samplers"].items():
            if block.get("description"):
                out.append(Unit(document["video_id"], chunk["chunk_id"], sampler,
                                block["description"], block.get("structured") or {},
                                start_ts=chunk.get("start_ts"), end_ts=chunk.get("end_ts")))
    return out


def rank_of(chunk_id, query_vec, units, vectors):
    hits = sorted(
        (Hit(u.video_id, u.chunk_id, u.sampler, cosine(query_vec, v), u.text,
             u.start_ts, u.end_ts, [])
         for u, v in zip(units, vectors)),
        key=lambda h: h.score, reverse=True)
    for position, moment in enumerate(to_moments(hits), start=1):
        if moment.chunk_id == chunk_id:
            return position
    return None


def main():
    from ver2 import db
    from ver2.embed import embedders as embedders_mod

    db.load_env()
    doc = json.loads(Path("out/test1/descriptions.json").read_text(encoding="utf-8"))
    pairs = json.loads(Path("eval/query_pairs.json").read_text(encoding="utf-8"))
    units = units_of(doc)
    emb = embedders_mod.build("openai")

    # Queries do not depend on the rendering, so they are embedded once.
    literals = [p["literal"] for p in pairs]
    paras = [p["paraphrase"] for p in pairs]
    qv = dict(zip(literals + paras, emb.embed_documents(literals + paras)))

    print(f"{len(units)} units, {len(pairs)} query pairs, "
          f"{len({u.chunk_id for u in units})} chunks "
          f"(random: top-1 0.200, MRR 0.457)\n")
    header = f"{'arm':<20}{'lit top-1':<11}{'lit MRR':<10}{'par top-1':<11}{'par MRR':<10}{'mean pairwise':<14}chars"
    print(header); print("-" * len(header))

    results = {}
    for name, render in ARMS.items():
        texts = []
        for u in units:
            body = render(u.structured)
            texts.append("\n\n".join([u.text, body]) if body else u.text)
        vecs = emb.embed_documents(texts)
        sims = [cosine(vecs[i], vecs[j])
                for i in range(len(vecs)) for j in range(i + 1, len(vecs))]
        row = {}
        for label, field in (("lit", "literal"), ("par", "paraphrase")):
            ranks = [rank_of(p["chunk_id"], qv[p[field]], units, vecs) for p in pairs]
            row[f"{label}_ranks"] = ranks          # kept for the paired test
            row[f"{label}_top1"] = sum(r == 1 for r in ranks) / len(ranks)
            row[f"{label}_mrr"] = sum(1 / r for r in ranks) / len(ranks)
        row["pairwise"] = sum(sims) / len(sims)
        row["chars"] = sum(len(t) for t in texts) // len(texts)
        results[name] = row
        print(f"{name:<20}{row['lit_top1']:<11.3f}{row['lit_mrr']:<10.3f}"
              f"{row['par_top1']:<11.3f}{row['par_mrr']:<10.3f}"
              f"{row['pairwise']:<14.3f}{row['chars']}")
    Path("eval/render_ab.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n-> eval/render_ab.json")


if __name__ == "__main__":
    main()
