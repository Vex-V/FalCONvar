"""Generate literal/paraphrase query pairs from a description document.

The pairs are the instrument, and a blunt instrument measures nothing. Asked
merely to "share no content words", a model keeps a median 50% of them and the
lexical half then appears to win on paraphrases -- which is the tool reporting
its own leakage as a result. So each query's own content words are forbidden
**by name** in the prompt, and disjointness is re-checked here afterwards:
a pair that still overlaps is retried once, and dropped if it overlaps again.

Queries are derived from the *structured* fields rather than the summaries,
because that is the half whose rendering is under test. This favours the
structured fields, exactly as the earlier measurement's caveat says -- it is a
comparison between two renderings of the same fields, not a claim about
retrieval in general.

Ground truth is the chunk the content came from. Five chunks, so a random
ranking scores 0.200 top-1 and 0.457 MRR.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

STOP = set("""a an the and or but of in on at to for with from by as is are was were be been
being this that these those it its his her their our your my he she they we you i not no
than then there here what which who whom whose when where why how all any both each few more
most other some such only own same so too very can will just don should now over under near
into out up down about again further once during before after above below between through""".split())


def content_words(text: str) -> set[str]:
    """Tokens that carry meaning, lightly stemmed so plurals do not hide overlap."""
    out = set()
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if len(w) < 3 or w in STOP:
            continue
        for suffix in ("ing", "ed", "es", "s"):
            if len(w) > len(suffix) + 2 and w.endswith(suffix):
                w = w[: -len(suffix)]
                break
        out.add(w)
    return out


def overlap(literal: str, paraphrase: str) -> set[str]:
    return content_words(literal) & content_words(paraphrase)


SYSTEM = (
    "You write short search queries for a video retrieval index, the kind a "
    "person types into a search box. Never explain, never number, return only "
    "the JSON asked for."
)

SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["pairs"],
    "properties": {"pairs": {
        "type": "array",
        "description": "One entry per query pair.",
        "items": {
            "type": "object", "additionalProperties": False,
            "required": ["literal", "paraphrase"],
            "properties": {
                "literal": {"type": "string", "description":
                            "A 3-8 word query using wording from the content."},
                "paraphrase": {"type": "string", "description":
                               "The same intent, 3-8 words, sharing NO content "
                               "word with the literal form."},
            }}}},
}


def prompt_for(content: str, count: int, forbidden: list[str]) -> str:
    return (
        f"Here is what one segment of video contains:\n\n{content}\n\n"
        f"Write {count} query pairs about DISTINCT things in it. Each pair is:\n"
        f"  literal    -- a query someone would type, reusing the wording above\n"
        f"  paraphrase -- the SAME intent said with entirely different words\n\n"
        "The paraphrase must not reuse ANY content word from its own literal "
        "form, including plurals and other inflections. Use synonyms, "
        "hypernyms or a description instead. These words in particular must "
        "not appear in any paraphrase:\n  "
        + ", ".join(sorted(forbidden)[:60])
    )


def generate(document: dict, per_unit: int = 3, model: str = "gpt-5.4-mini") -> list[dict]:
    from openai import OpenAI

    from ver2.embed.units import render
    from ver2.describe.vlm.openai_client import _api_key

    client = OpenAI(api_key=_api_key())
    pairs: list[dict] = []
    for chunk in document["chunks"]:
        for sampler, block in chunk["samplers"].items():
            content = render(block.get("structured") or {})
            if not content:
                continue
            forbidden = sorted(content_words(content))
            got = _ask(client, model, content, per_unit, forbidden)
            for pair in got:
                bad = overlap(pair["literal"], pair["paraphrase"])
                if bad:                       # one retry, told exactly what leaked
                    retry = _ask(client, model, content, 1,
                                 forbidden + sorted(bad))
                    if retry and not overlap(retry[0]["literal"], retry[0]["paraphrase"]):
                        pair = retry[0]
                    else:
                        print(f"  dropped (overlap {sorted(bad)}): {pair['literal']!r}",
                              file=sys.stderr)
                        continue
                pairs.append({"chunk_id": chunk["chunk_id"], "sampler": sampler,
                              "literal": pair["literal"].strip(),
                              "paraphrase": pair["paraphrase"].strip()})
    return pairs


def _ask(client, model: str, content: str, count: int, forbidden: list[str]) -> list[dict]:
    response = client.responses.create(
        model=model, instructions=SYSTEM,
        input=[{"role": "user", "content": prompt_for(content, count, forbidden)}],
        max_output_tokens=2000,
        text={"format": {"type": "json_schema", "name": "query_pairs",
                         "strict": True, "schema": SCHEMA}})
    return json.loads(response.output_text)["pairs"]


if __name__ == "__main__":
    from ver2 import db
    db.load_env()
    doc = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "eval/query_pairs.json")
    pairs = generate(doc)
    disjoint = sum(not overlap(p["literal"], p["paraphrase"]) for p in pairs)
    out.write_text(json.dumps(pairs, indent=2), encoding="utf-8")
    print(f"{len(pairs)} pairs, {disjoint} verified disjoint -> {out}")
