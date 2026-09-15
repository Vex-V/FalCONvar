# Cross-Frame Person Re-Identification via Structured Attribute Matching

## Problem

Given per-frame person descriptions from a VLM (e.g. _"old man, light grey hair; black pants, white shirt"_), link appearances of the same person across non-consecutive frames to build a narrative track — "Person A appears in frame 1 sitting, frame 3 walking, frame 7 leaving."

Embedding-based cosine similarity fails here because it measures **semantic meaning**, not **identity**. Two different people can have near-identical descriptions, producing high similarity despite being distinct individuals.

## Solution Overview

Replace semantic similarity with **structured attribute matching** in three stages:

1. **Structured extraction** — Parse free-text descriptions into a fixed attribute schema (gender, age group, hair color, clothing colors/types, accessories, distinguishing marks). Constrain the VLM output to a small canonical vocabulary to prevent drift ("ivory" vs "cream" vs "off-white").

2. **Weighted attribute distance** — Compute a custom distance function over the attribute struct. Weight fields by discriminative power: hard constraints (gender, age group) carry high penalty on mismatch; color fields use alias normalization before comparison; rare attributes like accessories or scars provide strong matching signal when shared.

3. **Optimal frame-to-frame assignment** — For each pair of nearby frames, build a cost matrix from pairwise attribute distances and solve with the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`). Assignments below a threshold are accepted as identity links; unmatched detections spawn new tracks. Stale tracks are pruned after a configurable frame gap.

## Attribute Schema

| Field           | Type       | Weight | Notes                                      |
|-----------------|------------|--------|---------------------------------------------|
| `gender`        | enum       | 3.0    | Hard constraint — mismatch is very costly   |
| `age_group`     | enum       | 2.0    | Hard constraint                             |
| `hair_color`    | string     | 2.5    | Normalized via alias map                    |
| `hair_style`    | string     | 1.0    | Soft match                                  |
| `top_color`     | string     | 2.0    | Normalized via alias map                    |
| `top_type`      | string     | 1.0    | Soft match                                  |
| `bottom_color`  | string     | 2.0    | Normalized via alias map                    |
| `bottom_type`   | string     | 1.0    | Soft match                                  |
| `accessories`   | list[str]  | bonus  | Shared items give a -2.0 reward             |
| `distinguishing`| string     | bonus  | Shared marks give a -2.0 reward             |

## Key Design Decisions

- **Canonical color vocabulary**: Map color synonyms to a small palette before comparison ("light grey" → "grey", "navy" → "dark blue") to prevent spurious mismatches from VLM phrasing variation.
- **Asymmetric weighting**: Immutable traits (gender, age, hair) are weighted higher than clothing, which could theoretically change — but in short videos, clothing is the strongest signal.
- **Hungarian assignment**: Guarantees globally optimal one-to-one matching per frame pair, avoiding greedy errors where an early match steals the correct assignment from a later comparison.
- **Gap tolerance**: Tracks survive up to N frames without a match, handling temporary occlusions or frames where the VLM misses a detection.

## Tuning

- **Threshold**: The accept/reject cutoff for assignments. Lower = stricter matching, more fragmented tracks. Higher = more permissive, risk of merging distinct people. Tune on labeled examples from actual footage.
- **Weights**: Adjust based on scene diversity. In a crowd wearing uniforms, clothing color is useless — lean on hair and accessories. In casual settings, clothing color alone often suffices.
- **Color aliases**: Extend the alias map based on observed VLM output. Alternatively, constrain the VLM prompt to output from a fixed color enum.