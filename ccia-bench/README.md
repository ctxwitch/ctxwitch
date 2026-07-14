# CCIA-Bench v0

Public benchmark for **Context Change Impact Analysis (CCIA)** — labeled before/after AI context pairs with expert severity labels.

## Quick start

```bash
# Regenerate 58 synthetic pairs (optional)
python ccia-bench/generate_pairs.py

# Run CBIA and print accuracy report
python ccia-bench/run_benchmark.py

# Show only mismatches
python ccia-bench/run_benchmark.py --failures

# Allow ±1 severity level (softer scoring)
python ccia-bench/run_benchmark.py --tolerance 1
```

## Pair format (`pairs.jsonl`)

Each line is one JSON object:

```json
{
  "id": "011",
  "description": "Tool removed: escalate",
  "source": "synthetic",
  "old": { "version": "v0.1.0", "components": { ... } },
  "new": { "version": "v0.1.0", "components": { ... } },
  "labels": {
    "compound": "breaking",
    "dimensions": {
      "tools_capability": "breaking"
    }
  }
}
```

- **`old` / `new`**: Partial `witch.yaml` context dicts (same shape CBIA expects)
- **`labels.compound`**: Expected overall severity (`no_change` … `breaking`)
- **`labels.dimensions`**: Expected severity per dimension (only non-`no_change` dims listed)

## v0 composition (58 pairs)

| Category | Count |
|----------|------:|
| No change / cosmetic | 2 |
| Temperature | 5 |
| Model | 3 |
| Tools | 4 |
| RAG | 5 |
| Guardrails / memory / tokens | 10 |
| Prompt-only | 15 |
| Compound | 6 |
| Numeric threshold shifts | 2 |
| Environment override changes | 3 |
| Orthography robustness (spelling-parity) | 3 |

Labels are **expert-intended ground truth**, not CBIA output. Mismatches in the report show where CBIA needs improvement.

## Roadmap

- **v1 (200+ pairs)**: Real diffs from open repos + 3–5 human annotators on ambiguous prompt pairs
- **v2**: Public leaderboard + eval-regression correlation subset
