# Contributing to ctxwitch

Thanks for wanting to improve ctxwitch! Contributions of all kinds are
welcome: bug reports, new benchmark pairs, detector improvements, docs.

## Quick start

```bash
git clone https://github.com/ctxwitch/ctxwitch.git
cd ctxwitch
pip install -e ".[dev]"
make test          # full suite must pass
make bench         # CBIA accuracy on ccia-bench (no regressions, please)
```

Try `witch tour` for a guided walk through the workflow you're improving.

## Contributor License Agreement

Before your first pull request can merge, you'll be asked to sign our
[CLA](CLA.md) — a bot comments on the PR and you reply with one line:

> I have read the CLA Document and I hereby sign the CLA

One time only; it covers all future contributions. In short, you keep
ownership of your code and grant the project the rights needed to keep
licensing it flexibly. If you're contributing on work time, check that
your employer permits it (CLA §4c).

## Pull request guidelines

- **Tests required.** Detector or CLI changes need tests in `tests/`;
  behavior changes to CBIA also need a benchmark check (`make bench`) —
  include before/after numbers in the PR description if they move.
- **New detectors** follow the existing pattern: deterministic, explainable
  (every impact carries a human-readable `reason`), and covered by both
  unit tests and at least one `ccia-bench` pair.
- **No new runtime dependencies** without discussion — Tiers 1–5 are
  deliberately zero-dependency beyond PyYAML/click/rich.
- Keep PRs focused; one change per PR merges fastest.

## Adding benchmark pairs

The most valuable low-effort contribution! Add a pair to
`ccia-bench/generate_pairs.py` with expert labels, regenerate
(`make bench-generate`), and explain in the PR why the expected severity
is what it is. Real-world-inspired pairs (anonymized) are especially
welcome.

## Reporting issues

Include your `witch --version`, the two context versions (redacted as
needed), and what CBIA reported vs. what you expected. Misclassifications
are gold — they become benchmark pairs.
