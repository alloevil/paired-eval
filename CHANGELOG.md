# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-09-04

First public cut. Everything below was built and verified in one continuous session; the reasoning behind each piece
is in [docs/lessons.md](docs/lessons.md), the measured results in [docs/findings.md](docs/findings.md), and the
conclusions that were later overturned in [docs/corrections.md](docs/corrections.md).

### Added
- **Verifier spectrum**: `answer_match` (exact / numeric / choice / set / `\boxed{}`), `claim_eval` (claim extraction,
  retrieval and trajectory grounding with three-valued verdicts), `rubric_eval` (per-criterion binary rubric judging
  with a canary against bluffing answers), `eval_task` (declarative task schema routed by verification class).
- **Paired benchmarking** (`paired_bench`): interleaved repeated runs, reliability matrix, pairwise comparison with
  Holm correction, saturation diagnostics, two-stage task screening (`screen_tasks`, `screen_graded`), and a single
  `report()` that emits nine diagnostics at once — effect size + CI, per-task permutation p, per-round McNemar, Holm,
  discordant-pair concentration, informative sample, refusals, the null's rule-out bound, ceiling/floor flags.
- **Statistical primitives** (`claim_eval`): `paired_compare` (with effective pairs), `mcnemar_exact`, `holm_adjust`,
  `wilson_ci`, `pass_hat_k`, `required_tasks`, `required_pairs`, `detectable_effect`, `p_floor`,
  `min_units_for_alpha`, `interpret` — the last translates a comparison into one of four honest verdicts
  (significant / bounded null / uninformative / powerless), naming the limiting basis and the remedy.
- **Finding reproduction** (`reproduce_findings.py`): three recorded conclusions as executable checks with
  drift-vs-power separation.
- **Self-verification**: 192 tests (~3 s), real-hook integration check, power calibration, 42 hand-picked semantic
  mutants, AST mutation ratchet over 473 points with a 30-entry archived equivalent-mutant baseline, crash-safe
  (SIGKILL) source backups, structural self-checks (defined == executed tests, doc anchors resolve, empty suite fails).
- **Project shell**: bilingual README whose examples are executed by tests, MIT license, `pyproject.toml` (zero
  dependencies, Python ≥ 3.9), CI matrix 3.9 / 3.12 / 3.13 plus a nightly full checklist, a standard-library-only
  OpenAI-compatible adapter example tested against a local HTTP server.

### Fixed (before release, recorded for the method)
- `interpret()` computed the p floor from the unit count; McNemar's attainable minimum p depends on **discordant
  pairs**, and the permutation test's on **nonzero-difference pairs**. Found by the integration test on its first run.
- A boundary test used `0.88 − 0.80` as "exactly the threshold"; it is `0.0799…` and passed on Python 3.10 only by
  rounding luck. Exposed by CI on 3.12 (Neumaier `sum()`); fixed with binary-exact values and a multi-interpreter layer.

[Unreleased]: https://github.com/alloevil/paired-eval/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/alloevil/paired-eval/releases/tag/v0.1.0
