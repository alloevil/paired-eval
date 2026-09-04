# paired-eval

A methodology toolbox for **paired evaluation** of models, agents, and harnesses — statistically honest, and able to verify itself.

[中文](README.md) · [Findings](docs/findings.md) · [Lessons](docs/lessons.md) · [Corrections](docs/corrections.md)

Zero third-party dependencies, pure standard-library Python ≥ 3.9. Every model call is **injected**: the repository binds to no vendor.
Code comments, test messages, and the docs are written in Chinese; this README is the English entry point.

---

## What question it answers

> "Use programmatic verification when ground truth exists, rubrics otherwise" — the right direction, but landing it takes three things:
> **separate the objects, turn the verifier into a spectrum rather than a binary, and evaluate the rubric itself.**

This repository is the runnable form of those three things:

| What you evaluate | Held constant | What you get here |
|---|---|---|
**model** (capability ceiling) | harness fixed | interleaved repeats + pass@1 / pass^k + informative-sample diagnostics |
**agent** (strategy) | environment fixed | the above, plus trajectory faithfulness (per-claim grounding) |
**harness** (scaffold) | **model fixed**, paired A/B | McNemar + permutation test + Holm correction + effect-size CI |
**their interaction** | 2×2 factorial | main effects on one scale, ceilings flagged automatically |

Verifiers are chosen as "the strongest layer available", and layers stack — **programmatic checks gate, rubrics score**:

```
exact match → program/execution check → property check (round-trip / invariants) → retrieval → judge / rubric
                                                                                 ↑ must itself pass a canary (a bluffing answer must not score high)
```

## The thing it cares about most: saying "I can't tell" when it can't

A non-significant paired comparison can be three entirely different conditions with different remedies:

```python
>>> import claim_eval as ce
>>> c = ce.paired_compare(scores_a, scores_b)
>>> ce.interpret(c)["text"]
# 'powerless test: only 3 nonzero-difference pairs, minimum attainable p is 0.250 >= alpha=0.05 — need at least 6 ...'
# or: 'no difference detected: p=0.727, this design can only rule out a one-sided win rate >= 10% (n=80, Δ=+0.025)'
# or: 'uninformative null: n=0 — cannot be used as evidence of "no difference"'
```

(The actual strings are Chinese; the three verdicts are `significant` / bounded `null` / powerless-or-uninformative `null`.)

This is not decoration. In this repository's own experiments **the same data was misread three times** — "too few samples" written as "the effect does not exist"; "the test cannot reach significance" read as "not significant"; a "CI just excludes 0" formula used as a power formula. Each is recorded in [docs/corrections.md](docs/corrections.md), and each became a mechanism in `interpret()` / `p_floor()` / `required_pairs()`.

## Quick start

```sh
sh runtests.sh            # 183 tests, ~3 s, no external dependencies (this is what pre-commit runs)
sh checkall.sh --fast     # first four layers of the release checklist (~4 min); drop --fast to add the mutation ratchet (~25 min)
```

A paired A/B between two systems — all you provide is `call(prompt) -> str`:

```python
import paired_bench as pb

strict = pb.make_model(lambda p: my_llm("Follow the format exactly, no extra text. Task: " + p))
bare   = pb.make_model(lambda p: my_llm(p))

tasks = [t for t in pb.ALL_TASKS if t["id"] in pb.SCAFFOLD_SENSITIVE]   # the 4 scaffold-sensitive tasks
out = pb.run_interleaved({"strict": strict, "bare": bare}, tasks=tasks, n=6, prompt_prefix="")
print(pb.report(out["reports"], refusals=out["refusals"])["text"])
```

`report()` gives nine things in one call: effect size + CI, per-task permutation p, per-round McNemar, Holm correction,
discordant-pair distribution with concentration, informative sample, refusal attribution, the null's rule-out bound,
and ceiling/floor diagnostics. **Omit any one and the conclusion can be wrong** — every one of the nine was omitted from a hand-written report at some point.

Scoring whether an agent's answer is faithful to the observations it saw:

```python
import eval_task as et
r = et.evaluate({"id": "q1", "instruction": "Summarise the sources",
                 "verification": {"class": "trajectory", "grounding_policy": "must_ground"}},
                response=agent_answer, observations=tool_observations, llm=my_judge)
r["score"], r["metrics"]["fabricated"]    # grounding rate, and the number of unsupported claims
```

`llm(prompt, system, schema) -> dict` is a schema-constrained structured call; `examples/adapter_openai_compat.py`
is a standard-library-only adapter for OpenAI-compatible endpoints.

## Module map

| File | Responsibility |
|---|---|
`answer_match.py` | L1: programmatic grading of numbers / choices / sets / `\boxed{}`, with width and unit normalisation |
`claim_eval.py` | claim extraction → retrieval / trajectory verification → three-valued verdicts; **statistical primitives**; metering, bounded retry, throttled parallel map |
`rubric_eval.py` | per-criterion binary rubric judging + weighted aggregation + canary (separation from a bluffing answer) |
`eval_task.py` | declarative task schema routed by `exact / retrieval / trajectory / rubric`; interleaved paired judging |
`paired_bench.py` | task sets, interleaved repeated runs, reliability matrix, pairwise comparison, task screening, the full report |
`reproduce_findings.py` | turns recorded conclusions into executable checks (needs real model calls injected) |
`mutate_auto.py` / `mutate.sh` | mutation testing: AST-generated mutants + hand-picked semantic mutants + an equivalent-mutant baseline ratchet |
`runtests.sh` / `checkall.sh` / `check_hooks.py` | fast suite, five-layer release checklist, real-hook verification |

## Statistical primitives (`claim_eval`)

| Primitive | Question it answers |
|---|---|
`paired_compare(a, b)` | permutation p, bootstrap CI, wins/losses/ties, **effective pairs** |
`mcnemar_exact(a_only, b_only)` | binary paired: exact two-sided p on discordant pairs |
`holm_adjust(pvalues)` | family-wise error control |
`wilson_ci(k, n)` / `pass_hat_k(s, n, k)` | proportion interval / reliability (all k runs pass) |
`required_tasks(p_win, p_loss)` | how many paired tasks to detect this win-rate gap |
`required_pairs(mean_diff, sd)` | continuous-score version: runs the real permutation test inside (planner and test share one ruler) |
`detectable_effect(n)` | the inverse: minimum detectable effect for a fixed task set |
`p_floor(n)` / `min_units_for_alpha(a)` | the test's p floor (2/2ⁿ) and the minimum units it needs |
`interpret(compare)` | translates a result into a reportable conclusion: significant / bounded null / uninformative / powerless |

`paired_bench.saturation()` tells you how many tasks contributed zero information; `screen_tasks()` / `screen_graded()` find tasks with headroom in two stages — **near-ceiling tasks (>0.9) cannot be screened reliably at any affordable n**, so they are excluded by default.

## How it verifies itself

```
1 fast suite        ~3 s    183 tests + structural self-checks (defined == executed, exactly one __main__, doc anchors resolve, empty suite fails)
2 hook integration  ~26 s   temporary repo + bare remote, real pre-commit / pre-push
3 power calibration ~5 s    required_pairs against the textbook power formula (real Monte Carlo)
4 hand-picked mutants ~81 s 42 high-level semantic mutants + rot detection
5 mutation ratchet  ~25 min 473 mutation points, compared with the archived equivalent-mutant baseline; any new survivor fails
6 finding replay    manual  reproduce_findings.py: direction / magnitude / significance / informative sample of three recorded conclusions
```

Mutation testing is the core discipline: **passing tests are not the same as effective tests**. Thirty equivalent mutants are archived after analysis — writing tests for them would only pin implementation details.
Every layer caught at least one real defect while it was being built (including a diagnostic error found by the integration test on its first run, while 181 unit tests were green).

## About the numbers in docs/

The conclusions in [findings.md](docs/findings.md) (scaffold +0.625, model <10%, self-check increment +0.183, …) are measurements of **one particular model pair and a few particular task families**.
They show how the tools are used and how a conclusion should be written (with a bound, a concentration, and what it can rule out) — **they are not general results to inherit**.
When you evaluate your own systems, rerun the tools; do not copy the numbers.

[lessons.md](docs/lessons.md) is general: each entry was first paid for on real data and then turned into a mechanism in the code. The `docs/lessons.md#...` anchors in code comments point there.

## Design principles

- **Dependency injection, zero vendor binding**: `llm` / `search` / `call` / `judge` are supplied by the caller; tests exercise every path with deterministic fakes.
- **Planner and test share one ruler**: sample-size planning runs the very test that will be used, never a closed-form approximation.
- **A null must carry its bound**: never "no difference", only "rules out effects ≥ X%" or "powerless — needs N more ×××".
- **Gates must test themselves**: every verification layer is verified by the next.
- **Record equivalent mutants instead of testing them**; **leave corrections in place instead of overwriting them**.

## License

MIT.
