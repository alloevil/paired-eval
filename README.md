<p align="center">
  <a href="https://alloevil.github.io/paired-eval/"><img src="docs/assets/logo.svg" width="96" height="96" alt="paired-eval"></a>
</p>
<h1 align="center">paired-eval</h1>
<p align="center"><em>Evaluate models, agents and harnesses: program checks first, rubrics for the rest, honest paired statistics.</em></p>
<p align="center">
  <a href="https://pypi.org/project/paired-eval/"><img src="https://img.shields.io/pypi/v/paired-eval.svg" alt="PyPI"></a>
  <a href="https://github.com/alloevil/paired-eval/actions/workflows/ci.yml"><img src="https://github.com/alloevil/paired-eval/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-3.9%2B-blue.svg" alt="Python 3.9+"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/dependencies-none-brightgreen.svg" alt="Dependencies: none"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-yellow.svg" alt="License: MIT"></a>
</p>
<p align="center"><a href="https://alloevil.github.io/paired-eval/">Website</a> · <a href="README.zh-CN.md">中文</a> · <a href="docs/README.md">Docs</a> · <a href="CHANGELOG.md">Changelog</a></p>

**You ran model A and model B on the same 40 tasks. A scored 0.72, B scored 0.65. Is A better?**
Usually you cannot tell from those two numbers — and an LLM judge's "A is better" is not evidence either.
paired-eval answers with a paired test on the per-task results, a confidence interval, and a verdict that
says *significant* / *bounded null* (with the effect it rules out) / *uninformative* / *powerless* — never a bare "p > 0.05".

```sh
pip install paired-eval
```

```python
import paired_eval as pe

a = [1, 0, 1, 1, 0, 1, 1, 0]          # per-task pass/fail (or scores) for system A
b = [1, 0, 0, 1, 0, 0, 1, 0]          # same tasks, same order, for system B
print(pe.interpret(pe.paired_compare(a, b))["text"])
```

Pure standard-library Python ≥ 3.9, no dependencies. You inject both the model (`call(prompt) -> str`) and the judge (`judge(prompt, system, schema) -> dict`); no vendor binding.

Three things it does, and why each exists:

| | Because |
|---|---|
| **Paired statistics** — per-round McNemar + per-task permutation, Holm correction, bootstrap CI, sample-size planning | Unpaired means on 40 tasks hide a 0.3 effect behind noise; paired tests on the same tasks do not |
| **Program gate, then rubric** — an answer that fails a programmatic check scores 0 and the judge is never called | Judges get fooled on "does it work"; programs do not. Spend judge calls only on ranking answers that already work |
| **Saturation & ceiling diagnostics** — tasks both systems always pass carry no information; a system at 1.0 has no headroom | Most "no difference" results are really "no informative tasks", and the remedy is different |

## What it evaluates

The three objects ask three different questions and hold different things constant; mixing them up yields nothing.

| Evaluate | Hold constant | Vary | What one A/B looks like |
|---|---|---|---|
**model** | same tasks, same scaffold | the model | `{"A": call_a, "B": call_b}` |
**harness** | same model | prompt / scaffold / tool wiring | `{"strict": strict prefix, "bare": bare prompt}` |
**agent** | same model, same scaffold | the strategy | `{"single": one pass, "self-check": draft then self-correct}` |
**their interaction** | 2×2 factorial | both factors | main effects on one scale, ceilings flagged automatically |

## How it verifies: programmatic checks first, judges only for what they cannot cover

```
a unique ground truth ──→ exact (numeric / choice / set / \boxed{}) or a programmatic check(response) -> bool
only sources of fact  ──→ retrieval (verify against search) / trajectory (ground each claim in what the agent saw)
only a quality bar    ──→ rubric (per-criterion binary judging, weighted; the rubric must pass a canary: a bluffing answer must not score high)
```

Layers stack — **the program gates, the rubric scores**. Failing the gate scores 0 and never calls the judge: you do not pay a judge for an answer that is already wrong, and the judge cannot be fooled on "does it work" — it only ranks quality among candidates that do.

```python
import paired_eval as pe

task = {"id": "sum-explained", "instruction": "What is 12×12? Explain, and give the result starting with 'Answer: '",
        "verification": {"class": "gated",
                         "gate":  {"class": "exact", "gold": "144", "kind": "numeric", "marker": "Answer"},
                         "score": {"class": "rubric", "criteria": [{"text": "explains the calculation", "weight": 1}]}}}
r = pe.evaluate(task, response=answer, llm=judge)
print(r["score"], r["verdict"])        # wrong -> 0.0 'gated_out' (judge never called); right -> the rubric score
```

Other `evaluate` routes: `exact` (programmatic, no LLM), `retrieval`, `trajectory`, `rubric`; `rubric_canary` checks whether a rubric can be gamed by a bluffing answer.

## Ten seconds (an offline demo of the statistics layer)

```sh
python3 -m paired_eval --lang en        # or pe.set_language("en") once in code
```

**`strict` / `bare` here are two stub functions, not models** — they replay one real failure mechanism (under a bare prompt the correct JSON gets wrapped in markdown fences and fails to parse), only to show the shape of a report without any API. The output is real (a test keeps it identical to the current code):

```
informative sample: 2/4 tasks informative (always-pass 2, always-fail 0)
refusals: {'strict': 0, 'bare': 0}
at ceiling (1.000): strict — no headroom; effects measured against it as the reference are compressed
bare vs strict: Δ=-0.500 CI95=[-1.000,+0.000] | per-task p=0.506 per-round McNemar=0.00781 Holm=0.00781 | discordant 0:8 concentration=0.50 | significant: Δ=-0.500 CI95=[-1.000,+0.000] p=0.0078 (n=16)
```

Two tasks were solved by both systems every time and carry no information (informative sample 2/4); all 8 discordant pairs favour `strict`, spread over 2 tasks (concentration 0.50); the per-task permutation p is 0.506 because with 2 informative tasks **its minimum attainable p is 0.5**, while per-round McNemar uses all 16 paired units. The last clause is a verdict you can paste into a report.

## With real models: one A/B per object

**1. Mix verification classes by what can be verified.** Ground truth → programmatic; ground truth plus a quality bar → `gated`; only sources → `trajectory`:

```python
my_tasks = [
    {"id": "date", "instruction": "Write 5 March 2024 as an ISO 8601 date, starting with 'Answer: '",
     "verification": {"class": "exact", "gold": "2024-03-05", "marker": "Answer"}},
    {"id": "sum-explained", "instruction": "What is 12×12? Explain, and give the result starting with 'Answer: '",
     "verification": {"class": "gated",
                      "gate":  {"class": "exact", "gold": "144", "kind": "numeric", "marker": "Answer"},
                      "score": {"class": "rubric", "criteria": [{"text": "explains the calculation", "weight": 1}]}}},
    {"id": "summary", "instruction": "Write a one-sentence summary using only the source. Source: X's 2023 revenue was 41.2bn.",
     "observations": [{"tool_call_id": "t1", "tool": "doc", "observation": "X's 2023 revenue was 41.2bn."}],
     "verification": {"class": "trajectory", "grounding_policy": "must_ground"}},
]
```

**2. Plug in models and a judge, and turn the task set into what the paired pipeline eats.** The adapter is [examples/adapter_openai_compat.py](examples/adapter_openai_compat.py) (standard library only, any compatible endpoint); `bench_tasks` binarises each task's `evaluate` score at an explicit threshold — "pass" means ≥ this much, and that belongs in the report.

```python
from examples.adapter_openai_compat import make_call, make_llm   # standard-library-only OpenAI-compatible adapter

call_a, call_b = make_call(model="model-a"), make_call(model="model-b")
judge = make_llm(model="judge-model")
tasks = pe.bench_tasks(my_tasks, threshold=1.0, llm=judge)
```

**3. Three objects, each holding its own variable constant.** `n` is repeats per task per system; interleaving rotates the order.

```python
STRICT = "Follow the format exactly, no extra text. Task: "

def self_check(call):                     # agent strategy: draft, then self-correct against the instruction (one extra call, no tools)
    return lambda p: call(f"Instruction: {p}\nDraft: {call(p)}\nCheck the draft against the instruction; output only the corrected answer.")

runs = {
    "model":   pe.run_interleaved({"A": pe.make_model(call_a), "B": pe.make_model(call_b)},
                                  tasks=tasks, n=6, prompt_prefix=STRICT),
    "harness": pe.run_interleaved({"strict": pe.make_model(lambda p: call_a(STRICT + p)),
                                   "bare": pe.make_model(call_a)}, tasks=tasks, n=6, prompt_prefix=""),
    "agent":   pe.run_interleaved({"single": pe.make_model(call_a),
                                   "self-check": pe.make_model(self_check(call_a))}, tasks=tasks, n=6, prompt_prefix=""),
}
for axis, run in runs.items():
    print(axis, pe.report(run["reports"], refusals=run["refusals"], lang="en")["text"], sep="\n")
```

Plan the sample size before running, and let `interpret` state the conclusion afterwards:

```python
pe.required_tasks(0.30, 0.10)         # A wins 30% / loses 10% of tasks: paired tasks needed for 80% power
pe.required_pairs(0.15, 0.30)         # continuous scores, mean diff 0.15, sd 0.30 -> runs the real permutation test (~20 s)
pe.detectable_effect(31)              # the inverse: the smallest one-sided win rate 31 tasks can detect
pe.interpret(pe.paired_compare(scores_a, scores_b), lang="en")["text"]   # afterwards: what this result can and cannot say
```

`screen_tasks` / `screen_graded` screen for tasks that discriminate, in two stages; near-ceiling tasks (> 0.9) are excluded by default — they cannot be screened reliably at any affordable number of runs.

## Reading the report

| Field | Meaning |
|---|---|
informative sample | Tasks both systems always pass or always fail carry no information; read the MDE against the informative count |
refusals | Dropped calls per system — the refusal rate is part of the result |
at ceiling / at floor | A system at 1.0 or 0.0 has no headroom; effects against it are compressed, interactions uninterpretable |
Δ, CI95 | Effect size with a bootstrap 95% interval |
per-task p / per-round McNemar / Holm | The two paired tests; Holm correction when several systems are compared |
discordant a:b, concentration | Direction and spread of the disagreements; 1.0 = all from one task |
verdict | significant · bounded null (with the ruled-out effect) · uninformative · powerless (with what is missing) |

"p > 0.05" means three different things with different remedies: too few units is *uninformative*; too few discordant pairs is *powerless* (add rounds, or tasks that separate the systems); enough units and still nothing is a *bounded null* — report the effect it rules out, never "no difference".

## How it relates to other tools

They are frameworks for *running* evaluations; paired-eval sits downstream and does not duplicate task libraries or model backends. Descriptions are taken from each project's own README.

| Tool | What it does | Relation |
|---|---|---|
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | 60+ academic benchmarks over many model backends; per-metric standard errors | Produces per-task scores → feed them to `paired_compare` / `interpret` |
[Inspect](https://github.com/UKGovernmentBEIS/inspect_ai) | Eval framework: prompt engineering, tool use, multi-turn dialog, model-graded components; 200+ pre-built evals | Same; scorer output is per-sample and pairs naturally |
[promptfoo](https://github.com/promptfoo/promptfoo) | Side-by-side model/prompt comparison with assertions; red teaming | Overlaps on "compare"; paired-eval adds stacked verifiers and the statistical verdict layer |
[openai/evals](https://github.com/openai/evals) | Registry of template-based evals fed by JSON data | Same downstream relation |

## API overview

`import paired_eval as pe`:

| Group | Entry points |
|---|---|
Verifiers | `evaluate` (`exact` / `retrieval` / `trajectory` / `rubric` / `gated`) `evaluate_pair` `evaluate_batch` `validate_task` `grade_answer` `extract_claims` `verify_trajectory` `run_rubric` `rubric_canary` |
Paired A/B | `make_model` `bench_tasks` `judge_check` `run_interleaved` `run_paired` `run_repeated` `report` `pairwise_compare` `reliability_matrix` `saturation` |
Statistics | `paired_compare` `mcnemar_exact` `holm_adjust` `wilson_ci` `pass_hat_k` `required_tasks` `required_pairs` `detectable_effect` `p_floor` `min_units_for_alpha` `interpret` |
Screening | `screen_tasks` `screen_graded` · built-in `ALL_TASKS` (31 Chinese smoke tasks, for examples and self-tests) |
Adapters | `make_resilient` `throttled_pmap` `Meter` `set_language` |

## Scope and status

- **Is**: a method and toolkit for evaluating models / agents / harnesses — stackable verifiers (program gate + rubric score), honest statistics for paired comparison, sample-size planning, task screening.
- **Is not**: a large task library (the 31 built-in tasks are examples and self-tests), an evaluation platform, a model client, or a rubric **generator** (the autorubric half is not here — this evaluates whether a rubric can be fooled). Run large suites with the frameworks above and hand their per-task scores to this.
- **Status**: 0.4.0, single author, API may change. Reports in English and Chinese (`set_language`); code comments and built-in tasks are Chinese.
- [docs/findings.md](docs/findings.md) is a case study done with this toolbox on one model pair and three task families: harness and agent effects of about +0.6–0.75, model effect < 10%, the two with diminishing but stackable returns. The numbers are instance-specific and show how a conclusion should be written.

## Docs · Contributing · License

[Docs index](docs/README.md) · [Methodology lessons](docs/lessons.md) (what each primitive guards against) · [中文 README](README.zh-CN.md) · [CONTRIBUTING.md](CONTRIBUTING.md) · MIT
