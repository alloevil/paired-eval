#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 docs/index.html —— 项目主页(GitHub Pages 从 docs/ 提供)。

为什么用生成器而不是手写 HTML: 页面里贴的 demo 输出、版本号、安装命令都是事实, 手写会漂移。
生成器从代码取真值; tests/test_site.py 断言"仓库里的 index.html == 生成器此刻的输出", 谁手改都会红。
页面零外部依赖(无外链脚本/样式/字体), 与项目"零依赖"一致; 中英切换是 15 行内联 JS。

用法: python3 build_site.py        # 重写 docs/index.html
      python3 build_site.py --check  # 只比对, 不写; 不一致时退出码 1
"""
import html
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import paired_eval as pe  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "docs" / "index.html"
REPO = "https://github.com/alloevil/paired-eval"
ABOUT = "Evaluate models, agents and harnesses: program checks first, rubrics for the rest, honest paired statistics."   # 与 GitHub About 同句; 主页 meta description 与 CITATION abstract 都从这里取
INSTALL = f"pip install git+{REPO}.git"


def demo_output(lang):
    lines = []
    pe.demo(out=lambda s: lines.extend(s.splitlines()), lang=lang)
    return "\n".join(lines)


TASKS_CODE_ZH = '''import paired_eval as pe
from examples.adapter_openai_compat import make_call, make_llm     # 只用标准库的 OpenAI 兼容适配器

my_tasks = [   # 按"有什么可验"混搭: 有真值 -> exact; 有真值又要看质量 -> gated; 只有资料 -> trajectory
    {"id": "date", "instruction": "把 2024年3月5日 写成 ISO 8601 日期; 以「答案: 」开头给出",
     "verification": {"class": "exact", "gold": "2024-03-05"}},
    {"id": "sum-explained", "instruction": "12×12 等于多少, 并说明过程; 以「答案: 」开头给出结果",
     "verification": {"class": "gated",
                      "gate":  {"class": "exact", "gold": "144", "kind": "numeric"},
                      "score": {"class": "rubric", "criteria": [{"text": "说明了计算过程", "weight": 1}]}}},
]
call_a, call_b, judge = make_call(model="model-a"), make_call(model="model-b"), make_llm(model="judge-model")
tasks = pe.bench_tasks(my_tasks, threshold=1.0, llm=judge)        # judge 评分的题也进同一套配对流水线

STRICT = "严格按要求输出, 不要任何多余内容。要求: "
runs = {
    "model":   pe.run_interleaved({"A": pe.make_model(call_a), "B": pe.make_model(call_b)}, tasks=tasks, n=6, prompt_prefix=STRICT),
    "harness": pe.run_interleaved({"strict": pe.make_model(lambda p: call_a(STRICT + p)), "bare": pe.make_model(call_a)}, tasks=tasks, n=6, prompt_prefix=""),
}
for axis, run in runs.items():
    print(axis, pe.report(run["reports"], refusals=run["refusals"])["text"], sep="\\n")'''

TASKS_CODE_EN = (TASKS_CODE_ZH
    .replace("# 只用标准库的 OpenAI 兼容适配器", "# standard-library-only OpenAI-compatible adapter")
    .replace("# 按\"有什么可验\"混搭: 有真值 -> exact; 有真值又要看质量 -> gated; 只有资料 -> trajectory",
             "# mix by what can be verified: ground truth -> exact; truth + quality bar -> gated; only sources -> trajectory")
    .replace("把 2024年3月5日 写成 ISO 8601 日期; 以「答案: 」开头给出", "Write 5 March 2024 as an ISO 8601 date, starting with \'Answer: \'")
    .replace('"gold": "2024-03-05"}', '"gold": "2024-03-05", "marker": "Answer"}')
    .replace("12×12 等于多少, 并说明过程; 以「答案: 」开头给出结果", "What is 12×12? Explain, and give the result starting with \'Answer: \'")
    .replace('"kind": "numeric"}', '"kind": "numeric", "marker": "Answer"}')
    .replace("说明了计算过程", "explains the calculation")
    .replace("# judge 评分的题也进同一套配对流水线", "# judge-scored tasks flow through the same paired pipeline")
    .replace("严格按要求输出, 不要任何多余内容。要求: ", "Follow the format exactly, no extra text. Task: ")
    .replace('refusals=run["refusals"])', 'refusals=run["refusals"], lang="en")'))

GATED_CODE_ZH = '''task = {"id": "sum-explained", "instruction": "12×12 等于多少, 并说明过程; 以「答案: 」开头给出结果",
        "verification": {"class": "gated",
                         "gate":  {"class": "exact", "gold": "144", "kind": "numeric"},
                         "score": {"class": "rubric", "criteria": [{"text": "说明了计算过程", "weight": 1}]}}}
r = pe.evaluate(task, response=answer, llm=judge)
print(r["score"], r["verdict"])        # 答错 -> 0.0 'gated_out'(judge 一次都没调用); 答对 -> rubric 分'''

GATED_CODE_EN = (GATED_CODE_ZH
    .replace("12×12 等于多少, 并说明过程; 以「答案: 」开头给出结果", "What is 12×12? Explain, and give the result starting with \'Answer: \'")
    .replace('"kind": "numeric"}', '"kind": "numeric", "marker": "Answer"}')
    .replace("说明了计算过程", "explains the calculation")
    .replace("# 答错 -> 0.0 'gated_out'(judge 一次都没调用); 答对 -> rubric 分", "# wrong -> 0.0 'gated_out' (judge never called); right -> the rubric score"))

FEATURES = [
    ("三个对象各固定各的变量", "评 model 固定脚手架、评 harness 固定模型、评 agent 固定两者只换策略；2×2 因子看交互。",
     "Three objects, each holding its own variable constant", "Model with the scaffold fixed, harness with the model fixed, agent with both fixed; a 2×2 for the interaction."),
    ("程序做 gate，rubric 做 score", "gated 类：验不过直接 0 分且不调用评委；验过的才由 rubric 分质量。",
     "The program gates, the rubric scores", "gated class: failing the gate scores 0 and never calls the judge; only passing answers are ranked by the rubric."),
    ("同题交错重复、逐题配对", "不是两个独立均值的差；报效应量与 bootstrap 区间。",
     "Interleaved repeats, paired per task", "Not a difference of independent means; effect size with a bootstrap interval."),
    ("配对精确检验", "McNemar（二值）、符号翻转置换（连续分），Holm 校正；不假设正态。",
     "Exact paired tests", "McNemar (binary), sign-flip permutation (continuous), Holm correction; no normality assumption."),
    ("四种结论，不是一个 p 值", "显著 / 有界的 null / 无信息 / 检验无力——缺的是不一致对还是单元，处方不同。",
     "Four verdicts, not one p-value", "Significant / bounded null / uninformative / powerless — and whether what's missing is discordant pairs or units."),
    ("有效样本与触顶诊断", "哪些题没在贡献信息、哪个系统满分导致效应被压缩。",
     "Informative-sample and ceiling diagnostics", "Which tasks contribute nothing; which system's perfect score compresses the effect."),
    ("事前样本量规划", "required_tasks / required_pairs 内部真跑将要使用的检验，不用闭式近似。",
     "Sample-size planning", "required_tasks / required_pairs run the very test that will be used, not a closed-form approximation."),
    ("rubric 自身要过 canary", "一份会被糊弄回答骗过的 rubric 不能用来评分；rubric_canary 先测分离度。",
     "The rubric itself must pass a canary", "A rubric that a bluffing answer can game is not fit to score; rubric_canary measures the separation first."),
]

VERDICTS = [
    ("显著", "效应量与区间，可直接写进报告。", "significant", "Effect size and interval, ready for a report."),
    ("有界的 null", "未检出差异，但说清能排除多大效应（MDE）。", "bounded null", "No difference detected — with the effect size this design rules out (MDE)."),
    ("无信息", "样本太小，任何效应都检不出；不能当作『无差异』的证据。", "uninformative", "Too small to detect anything; not evidence of 'no difference'."),
    ("检验无力", "p 有地板：不一致对/非零差值对不够，效应再大也到不了显著，并说明缺多少。", "powerless", "The p-value has a floor: too few discordant or nonzero-difference pairs; no effect could reach significance. Says how many are missing."),
]

RELATED = [
    ("lm-evaluation-harness", "https://github.com/EleutherAI/lm-evaluation-harness", "60+ 学术基准、多种模型后端", "60+ academic benchmarks, many model backends"),
    ("Inspect", "https://github.com/UKGovernmentBEIS/inspect_ai", "评测框架，200+ 预置评测", "Eval framework with 200+ pre-built evals"),
    ("promptfoo", "https://github.com/promptfoo/promptfoo", "side-by-side 对比与 red teaming", "Side-by-side comparison and red teaming"),
    ("openai/evals", "https://github.com/openai/evals", "模板 + JSON 数据的评测注册表", "Registry of template-based evals"),
]

CSS = """
:root{--bg:#ffffff;--fg:#1f2328;--muted:#59636e;--line:#d0d7de;--card:#f6f8fa;--accent:#3b82f6;--accent2:#f59e0b;--code:#0d1117;--codefg:#e6edf3}
@media (prefers-color-scheme:dark){:root{--bg:#0d1117;--fg:#e6edf3;--muted:#9198a1;--line:#30363d;--card:#161b22;--code:#010409;--codefg:#e6edf3}}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.6 ui-sans-serif,-apple-system,"Segoe UI",Helvetica,Arial,"PingFang SC","Noto Sans CJK SC",sans-serif}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
.wrap{max-width:960px;margin:0 auto;padding:0 20px}
[data-lang="zh"] .en,[data-lang="en"] .zh{display:none}
header.hero{padding:64px 0 40px;text-align:center;border-bottom:1px solid var(--line)}
.hero img{width:88px;height:88px}
.hero h1{margin:12px 0 4px;font-size:40px;letter-spacing:-.5px}
.hero .tagline{margin:0 auto;max-width:640px;font-size:19px;color:var(--muted)}
.actions{margin:24px 0 8px;display:flex;gap:10px;justify-content:center;flex-wrap:wrap}
.btn{display:inline-block;padding:9px 16px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg);font-weight:600}
.btn.primary{background:var(--accent);border-color:var(--accent);color:#fff}
.btn:hover{text-decoration:none;filter:brightness(1.05)}
button.lang{cursor:pointer;font:inherit}
pre{margin:0;padding:14px 16px;border-radius:10px;background:var(--code);color:var(--codefg);overflow:auto;font:13.5px/1.55 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
pre.install{display:inline-block;margin-top:18px;padding:10px 18px;font-size:15px}
.meta{color:var(--muted);font-size:14px;margin-top:14px}
section{padding:40px 0;border-bottom:1px solid var(--line)}
section h2{font-size:26px;margin:0 0 6px}
section .lead{color:var(--muted);margin:0 0 18px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px}
.card h3{margin:0 0 6px;font-size:16px}.card p{margin:0;color:var(--muted);font-size:14.5px}
table{width:100%;border-collapse:collapse;font-size:15px}th,td{text-align:left;padding:9px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:600}
.read{margin-top:14px;color:var(--muted);font-size:15px}
.bar{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:middle}
footer{padding:28px 0 48px;color:var(--muted);font-size:14px;text-align:center}
code.inl{background:var(--card);border:1px solid var(--line);border-radius:5px;padding:1px 5px;font-size:.92em}
"""

JS = """
(function(){var k='paired-eval-lang',b=document.body,btn=document.getElementById('lang');
function set(l){b.setAttribute('data-lang',l);document.documentElement.lang=l==='zh'?'zh-CN':'en';btn.textContent=l==='zh'?'English':'中文';try{localStorage.setItem(k,l)}catch(e){}}
var saved=null;try{saved=localStorage.getItem(k)}catch(e){}
set(saved||((navigator.language||'').toLowerCase().indexOf('zh')===0?'zh':'en'));
btn.addEventListener('click',function(){set(b.getAttribute('data-lang')==='zh'?'en':'zh')});})();
"""


def zh_en(zh, en, tag="span", cls=""):
    c = (" " + cls) if cls else ""
    return f'<{tag} class="zh{c}">{zh}</{tag}><{tag} class="en{c}">{en}</{tag}>'


def build():
    e = html.escape
    version = pe.__version__
    demo_zh, demo_en = demo_output(None), demo_output("en")
    cards = "\n".join(
        f'<div class="card"><h3>{zh_en(e(z1), e(e1))}</h3><p>{zh_en(e(z2), e(e2))}</p></div>'
        for z1, z2, e1, e2 in FEATURES)
    verdict_rows = "\n".join(
        f"<tr><td><strong>{zh_en(e(z1), e(e1))}</strong></td><td>{zh_en(e(z2), e(e2))}</td></tr>"
        for z1, z2, e1, e2 in VERDICTS)
    related_rows = "\n".join(
        f'<tr><td><a href="{u}">{e(n)}</a></td><td>{zh_en(e(z), e(en_))}</td></tr>' for n, u, z, en_ in RELATED)
    doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>paired-eval — 配对评测 LLM 系统 · Paired A/B evaluation for LLM systems</title>
<meta name="description" content="{ABOUT} Zero dependencies, Python 3.9+.">
<link rel="icon" type="image/svg+xml" href="assets/logo.svg">
<style>{CSS}</style>
</head>
<body data-lang="zh">
<header class="hero"><div class="wrap">
<img src="assets/logo.svg" alt="paired-eval" width="88" height="88">
<h1>paired-eval</h1>
{zh_en("给模型、agent、harness 做评测：能用程序验证的先验，验不过的再交给 rubric；每一次比较都做成统计上诚实的配对。",
       "Evaluate models, agents and harnesses: verify programmatically whatever can be verified, hand the rest to a rubric — and make every comparison a statistically honest paired one.", "p", "tagline")}
<div class="actions">
<a class="btn primary" href="{REPO}">GitHub</a>
<a class="btn" href="{REPO}/blob/main/README.md">{zh_en("README", "README（中文）")}</a>
<a class="btn" href="{REPO}/blob/main/README.en.md">{zh_en("English README", "README")}</a>
<a class="btn" href="{REPO}/blob/main/docs/README.md">{zh_en("文档", "Docs")}</a>
<button class="btn lang" id="lang" type="button">English</button>
</div>
<pre class="install"><code>{e(INSTALL)}</code></pre>
<p class="meta">v{e(version)} · MIT · Python ≥ 3.9 · {zh_en("零第三方依赖 · 模型调用由你注入", "no third-party dependencies · you inject the model call")}</p>
</div></header>

<main class="wrap">
<section id="objects">
<h2>{zh_en("它评什么", "What it evaluates")}</h2>
<p class="lead">{zh_en("三个对象问的是三个不同的问题，各自固定不同的变量；混着评什么都得不到。",
                       "The three objects ask three different questions and hold different things constant; mixing them up yields nothing.")}</p>
<table><tbody>
<tr><th>{zh_en("评什么", "Evaluate")}</th><th>{zh_en("固定什么", "Hold constant")}</th><th>{zh_en("换什么", "Vary")}</th></tr>
<tr><td><strong>model</strong></td><td>{zh_en("同一批题、同一脚手架", "same tasks, same scaffold")}</td><td>{zh_en("模型", "the model")}</td></tr>
<tr><td><strong>harness</strong></td><td>{zh_en("同一模型", "same model")}</td><td>{zh_en("提示 / 脚手架 / 工具接线", "prompt / scaffold / tool wiring")}</td></tr>
<tr><td><strong>agent</strong></td><td>{zh_en("同一模型、同一脚手架", "same model, same scaffold")}</td><td>{zh_en("策略（单遍 vs 自检修正…）", "the strategy (one pass vs self-check…)")}</td></tr>
<tr><td><strong>{zh_en("交互", "interaction")}</strong></td><td colspan="2">{zh_en("2×2 因子设计：同一把尺子上比主效应，自动报出天花板", "2×2 factorial: main effects on one scale, ceilings flagged automatically")}</td></tr>
</tbody></table>
</section>

<section id="verify">
<h2>{zh_en("怎么验：能用程序验的先验，验不过的再判", "How it verifies: programmatic checks first, judges only for the rest")}</h2>
<p class="lead">{zh_en("有唯一真值 → exact 或程序判定；只有事实来源 → retrieval / trajectory（逐条 claim grounding）；只有质量标准 → rubric（自身要过 canary）。层可叠加：<strong>程序做 gate，rubric 做 score</strong>——验不过直接 0 分且不调用评委。",
                       "A unique ground truth → exact or a programmatic check; only sources of fact → retrieval / trajectory (per-claim grounding); only a quality bar → rubric (which must pass a canary). Layers stack: <strong>the program gates, the rubric scores</strong> — failing the gate scores 0 and never calls the judge.")}</p>
<pre class="zh"><code>{e(GATED_CODE_ZH)}</code></pre>
<pre class="en"><code>{e(GATED_CODE_EN)}</code></pre>
</section>

<section id="demo">
<h2>{zh_en("十秒钟看效果（统计层的离线演示）", "Ten seconds (an offline demo of the statistics layer)")}</h2>
<p class="lead">{zh_en("<strong>这里的 strict / bare 是两个桩函数，不是模型</strong>——它们复刻一种真实观察到的失败机理（裸指令下正确答案被包进 markdown 围栏），只为在没有 API 时展示报告的形状。输出是真的（由测试与当前代码逐行比对）。",
                       "<strong>strict / bare here are two stub functions, not models</strong> — they replay one real failure mechanism (a bare prompt wraps the correct JSON in markdown fences), only to show the shape of a report without any API. The output is real (a test keeps it identical to the current code).")}</p>
<pre class="zh"><code>$ python3 paired_eval.py
{e(demo_zh)}</code></pre>
<pre class="en"><code>$ python3 paired_eval.py --lang en
{e(demo_en)}</code></pre>
<p class="read">{zh_en("两题两系统全对、不携带信息（有效样本 2/4）；8 个不一致对全部偏向 strict，分布在 2 题上（集中度 0.50，非单题异常）；逐题置换 p=0.506 是因为只有 2 道有信息题，它的最小可能 p 就是 0.5——而逐轮 McNemar 用上全部 16 个配对单元。末段是可直接写进报告的结论。",
                        "Two tasks were solved by both systems every time and carry no information (informative sample 2/4); all 8 discordant pairs favour strict, spread over 2 tasks (concentration 0.50); the per-task permutation p is 0.506 because with 2 informative tasks its minimum attainable p is 0.5 — per-round McNemar uses all 16 paired units. The last clause is a verdict you can paste into a report.")}</p>
</section>

<section id="features">
<h2>{zh_en("特性", "Features")}</h2>
<div class="grid">
{cards}
</div>
</section>

<section id="verdicts">
<h2>{zh_en("四种结论，而不是一个 p 值", "Four verdicts, not one p-value")}</h2>
<p class="lead">{zh_en("“p > 0.05” 有三种完全不同的含义，处方各异。<code class='inl'>interpret()</code> 把每次比较翻译成其中一种：",
                       "“p > 0.05” means three very different things with different remedies. <code class='inl'>interpret()</code> translates every comparison into one of these:")}</p>
<table><tbody>
{verdict_rows}
</tbody></table>
</section>

<section id="usage">
<h2>{zh_en("用真实模型：三个对象各做一次 A/B", "With real models: one A/B per object")}</h2>
<p class="lead">{zh_en("任务按“有什么可验”混搭验证类；<code class='inl'>bench_tasks</code> 把 judge 评分的题按显式阈值二值化，进同一套配对流水线；三个维度各固定各的变量。",
                       "Mix verification classes by what can be verified; <code class='inl'>bench_tasks</code> binarises judge-scored tasks at an explicit threshold so they flow through the same paired pipeline; each axis holds its own variable constant.")}</p>
<pre class="zh"><code>{e(TASKS_CODE_ZH)}</code></pre>
<pre class="en"><code>{e(TASKS_CODE_EN)}</code></pre>
<p class="read">{zh_en("agent 维度同理：同一模型同一脚手架，比“单遍”与“出草稿后自检修正”。还能做：事前样本量规划（required_tasks / required_pairs）、筛出有区分力的题（screen_tasks / screen_graded）。完整示例见 README。",
                        "The agent axis works the same way: same model, same scaffold, one pass vs draft-then-self-correct. Also: sample-size planning (required_tasks / required_pairs) and screening for tasks that discriminate (screen_tasks / screen_graded). Full example in the README.")}</p>
</section>

<section id="related">
<h2>{zh_en("与其他工具的关系", "How it relates to other tools")}</h2>
<p class="lead">{zh_en("它们运行评测、产出逐题分数；paired-eval 接在下游，回答“这个差别可信吗、还差多少样本”。描述取自各项目自己的 README。",
                       "They run evaluations and produce per-task scores; paired-eval sits downstream and answers “is this difference credible, and how many more samples would it take?”. Descriptions come from each project's own README.")}</p>
<table><tbody>
{related_rows}
</tbody></table>
</section>

<section id="docs">
<h2>{zh_en("文档", "Documentation")}</h2>
<div class="grid">
<div class="card"><h3><a href="{REPO}/blob/main/docs/lessons.md">{zh_en("方法学教训", "Methodology lessons")}</a></h3><p>{zh_en("每个统计原语防的是什么错：p 地板、地板的基数、MDE、null 必须附界、天花板、筛选可靠性……", "What each primitive guards against: the p floor and its basis, MDE, nulls with bounds, ceilings, screening reliability …")}</p></div>
<div class="card"><h3><a href="{REPO}/blob/main/docs/findings.md">{zh_en("案例研究", "Case study")}</a></h3><p>{zh_en("用本工具对一对模型、三族任务做的实测；数字是实例特定的，展示的是报告该怎么写。", "One model pair, three task families; the numbers are instance-specific and show how a conclusion should be written.")}</p></div>
<div class="card"><h3><a href="{REPO}/blob/main/docs/corrections.md">{zh_en("纠正记录", "Corrections")}</a></h3><p>{zh_en("被推翻或修正过的结论，原文保留、指向新证据。", "Conclusions that were overturned or corrected, kept in place and pointing at the new evidence.")}</p></div>
<div class="card"><h3><a href="{REPO}/blob/main/CONTRIBUTING.md">{zh_en("参与贡献", "Contributing")}</a></h3><p>{zh_en("钩子、变异测试基线、统计代码的额外纪律。", "Hooks, the mutation-testing baseline, extra discipline for statistical code.")}</p></div>
<div class="card"><h3><a href="{REPO}/blob/main/CHANGELOG.md">{zh_en("更新日志", "Changelog")}</a></h3><p>{zh_en("Keep a Changelog 格式；", "Keep a Changelog format; ")}<a href="{REPO}/releases">{zh_en("发布页", "releases")}</a></p></div>
<div class="card"><h3><a href="{REPO}/blob/main/examples/adapter_openai_compat.py">{zh_en("OpenAI 兼容适配器", "OpenAI-compatible adapter")}</a></h3><p>{zh_en("只用标准库；把任意兼容端点接成 call / judge。", "Standard library only; turns any compatible endpoint into call / judge.")}</p></div>
</div>
</section>
</main>

<footer class="wrap">
<span class="bar" style="background:var(--accent)"></span><span class="bar" style="background:var(--accent2)"></span>
paired-eval v{e(version)} · MIT · <a href="{REPO}">{REPO.replace("https://", "")}</a> · {zh_en("本页无外部脚本、样式或字体。", "This page loads no external scripts, styles or fonts.")}
</footer>
<script>{JS}</script>
</body>
</html>
"""
    return doc


def main(argv):
    doc = build()
    if "--check" in argv:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current != doc:
            print("docs/index.html 与生成器输出不一致 —— 运行 python3 build_site.py 重新生成")
            return 1
        print("docs/index.html 与生成器一致")
        return 0
    OUT.write_text(doc, encoding="utf-8")
    print(f"已写入 {OUT.relative_to(ROOT)} ({len(doc.splitlines())} 行, v{pe.__version__})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
