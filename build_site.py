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
INSTALL = f"pip install git+{REPO}.git"


def demo_output(lang):
    lines = []
    pe.demo(out=lambda s: lines.extend(s.splitlines()), lang=lang)
    return "\n".join(lines)


TASKS_CODE_ZH = '''import paired_eval as pe

tasks = [
    {"id": "date-iso", "instruction": "把 2024年3月5日 写成 ISO 8601 日期, 只输出日期",
     "check": lambda r: r.strip() == "2024-03-05", "canonical": "2024-03-05"},
    {"id": "json-pair", "instruction": '输出 JSON 对象 {"a": 1}, 不要其他内容',
     "check": lambda r: r.strip() == '{"a": 1}', "canonical": '{"a": 1}'},
]
systems = {"strict": pe.make_model(call_strict), "bare": pe.make_model(call_bare)}   # 任何 call(prompt) -> str

run = pe.run_interleaved(systems, tasks=tasks, n=8, prompt_prefix="")
print(pe.report(run["reports"], refusals=run["refusals"])["text"])'''

TASKS_CODE_EN = TASKS_CODE_ZH.replace(
    '"把 2024年3月5日 写成 ISO 8601 日期, 只输出日期"', '"Write 5 March 2024 as an ISO 8601 date. Output only the date."'
).replace(
    "'输出 JSON 对象 {\"a\": 1}, 不要其他内容'", "'Output the JSON object {\"a\": 1} and nothing else.'"
).replace("# 任何 call(prompt) -> str", "# any call(prompt) -> str")

FEATURES = [
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
    ("可组合的验证器", "精确匹配 → 程序判定 → 检索核实 → agent 轨迹忠实性 → rubric（自带 canary）。",
     "Composable verifiers", "Exact match → programmatic check → retrieval → agent-trajectory faithfulness → rubric (with a canary)."),
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
<meta name="description" content="Paired A/B evaluation for LLM systems with honest statistics: is the difference real, and is the sample big enough? Zero dependencies, Python 3.9+.">
<link rel="icon" type="image/svg+xml" href="assets/logo.svg">
<style>{CSS}</style>
</head>
<body data-lang="zh">
<header class="hero"><div class="wrap">
<img src="assets/logo.svg" alt="paired-eval" width="88" height="88">
<h1>paired-eval</h1>
{zh_en("比较两个 LLM 系统——不同的模型、提示、脚手架或 agent 策略——哪个更好，结论有多可靠，样本够不够。",
       "Compare two LLM systems — models, prompts, scaffolds or agent strategies — and learn which is better, how reliable that is, and whether you have enough samples.", "p", "tagline")}
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
<section id="demo">
<h2>{zh_en("十秒钟看效果", "Ten seconds")}</h2>
<p class="lead">{zh_en("两个桩系统在 4 道内置格式题上做配对 A/B，不需要任何 API。下面是真实输出（由测试与当前代码逐行比对）。",
                       "Two stub systems run a paired A/B on 4 built-in format tasks, no API needed. Actual output below (a test keeps it identical to the current code).")}</p>
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
<h2>{zh_en("用你自己的系统和任务", "Your own systems and tasks")}</h2>
<p class="lead">{zh_en("一个任务 = 指令 + 程序判定器 + 一个合法输出样例。任何 <code class='inl'>call(prompt) -&gt; str</code> 都能接上；<code class='inl'>make_model</code> 负责有界重试，持续失败转为“拒答”成对丢弃。",
                       "A task = an instruction + a programmatic checker + one valid output. Any <code class='inl'>call(prompt) -&gt; str</code> plugs in; <code class='inl'>make_model</code> adds bounded retries and drops persistent failures pairwise as refusals.")}</p>
<pre class="zh"><code>{e(TASKS_CODE_ZH)}</code></pre>
<pre class="en"><code>{e(TASKS_CODE_EN)}</code></pre>
<p class="read">{zh_en("还能做：评 agent 回答是否忠实于它看到的观察（逐条 claim grounding）、rubric 评分并检验 rubric 自身、事前样本量规划、筛出有区分力的题。见 README。",
                        "Also: score whether an agent's answer is faithful to its observations (per-claim grounding), rubric judging with a self-check canary, sample-size planning, and screening for tasks that discriminate. See the README.")}</p>
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
