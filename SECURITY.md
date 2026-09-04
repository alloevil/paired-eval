# Security Policy

## Scope

`paired-eval` is a library and a set of local scripts. It makes **no network calls** except through callables you inject
(`call`, `llm`, `search`, `judge`). The only networking code in the repository is `examples/adapter_openai_compat.py`,
which sends the prompts you pass it to the endpoint you configure and nothing else; it never logs or stores API keys.

Things worth knowing when you evaluate untrusted model output with this toolbox:

- `answer_match` and `paired_bench` **never execute** model output; grading is string/JSON parsing only.
- `mutate_auto.py` and `mutate.sh` **rewrite source files in place** while running (with crash-safe sidecar backups).
  Run them only inside a checkout you control, never on a shared working tree.
- The hooks under `hooks/` run `runtests.sh` and `mutate_auto.py`; they execute repository code, as any test hook does.

## Reporting a vulnerability

Open a private security advisory on GitHub (**Security → Report a vulnerability**) or an issue if the problem is not
sensitive. Please include the affected file, a minimal reproduction, and the Python version. You will get a reply within
a week; fixes land on `main` with a note in [CHANGELOG.md](CHANGELOG.md).

## 中文

本仓库不主动联网，唯一的网络代码是 `examples/` 里的适配器示例，只向你配置的端点发送你传入的提示词。
`mutate_auto.py` / `mutate.sh` 运行时会原地改写源文件（有崩溃安全的旁路备份），只在你自己的检出里跑。
安全问题请走 GitHub 的私密安全通告；一周内回复。
