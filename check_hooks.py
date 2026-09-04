#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""钩子集成检查: pre-commit 与 pre-push 是强制层, 但它们的逻辑(上游探测、
退化到 HEAD~1、调用 runtests/变异检查)此前只做过手工验证, 会静默回归。

刻意不叫 test_*.py: 它要建临时仓库+裸远端并跑真实钩子, 约30秒。
放进 1.5 秒的快速套件会让 pre-commit 变慢, 而慢门禁会被 --no-verify 绕过 ——
那比没有门禁更糟。分层: 快检查进套件, 慢检查在外层手动/CI 跑。

用法: python3 check_hooks.py
"""
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent
GIT_ID = ["-c", "user.name=hookcheck", "-c", "user.email=hook@check"]
UNTESTED = '''def score_band(score, high=0.8):
    """无测试保护的新函数(检查用)。"""
    return "high" if (score or 0) >= high else "low"


'''


def git(repo, *args, check=True):
    return subprocess.run(["git", *GIT_ID, *args], cwd=repo,
                          capture_output=True, text=True, check=check)


BRANCH = "hookcheck"   # 临时工作仓统一用这个分支名, 与真实仓库当前在哪个分支无关


def setup(tmp):
    """从真实仓库克隆一份工作仓 + 一个裸远端, 并启用版本化钩子。
    克隆后立刻切到固定分支名: 初版硬编码 push `main`, 在 feature 分支或 PR 的 detached
    checkout 上跑就会 "src refspec main does not match any"(Dependabot 的第一个 PR 就撞上了)。
    钩子检查验证的是钩子的行为, 不该依赖被检出的分支叫什么。"""
    work = pathlib.Path(tmp) / "work"
    bare = pathlib.Path(tmp) / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)
    subprocess.run(["git", "clone", "-q", str(ROOT), str(work)], check=True)
    git(work, "checkout", "-q", "-B", BRANCH)          # detached HEAD 也能落到一个具名分支
    git(work, "config", "core.hooksPath", "hooks")
    git(work, "remote", "add", "bare", str(bare))
    return work, bare


def case_precommit_rejects_broken_test(work):
    p = work / "tests" / "test_answer_match.py"
    p.write_text(p.read_text(encoding="utf-8") +
                 "\n\ndef test_injected_failure():\n    assert False\n", encoding="utf-8")
    git(work, "add", "-A")
    r = git(work, "commit", "-m", "should be rejected", check=False)
    # 被拒的提交会把改动留在索引里, `checkout -- .` 是从索引恢复(等于没恢复) ——
    # 必须 reset --hard, 否则坏测试泄漏到后续用例, 全线失败(初版实测踩到)。
    git(work, "reset", "-q", "--hard", "HEAD")
    assert r.returncode != 0, "pre-commit 应拒绝失败的测试"
    assert "提交被拒" in r.stdout + r.stderr, r.stdout + r.stderr
    return "pre-commit 拒绝失败测试"


def case_precommit_accepts_clean(work):
    (work / "notes.txt").write_text("clean change\n", encoding="utf-8")
    git(work, "add", "-A")
    r = git(work, "commit", "-m", "clean change", check=False)
    assert r.returncode == 0, f"干净改动应放行: {r.stdout}{r.stderr}"
    return "pre-commit 放行干净改动"


def case_prepush_accepts_clean(work):
    r = git(work, "push", "bare", BRANCH, check=False)
    assert r.returncode == 0, f"干净推送应通过: {r.stdout}{r.stderr}"
    return "pre-push 放行干净推送"


def case_prepush_rejects_untested_code(work):
    p = work / "paired_eval" / "rubric_eval.py"
    src = p.read_text(encoding="utf-8")
    p.write_text(src.replace("def aggregate_rubric(results):",
                             UNTESTED + "def aggregate_rubric(results):", 1), encoding="utf-8")
    git(work, "add", "-A")
    git(work, "commit", "-m", "add untested helper")
    r = git(work, "push", "bare", BRANCH, check=False)
    out = r.stdout + r.stderr
    assert r.returncode != 0, f"含未测代码的推送应被拒: {out}"
    assert "推送被拒" in out and "新增存活" in out, out
    return "pre-push 拒绝未测代码"


def main():
    cases = [case_precommit_rejects_broken_test, case_precommit_accepts_clean,
             case_prepush_accepts_clean, case_prepush_rejects_untested_code]
    with tempfile.TemporaryDirectory() as tmp:
        work, _ = setup(tmp)
        failed = 0
        for case in cases:
            try:
                print(f"ok  {case(work)}")
            except AssertionError as exc:
                failed += 1
                print(f"FAIL {case.__name__}: {exc}")
        print(f"\n{len(cases) - failed}/{len(cases)} 项钩子检查通过")
        return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
