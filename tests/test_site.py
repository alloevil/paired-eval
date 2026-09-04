#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""项目主页 docs/index.html 的守卫: 它是生成的, 贴的是事实, 且零外部依赖。

- 文件必须等于 build_site.py 此刻的输出 —— 手改 HTML 或改了 demo/版本却没重新生成, 这里会红。
- demo 输出(两种语言)与版本号必须与当前代码一致(生成器保证, 这里独立复核)。
- 不得引用任何外链脚本 / 样式 / 字体; 页内相对资源(logo)必须存在。
- HTML 标签配平(用标准库 html.parser 走一遍)。
- README 头部引用的 logo 与主页 URL 必须存在/一致。"""
import pathlib as _pathlib
import sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))  # 项目根: 让 `python3 tests/x.py` 直接可跑
import html
import re
from html.parser import HTMLParser

import build_site
import paired_eval as pe

ROOT = _pathlib.Path(__file__).resolve().parent.parent
INDEX = ROOT / "docs" / "index.html"


def test_index_html_is_up_to_date_with_generator():
    assert INDEX.exists(), "docs/index.html 不存在: 运行 python3 build_site.py"
    assert INDEX.read_text(encoding="utf-8") == build_site.build(), \
        "docs/index.html 与生成器输出不一致 —— 运行 python3 build_site.py 重新生成(不要手改 HTML)"
    assert build_site.main(["--check"]) == 0


def test_index_html_states_true_facts():
    src = INDEX.read_text(encoding="utf-8")
    for lang in (None, "en"):
        lines = []
        pe.demo(out=lambda s: lines.extend(s.splitlines()), lang=lang)
        assert html.escape("\n".join(lines)) in src, f"主页贴的 demo 输出({lang or 'zh'})与实际不一致"
    assert f"v{pe.__version__}" in src, "主页版本号过期"
    assert html.escape(build_site.INSTALL) in src
    # 主页的安装命令与 README 的一致
    for name in ("README.md", "README.en.md"):
        assert build_site.INSTALL in (ROOT / name).read_text(encoding="utf-8"), name
    # 文档卡片链到的文件都存在
    for rel in re.findall(r'href="https://github\.com/alloevil/paired-eval/blob/main/([^"]+)"', src):
        assert (ROOT / rel).exists(), f"主页链接了不存在的文件: {rel}"


class _Check(HTMLParser):
    VOID = {"meta", "link", "img", "br", "hr", "input"}

    def __init__(self):
        super().__init__()
        self.stack, self.problems, self.external, self.local = [], [], [], []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag not in self.VOID:
            self.stack.append(tag)
        if tag == "script" and a.get("src"):
            self.external.append(a["src"])
        if tag == "link" and a.get("rel") == "stylesheet":
            self.external.append(a.get("href"))
        if tag in ("img", "link") and a.get("href" if tag == "link" else "src", "").startswith(("http://", "https://")):
            self.external.append(a.get("href" if tag == "link" else "src"))
        for k in ("src", "href"):
            v = a.get(k)
            if v and not v.startswith(("http://", "https://", "#", "mailto:")):
                self.local.append(v)

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.problems.append(f"</{tag}> 与栈顶 {self.stack[-1:]} 不配")
        else:
            self.stack.pop()


def test_index_html_is_self_contained_and_well_formed():
    src = INDEX.read_text(encoding="utf-8")
    c = _Check()
    c.feed(src)
    assert not c.problems and not c.stack, (c.problems, c.stack)
    assert c.external == [], f"主页不得加载外部资源(与零依赖一致): {c.external}"
    for rel in c.local:
        assert (INDEX.parent / rel).exists(), f"页内相对资源不存在: {rel}"
    assert "@import" not in src and "url(http" not in src, "CSS 里也不许外链"
    assert 'lang="zh-CN"' in src and 'data-lang="zh"' in src
    assert (ROOT / "docs" / ".nojekyll").exists(), "Pages 需 .nojekyll: 静态提供 index.html, 不经 Jekyll"


def test_readme_headers_reference_existing_logo_and_site():
    for name in ("README.md", "README.en.md"):
        t = (ROOT / name).read_text(encoding="utf-8")
        assert 'src="docs/assets/logo.svg"' in t, f"{name}: 头部应用仓库内的 logo"
        assert (ROOT / "docs/assets/logo.svg").exists()
        assert "https://alloevil.github.io/paired-eval/" in t, f"{name}: 应链到主页"
        assert '<h1 align="center">paired-eval</h1>' in t


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} tests passed")
