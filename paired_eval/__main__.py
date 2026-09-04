"""`python3 -m paired_eval [--lang en]` — 离线 demo, 无需 API。"""
import sys

from . import demo

args = sys.argv[1:]
lang = args[args.index("--lang") + 1] if "--lang" in args and args.index("--lang") + 1 < len(args) else None
demo(lang=lang)
