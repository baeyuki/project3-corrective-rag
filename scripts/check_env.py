"""Environment check.

    python scripts/check_env.py           # Python version, packages, .env
    python scripts/check_env.py --ping    # also one tiny chat call and one embedding call (cost < $0.0001)
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ["openai", "numpy", "pytest"]


def load_env(path: Path = ROOT / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ping", action="store_true", help="make one small API call per model")
    args = parser.parse_args()
    ok = True

    print(f"Python {sys.version.split()[0]}", "OK" if sys.version_info >= (3, 10) else "-> need 3.10+")
    ok &= sys.version_info >= (3, 10)
    for name in REQUIRED:
        try:
            mod = importlib.import_module(name)
            print(f"{name:8s} {getattr(mod, '__version__', '?')}")
        except ImportError:
            print(f"{name:8s} MISSING -> pip install -r requirements.txt")
            ok = False

    load_env()
    key = os.getenv("OPENAI_API_KEY", "")
    print("OPENAI_API_KEY", "set" if key else "MISSING -> copy .env.example to .env and fill it")
    ok &= bool(key)

    if args.ping and ok:
        from openai import OpenAI
        client = OpenAI(api_key=key, timeout=30)
        chat = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"), max_tokens=5,
            messages=[{"role": "user", "content": "Trả lời đúng một từ: xin chào"}])
        print("chat     ", repr(chat.choices[0].message.content), f"({chat.usage.total_tokens} tokens)")
        emb = client.embeddings.create(model=os.getenv("EMBED_MODEL", "text-embedding-3-small"), input=["bảo hành"])
        print("embedding", f"dim={len(emb.data[0].embedding)}")

    print("ALL OK" if ok else "FIX THE ITEMS ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
