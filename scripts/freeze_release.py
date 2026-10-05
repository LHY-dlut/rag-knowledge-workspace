"""Freeze public source/configuration hashes without including private credentials."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def inventory():
    folders = (
        "app",
        "scripts",
        "tests",
        "docs",
        "infra",
        "examples",
        "frontend/src",
        "frontend/tests",
    )
    names = (
        "README.md",
        "pyproject.toml",
        "requirements.lock",
        "Dockerfile",
        "compose.yml",
        "compose.dev.yml",
        "compose.acceptance.yml",
        ".env.example",
        ".env.production.example",
        ".dockerignore",
        ".gitignore",
        "frontend/package.json",
        "frontend/package-lock.json",
        "frontend/vite.config.js",
        "frontend/index.html",
        "frontend/Dockerfile",
        "artifacts/human_eval_20261002_v1/confirmed_dataset.json",
    )
    files = {ROOT / name for name in names if (ROOT / name).is_file()}
    for folder in folders:
        files.update(
            p
            for p in (ROOT / folder).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.suffix not in {".pyc", ".pyo"}
        )
    return {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(files)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "verify"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", default="production-candidate-v1")
    args = parser.parse_args()
    current = inventory()
    if args.action == "prepare":
        # Do not replace a previously frozen revision.
        with args.output.open("x", encoding="utf-8") as file:
            json.dump(
                {
                    "version": args.version,
                    "files": current,
                    "private_configuration": "excluded; preserve locally, never ship secrets",
                },
                file,
                ensure_ascii=False,
                indent=2,
            )
    else:
        frozen = json.loads(args.output.read_text(encoding="utf-8"))["files"]
        if frozen != current:
            changed = sorted(
                k for k in frozen.keys() | current.keys() if frozen.get(k) != current.get(k)
            )
            raise SystemExit(f"Frozen source/configuration changed: {changed}")
    print(f"{args.action}: {len(current)} source/configuration files verified")


if __name__ == "__main__":
    main()
