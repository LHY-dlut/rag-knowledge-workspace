"""Generate a local configuration without printing secrets."""

import argparse
import secrets
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["demo", "production"], default="demo")
    args = parser.parse_args()
    destination = Path(".env")
    if destination.exists():
        raise SystemExit(".env 已存在；请直接编辑，脚本不会覆盖现有配置。")
    source = Path(".env.example" if args.mode == "demo" else ".env.production.example").read_text()
    if args.mode == "production":
        for key in ("JWT_SECRET", "MYSQL_PASSWORD", "MYSQL_ROOT_PASSWORD", "POSTGRES_PASSWORD"):
            value = secrets.token_hex(32)
            source = source.replace(f"{key}=\n", f"{key}={value}\n")
            if key == "MYSQL_PASSWORD":
                source = source.replace("mysql://rag:SET_PASSWORD", f"mysql://rag:{value}")
            if key == "POSTGRES_PASSWORD":
                source = source.replace("postgres://rag:SET_PASSWORD", f"postgres://rag:{value}")
    destination.write_text(source, encoding="utf-8")
    destination.chmod(0o600)
    print(f"已写入 .env（{args.mode}）；密钥不会输出到终端。")


if __name__ == "__main__":
    main()
