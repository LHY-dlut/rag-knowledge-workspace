"""Start the existing isolated acceptance stack; never reset volumes or print secrets."""

import argparse
import csv
import json
import os
import secrets
import subprocess
from datetime import datetime
from pathlib import Path
from uuid import uuid4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["setup", "start", "stop", "status"])
    parser.add_argument("--real-models", action="store_true")
    parser.add_argument("--credentials", type=Path)
    parser.add_argument("--backend-image", default="rag-acceptance-r2-backend:engineering-v1")
    parser.add_argument("--report-dir", type=Path, default=Path("artifacts/runtime_startup"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    private = root / ".acceptance/round2"
    envfile = private / "database.env"
    if args.action == "setup" and not envfile.exists():
        private.mkdir(parents=True, exist_ok=True)
        values = {
            key: secrets.token_hex(24)
            for key in ("MYSQL_PASSWORD", "MYSQL_ROOT_PASSWORD", "POSTGRES_PASSWORD", "JWT_SECRET")
        }
        envfile.write_text(
            "\n".join(f"{k}={v}" for k, v in values.items()) + "\n", encoding="utf-8"
        )
        (private / "database_private.json").write_text(
            json.dumps(
                {
                    "TEST_BUSINESS_DB_URL": f"mysql://rag:{values['MYSQL_PASSWORD']}@127.0.0.1:13307/rag_business",
                    "TEST_VECTOR_DB_URL": f"postgres://rag:{values['POSTGRES_PASSWORD']}@127.0.0.1:15433/rag_vectors",
                    "ACCEPTANCE_ISOLATED_DATABASES": "1",
                }
            ),
            encoding="utf-8",
        )
    if not envfile.exists():
        raise SystemExit("Missing local isolated environment: .acceptance/round2/database.env")
    env = dict(os.environ)
    env["MODEL_PROVIDER"] = "demo"
    if args.real_models:
        credentials = dict(
            csv.reader(
                (args.credentials or root.parents[1] / "默认业务空间-apiKey-7545941.csv")
                .read_text(encoding="utf-8-sig")
                .splitlines()
            )
        )
        env.update(
            MODEL_PROVIDER="dashscope",
            DASHSCOPE_API_KEY=credentials["apiKey"],
            DASHSCOPE_HTTP_BASE_URL=credentials["dashScope"],
            DASHSCOPE_CHAT_BASE_URL=credentials["openAiCompatible"],
        )
    override = private / "application.yml"
    override.write_text(
        f"""services:
  mysql:
    ports: ["127.0.0.1:13307:3306"]
  postgres:
    ports: ["127.0.0.1:15433:5432"]
  migrate:
    image: {args.backend_image}
  api:
    image: {args.backend_image}
    ports: ["127.0.0.1:18086:8000"]
    environment:
      ENABLE_API_WORKERS: "false"
  worker:
    image: {args.backend_image}
  frontend:
    image: rag-acceptance-r2-frontend:local
    ports: !override ["127.0.0.1:18088:80"]
""",
        encoding="utf-8",
    )
    command = [
        "docker",
        "compose",
        "--project-name",
        "rag-acceptance-r2",
        "--env-file",
        str(envfile),
        "-f",
        str(root / "compose.yml"),
        "-f",
        str(override),
    ]
    command += {
        "setup": ["up", "-d", "--build", "--wait"],
        "start": ["up", "-d", "--no-build", "--wait"],
        "stop": ["stop"],
        "status": ["ps"],
    }[args.action]
    result = subprocess.run(
        command, env=env, capture_output=True, encoding="utf-8", errors="replace"
    )
    output = result.stdout + result.stderr
    for key in ["DASHSCOPE_API_KEY"]:
        if env.get(key):
            output = output.replace(env[key], "<redacted>")
    print(output)
    report_dir = root / args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
    (report_dir / f"compose_{args.action}_{stamp}.txt").write_text(output, encoding="utf-8")
    if args.action in {"setup", "start"} and result.returncode == 0:
        print(
            json.dumps(
                {
                    "frontend": "http://127.0.0.1:18088",
                    "api": "http://127.0.0.1:18086",
                    "provider": env["MODEL_PROVIDER"],
                }
            )
        )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
