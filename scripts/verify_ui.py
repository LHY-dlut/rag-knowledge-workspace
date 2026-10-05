"""Optional UI smoke verification; ports 8000 and 5173 must be free."""

import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
from verification_runtime import isolated_runtime, stop_process


def main():
    root = Path(__file__).resolve().parent.parent
    with isolated_runtime(prefix="rag-ui-") as directory:
        runtime = Path(directory)
        env = {
            **os.environ,
            "APP_MODE": "demo",
            "MODEL_PROVIDER": "demo",
            "ENABLE_API_WORKERS": "false",
            "WORKER_POLL_SECONDS": "0.1",
            "BUSINESS_DB_URL": f"sqlite://{runtime}/business.sqlite3",
            "VECTOR_DB_URL": f"sqlite://{runtime}/vectors.sqlite3",
            "UPLOAD_DIR": str(runtime / "uploads"),
            "VITE_API_TARGET": "http://127.0.0.1:" + os.environ.get("UI_API_PORT", "18081"),
            "UI_BASE_URL": "http://127.0.0.1:" + os.environ.get("UI_PORT", "15173"),
        }
        if os.environ.get("ACCEPTANCE_ISOLATED_DATABASES") == "1":
            env.update(
                APP_MODE="production",
                BUSINESS_DB_URL=os.environ["TEST_BUSINESS_DB_URL"],
                VECTOR_DB_URL=os.environ["TEST_VECTOR_DB_URL"],
            )
        processes = []
        with (runtime / "api.log").open("w") as api_log, (runtime / "ui.log").open("w") as ui_log:
            subprocess.run(
                [sys.executable, "-m", "app.bootstrap"],
                cwd=root,
                env=env,
                check=True,
                stdout=api_log,
                stderr=api_log,
            )
            processes.append(
                subprocess.Popen(
                    [sys.executable, "-m", "scripts.acceptance_process_worker"],
                    cwd=root,
                    env=env,
                    stdout=api_log,
                    stderr=api_log,
                )
            )
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "uvicorn",
                        "app.main:app",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        os.environ.get("UI_API_PORT", "18081"),
                    ],
                    cwd=root,
                    env=env,
                    stdout=api_log,
                    stderr=api_log,
                )
            )
            processes.append(
                subprocess.Popen(
                    [
                        "node",
                        "node_modules/vite/bin/vite.js",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        os.environ.get("UI_PORT", "15173"),
                        "--strictPort",
                    ],
                    cwd=root / "frontend",
                    env=env,
                    stdout=ui_log,
                    stderr=ui_log,
                )
            )
            try:
                with httpx.Client(trust_env=False) as client:
                    for url in (env["VITE_API_TARGET"] + "/api/health", env["UI_BASE_URL"]):
                        for _ in range(150):
                            try:
                                if client.get(url).status_code == 200:
                                    break
                            except httpx.TransportError:
                                pass
                            if any(p.poll() is not None for p in processes):
                                raise RuntimeError(
                                    (runtime / "api.log").read_text(
                                        encoding="utf-8", errors="replace"
                                    )
                                    + (runtime / "ui.log").read_text(
                                        encoding="utf-8", errors="replace"
                                    )
                                )
                            time.sleep(0.1)
                        else:
                            raise TimeoutError(f"Server failed to start: {url}")
                subprocess.run(["node", "scripts/ui_e2e.cjs"], cwd=root, env=env, check=True)
            finally:
                for process in processes:
                    stop_process(process)


if __name__ == "__main__":
    main()
