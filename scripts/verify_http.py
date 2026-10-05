"""Start an isolated demo server and exercise it over real local TCP/HTTP."""

import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
from verification_runtime import isolated_runtime, stop_process


def main():
    root = Path(__file__).resolve().parent.parent
    port = int(os.environ.get("HTTP_TEST_PORT", "18080"))
    with isolated_runtime(prefix="rag-http-") as directory:
        runtime = Path(directory)
        env = {
            **os.environ,
            "APP_MODE": "demo",
            "MODEL_PROVIDER": "demo",
            "BUSINESS_DB_URL": f"sqlite://{runtime}/business.sqlite3",
            "VECTOR_DB_URL": f"sqlite://{runtime}/vectors.sqlite3",
            "UPLOAD_DIR": str(runtime / "uploads"),
        }
        with (runtime / "server.log").open("w") as log:
            server = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "app.main:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                ],
                cwd=root,
                env=env,
                stdout=log,
                stderr=log,
            )
            try:
                with httpx.Client(trust_env=False) as client:
                    for _ in range(100):
                        try:
                            response = client.get(f"http://127.0.0.1:{port}/api/health")
                            if response.status_code == 200:
                                break
                        except httpx.TransportError:
                            pass
                        if server.poll() is not None:
                            raise RuntimeError(
                                (runtime / "server.log").read_text(
                                    encoding="utf-8", errors="replace"
                                )
                            )
                        time.sleep(0.1)
                    else:
                        raise TimeoutError("Test server startup failed")
                subprocess.run(
                    [
                        sys.executable,
                        "scripts/smoke_http.py",
                        "--url",
                        f"http://127.0.0.1:{port}",
                        "--report",
                        os.environ.get("HTTP_REPORT", "artifacts/smoke_http.json"),
                    ],
                    cwd=root,
                    env=env,
                    check=True,
                )
            finally:
                stop_process(server)


if __name__ == "__main__":
    main()
