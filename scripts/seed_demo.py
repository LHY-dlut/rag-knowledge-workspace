"""Create three screenshot-style applications via the same public APIs as Vue."""

import argparse
import getpass
import time
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--username", required=True)
    args = parser.parse_args()
    password = getpass.getpass("账号密码（不会回显）：")
    root = Path(__file__).resolve().parent.parent
    with httpx.Client(base_url=args.url, timeout=120, trust_env=False) as client:
        body = {"username": args.username, "password": password}
        response = client.post("/api/auth/login", json=body)
        if response.status_code == 401:
            response = client.post("/api/auth/register", json=body)
        response.raise_for_status()
        client.headers["Authorization"] = "Bearer " + response.json()["data"]["access_token"]

        def post(path, **kwargs):
            result = client.post(path, **kwargs)
            result.raise_for_status()
            return result.json()["data"]

        samples = [
            ("产品技术", "产品技术手册.md", "recursive", "产品技术助手", "rag"),
            ("企业制度", "差旅与休假制度.md", "parent_child", "企业制度助手", "agent"),
            ("售后服务", "售后服务规范.md", "recursive_short", "售后服务助手", "agent"),
        ]
        for name, filename, strategy, application, mode in samples:
            kb = post(
                "/api/knowledge-bases", json={"name": name, "config": {"chunk_strategy": strategy}}
            )
            raw = (root / "examples" / filename).read_bytes()
            upload = post(
                f"/api/knowledge-bases/{kb['id']}/documents",
                files={"file": (filename, raw, "text/markdown")},
            )
            for _ in range(120):
                response = client.get(f"/api/jobs/{upload['job_id']}")
                response.raise_for_status()
                job = response.json()["data"]
                if job["status"] == "done":
                    break
                if job["status"] == "failed":
                    raise RuntimeError(job["error"])
                time.sleep(0.5)
            else:
                raise TimeoutError("入库未完成，请查看 worker 日志")
            post(
                "/api/applications",
                json={"kb_id": kb["id"], "name": application, "strategy": "hybrid", "mode": mode},
            )
            print(f"已建立 {name} → {application} ({mode})")
        print("完成；从前端选择知识库与应用开始提问。样例均为合成数据。")


if __name__ == "__main__":
    main()
