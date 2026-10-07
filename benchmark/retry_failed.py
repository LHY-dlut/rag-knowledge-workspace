"""重试知识库中入库失败的文档（如模型日上限、瞬时故障导致）。

用法：
  python retry_failed.py --api-base http://127.0.0.1:18790 \
      --username bench_admin --password <pw> --kb-id <kb_id>
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_crud_benchmark import Api, kb_documents, log  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--kb-id", required=True)
    parser.add_argument("--max-wait", type=int, default=3600)
    args = parser.parse_args()

    api = Api(args.api_base)
    token = api.req(
        "POST", "/api/auth/login", {"username": args.username, "password": args.password}
    )["data"]["access_token"]
    api = Api(args.api_base, token)

    docs = kb_documents(api, args.kb_id)
    failed = [d for d in docs if d["status"] == "failed"]
    log(f"知识库文档 {len(docs)} 份，失败 {len(failed)} 份")
    if not failed:
        log("无需重试")
        return
    for f in failed[:3]:
        log(f"  失败示例: {f['filename'][:40]} | {(f.get('error') or '')[:60]}")

    ok = 0
    for d in failed:
        try:
            api.req("POST", f"/api/documents/{d['id']}/reindex")
            ok += 1
        except RuntimeError as exc:
            log(f"  重试请求失败 {d['filename'][:30]}: {exc}")
    log(f"已提交重试 {ok}/{len(failed)} 份，等待入库 …")

    deadline = time.time() + args.max_wait
    while time.time() < deadline:
        time.sleep(15)
        docs = kb_documents(api, args.kb_id)
        pending = [d for d in docs if d["status"] in ("queued", "processing")]
        still_failed = [d for d in docs if d["status"] == "failed"]
        ready = [d for d in docs if d["status"] == "ready"]
        log(f"  ready={len(ready)} pending={len(pending)} failed={len(still_failed)}")
        if not pending:
            break
    if still_failed:
        log(f"仍有 {len(still_failed)} 份失败：")
        for f in still_failed[:10]:
            log(f"  {f['filename'][:40]} | {(f.get('error') or '')[:80]}")
    else:
        log(f"全部就绪：{len(ready)} 份 ready")


if __name__ == "__main__":
    main()
