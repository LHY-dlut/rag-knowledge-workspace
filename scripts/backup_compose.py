"""Quiescent dual-database/upload backup; restore only into empty target stores."""

import argparse
import hashlib
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def docker(*args, output=None, source=None):
    result = subprocess.run(
        ["docker", *args],
        cwd=ROOT,
        stdin=source,
        stdout=output if output else subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode:
        # Database tools may print connection parameters. Don't echo stderr.
        raise RuntimeError("Docker/database operation failed; no credentials logged")
    return result.stdout


def containers(project):
    ids = (
        docker(
            "ps",
            "-a",
            "--filter",
            f"label=com.docker.compose.project={project}",
            "--format",
            "{{.ID}}",
        )
        .decode()
        .split()
    )
    if not ids:
        raise RuntimeError("Compose project not found")
    # Request only labels/state, never a full inspect containing environment secrets.
    services = {}
    for identity in ids:
        value = json.loads(docker("inspect", "--format", "{{json .Config.Labels}}", identity))
        services.setdefault(value["com.docker.compose.service"], []).append(identity)
    if len(services.get("mysql", [])) != 1 or len(services.get("postgres", [])) != 1:
        raise RuntimeError("Expected exactly one MySQL and PostgreSQL container")
    return services


def require_quiescent(services):
    for service in ("api", "worker", "frontend"):
        for identity in services.get(service, []):
            if docker("inspect", "--format", "{{.State.Running}}", identity).strip() != b"false":
                raise RuntimeError("Stop frontend/API/workers gracefully before backup or restore")


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def uploads_snapshot(directory):
    return {
        p.relative_to(directory).as_posix(): sha(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }


def backup(args, services):
    require_quiescent(services)
    args.directory.mkdir(parents=True, exist_ok=False)
    mysql, postgres = services["mysql"][0], services["postgres"][0]
    with (args.directory / "business.sql").open("wb") as file:
        docker(
            "exec",
            mysql,
            "sh",
            "-c",
            'MYSQL_PWD="$MYSQL_PASSWORD" mysqldump -u rag --single-transaction --hex-blob --no-tablespaces rag_business',
            output=file,
        )
    with (args.directory / "vectors.dump").open("wb") as file:
        docker(
            "exec",
            postgres,
            "pg_dump",
            "-U",
            "rag",
            "-d",
            "rag_vectors",
            "--format=custom",
            "--no-owner",
            "--no-acl",
            output=file,
        )
    worker = services["worker"][0]
    docker("cp", f"{worker}:/app/data/uploads", str(args.directory / "uploads"))
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "project": args.project,
        "quiescent": True,
        "files": {name: sha(args.directory / name) for name in ("business.sql", "vectors.dump")},
        "uploads": uploads_snapshot(args.directory / "uploads"),
        "contains_private_data": True,
    }
    (args.directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": "backup_created", "upload_files": len(manifest["uploads"])}))


def restore(args, services):
    require_quiescent(services)
    manifest = json.loads((args.directory / "manifest.json").read_text(encoding="utf-8"))
    for name in ("business.sql", "vectors.dump"):
        if sha(args.directory / name) != manifest["files"][name]:
            raise RuntimeError("Backup checksum mismatch")
    if uploads_snapshot(args.directory / "uploads") != manifest["uploads"]:
        raise RuntimeError("Uploaded file checksum mismatch")
    mysql, postgres = services["mysql"][0], services["postgres"][0]
    business_count = docker(
        "exec",
        mysql,
        "sh",
        "-c",
        'MYSQL_PWD="$MYSQL_PASSWORD" mysql -u rag -N -e "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema=\'rag_business\'"',
    ).strip()
    vector_count = docker(
        "exec",
        postgres,
        "psql",
        "-U",
        "rag",
        "-d",
        "rag_vectors",
        "-At",
        "-c",
        "SELECT count(*) FROM information_schema.tables WHERE table_schema='public';",
    ).strip()
    if business_count != b"0" or vector_count != b"0":
        raise RuntimeError("Restore target must be empty; existing data was not changed")
    worker = services.get("worker", services.get("api", []))
    if not worker:
        raise RuntimeError("Create a stopped API/worker container to receive uploads first")
    # Verify target uploads through docker cp, also works with stopped containers.
    import tempfile

    with tempfile.TemporaryDirectory(prefix="rag-restore-inspect-") as scratch:
        target = Path(scratch) / "uploads"
        docker("cp", f"{worker[0]}:/app/data/uploads", str(target))
        if uploads_snapshot(target):
            raise RuntimeError("Restore target upload directory must be empty")
    with (args.directory / "business.sql").open("rb") as file:
        docker(
            "exec",
            "-i",
            mysql,
            "sh",
            "-c",
            'MYSQL_PWD="$MYSQL_PASSWORD" mysql -u rag rag_business',
            source=file,
        )
    with (args.directory / "vectors.dump").open("rb") as file:
        docker(
            "exec",
            "-i",
            postgres,
            "pg_restore",
            "-U",
            "rag",
            "-d",
            "rag_vectors",
            "--exit-on-error",
            "--no-owner",
            "--no-acl",
            source=file,
        )
    docker("cp", str(args.directory / "uploads") + "/.", f"{worker[0]}:/app/data/uploads")
    # Restore Linux ownership without making the runtime application root.
    api = services.get("api", [worker[0]])[0]
    docker("start", api)
    docker("exec", "-u", "root", api, "chown", "-R", "rag:rag", "/app/data/uploads")
    print(
        json.dumps(
            {
                "status": "restored",
                "follow_up": "verify migrations, counts, vectors and HTTP; start workers/frontend",
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["backup", "restore-empty"])
    parser.add_argument("--project", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z][a-z0-9_-]{1,62}", args.project):
        parser.error("Invalid explicit Compose project name")
    args.directory = args.directory.resolve()
    if args.directory.is_relative_to(ROOT):
        parser.error("Keep private backups outside the source directory")
    services = containers(args.project)
    (backup if args.action == "backup" else restore)(args, services)
