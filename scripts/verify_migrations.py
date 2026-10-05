"""Exercise both checked-in migration chains on fresh temporary SQLite files."""

import json
import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

from verification_runtime import isolated_runtime


def main():
    root = Path(__file__).resolve().parent.parent
    with isolated_runtime(prefix="rag-migrations-") as directory:
        temporary = Path(directory)
        env = {
            **os.environ,
            "APP_MODE": "demo",
            "MODEL_PROVIDER": "demo",
            "BUSINESS_DB_URL": f"sqlite://{temporary}/business.sqlite3",
            "VECTOR_DB_URL": f"sqlite://{temporary}/vectors.sqlite3",
        }
        # Applying twice verifies that version tracking prevents duplicate DDL.
        for _ in range(2):
            subprocess.run([sys.executable, "-m", "app.bootstrap"], cwd=root, env=env, check=True)
        with closing(sqlite3.connect(temporary / "business.sqlite3")) as db:
            tables = {
                row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            columns = {row[1] for row in db.execute("PRAGMA table_info(conversation)")}
            assert {"application", "feedback", "evaluationdataset", "agentrun"} <= tables
            assert "application_id" in columns
            columns = {row[1] for row in db.execute("PRAGMA table_info(document)")}
            assert "index_revision" in columns
            assert "runevent" in tables
            columns = {row[1] for row in db.execute("PRAGMA table_info(agentrun)")}
            assert {"lease_owner", "lease_until", "event_sequence", "request_snapshot"} <= columns
        with closing(sqlite3.connect(temporary / "vectors.sqlite3")) as db:
            columns = {row[1] for row in db.execute("PRAGMA table_info(chunk_vector)")}
            assert {"embedding", "parent_id", "owner_id"} <= columns
    result = {
        "passed": True,
        "dialect": "SQLite migration execution + separately exported MySQL/PostgreSQL DDL",
        "business_migrations": [
            "0001_initial",
            "0002_applications_feedback_datasets",
            "0003_document_index_revision",
            "0004_durable_chat",
        ],
        "vector_migrations": ["0001_initial"],
        "idempotent_apply": True,
        "real_mysql_postgresql_verified": False,
    }
    Path(
        os.environ.get("MIGRATION_REPORT", str(root / "artifacts" / "migration_verification.json"))
    ).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
