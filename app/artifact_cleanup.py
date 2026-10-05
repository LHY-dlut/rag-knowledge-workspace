"""Dry-run by default. Quarantine only old files with no committed artifact row."""

import argparse
import asyncio
import json
import time
from datetime import UTC, datetime
from uuid import UUID, uuid4

from app.artifact_storage import artifact_directory, publication_lock
from app.database import close_database, init_database
from app.models import GeneratedArtifact
from app.settings import Settings


async def quarantine_orphans(settings, *, apply=False, min_age_seconds=86400, limit=100):
    if min_age_seconds < 0 or not 1 <= limit <= 1000:
        raise ValueError("Invalid maintenance limits")
    report = {
        "time": datetime.now(UTC).isoformat(),
        "mode": "quarantine" if apply else "dry_run",
        "min_age_seconds": min_age_seconds,
        "limit": limit,
        "valid_files_preserved": 0,
        "unrecognized_or_recent_preserved": 0,
        "orphans": [],
    }
    # API/worker holds this same lock through transaction commit. Without it,
    # an uncommitted row could be mistaken for an orphan and lose its file.
    async with publication_lock(settings):
        base = artifact_directory(settings)
        candidates = []
        now = time.time()
        for path in sorted(base.iterdir()):
            if path.name.startswith("."):
                continue
            try:
                canonical = str(UUID(path.stem))
                recognized = path.name == canonical + path.suffix and path.suffix in {
                    ".json",
                    ".html",
                }
            except ValueError:
                recognized = False
            if (
                not recognized
                or path.is_symlink()
                or not path.is_file()
                or now - path.stat().st_mtime < min_age_seconds
            ):
                report["unrecognized_or_recent_preserved"] += 1
                continue
            candidates.append(path)
        # Fetch in bounded batches; no mutation takes place before all lookups
        # succeed. A DB outage must not trigger a destructive guess.
        live = set()
        for offset in range(0, len(candidates), 200):
            live.update(
                await GeneratedArtifact.filter(
                    id__in=[p.stem for p in candidates[offset : offset + 200]]
                ).values_list("id", flat=True)
            )
        orphans = []
        for path in candidates:
            if path.stem in live:
                report["valid_files_preserved"] += 1
            else:
                orphans.append(path)
        report["total_orphans_found"] = len(orphans)
        destination = (
            base
            / ".quarantine"
            / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex)
        )
        for path in orphans[:limit]:
            item = {"filename": path.name, "action": "would_quarantine"}
            if apply:
                # Direct children only, same-volume rename, no recursive delete.
                destination.mkdir(parents=True, exist_ok=True)
                target = destination / path.name
                path.rename(target)
                item.update(action="quarantined", destination=target.relative_to(base).as_posix())
            report["orphans"].append(item)
    return report


async def run(args):
    settings = Settings()
    await init_database(settings)
    try:
        return await quarantine_orphans(
            settings,
            apply=args.apply,
            min_age_seconds=args.min_age_hours * 3600,
            limit=args.limit,
        )
    finally:
        await close_database()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Move orphans to quarantine")
    parser.add_argument("--min-age-hours", type=float, default=24)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
