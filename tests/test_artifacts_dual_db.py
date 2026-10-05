"""The same artifact publication guards exercised on real independent databases."""

import pytest

from tests import test_artifact_storage as storage
from tests import test_artifacts as artifacts


@pytest.mark.integration
async def test_dual_artifact_publication_and_owner_isolation(dual_client):
    await artifacts.test_artifact_is_durable_before_terminal_and_owned_after_refresh(dual_client)


@pytest.mark.integration
async def test_dual_artifact_conversion_and_stale_sources(dual_client):
    await artifacts.test_conversion_is_idempotent_and_stale_version_cannot_create_new_artifact(
        dual_client
    )


@pytest.mark.integration
async def test_dual_artifact_expired_publication_fenced(dual_client, monkeypatch):
    await artifacts.test_lease_expiring_during_artifact_io_cannot_publish(dual_client, monkeypatch)


@pytest.mark.integration
async def test_dual_cancel_during_generation_blocks_artifact(dual_client, monkeypatch):
    await artifacts.test_cancel_during_generation_cannot_publish_artifact(dual_client, monkeypatch)


@pytest.mark.integration
async def test_dual_rollback_cleans_new_file_preserving_old_artifact(dual_client):
    await storage.test_outer_rollback_removes_only_new_file_preserving_committed_artifact(
        dual_client
    )


@pytest.mark.integration
async def test_dual_orphan_quarantine_preserves_live_files(dual_client):
    await storage.test_cleanup_dry_run_and_reversible_quarantine_preserve_live_recent_unknown(
        dual_client
    )


@pytest.mark.integration
async def test_dual_cancelled_writer_cleanup(dual_client):
    await storage.test_cancelled_writer_settles_before_rollback_and_lock_release(dual_client)
