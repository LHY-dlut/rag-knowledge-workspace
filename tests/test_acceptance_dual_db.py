"""Run only on explicitly provisioned empty acceptance databases, never business data."""

import pytest

from tests import test_acceptance_lifecycle as lifecycle
from tests import test_acceptance_regressions as regressions


@pytest.mark.integration
async def test_real_dual_db_claim_and_heartbeat(dual_client, monkeypatch):
    await lifecycle.test_two_ingest_workers_claim_once_and_heartbeat(dual_client, monkeypatch)


@pytest.mark.integration
async def test_real_dual_db_stale_executor_fencing(dual_client, monkeypatch):
    await regressions.test_expired_ingest_executor_cannot_overwrite_new_index(
        dual_client, monkeypatch
    )


@pytest.mark.integration
async def test_real_dual_db_partial_publication(dual_client, monkeypatch):
    await lifecycle.test_business_publication_failure_hides_committed_vectors_and_retry(
        dual_client, monkeypatch
    )
