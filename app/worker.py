import asyncio
import logging

from app.agent import RAGAgent
from app.chat_jobs import ChatWorker
from app.database import close_database, init_database
from app.ingestion import IngestionService, IngestionWorker
from app.operations import shutdown_signals, supervise_workers, worker_health_loop
from app.providers import make_provider
from app.retrieval import AdvancedRetrieverPipeline
from app.settings import Settings
from app.vector_store import VectorStore


async def main():
    logging.basicConfig(level=logging.INFO)
    settings = Settings()
    await init_database(settings, create_schema=settings.app_mode == "demo")
    provider = make_provider(settings)
    worker = IngestionWorker(IngestionService(VectorStore(settings), provider, settings), settings)
    chat_worker = ChatWorker(
        RAGAgent(provider, AdvancedRetrieverPipeline(VectorStore(settings), provider, settings)),
        settings,
    )
    stopping = asyncio.Event()
    health = asyncio.create_task(worker_health_loop(settings.worker_health_file, stopping))
    try:
        with shutdown_signals(stopping):
            await supervise_workers(
                (worker, chat_worker), stopping, settings.worker_shutdown_seconds
            )
    finally:
        stopping.set()
        try:
            await health
        finally:
            try:
                await provider.close()
            finally:
                await close_database()


if __name__ == "__main__":
    asyncio.run(main())
