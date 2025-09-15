# -------------------- wisdom_collector.py (Arquivo Novo) --------------------
import asyncio
import logging
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from database.models import AgentRepository

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configurações ---
SHARED_STORAGE_PATH = Path("agent_storage/_shared")
LIVE_MEMORY_STORE_PATH = SHARED_STORAGE_PATH / "live_memory_store"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class WisdomCollector:
    def __init__(self):
        self.db_repo = AgentRepository()
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vector_store = self._load_vector_store()

    def _load_vector_store(self):
        SHARED_STORAGE_PATH.mkdir(exist_ok=True, parents=True)
        if LIVE_MEMORY_STORE_PATH.exists():
            logger.info("Collector: Loading existing Live Memory vector store...")
            return FAISS.load_local(str(LIVE_MEMORY_STORE_PATH), self.embeddings, allow_dangerous_deserialization=True)
        else:
            logger.info("Collector: Creating new Live Memory vector store.")
            # Criar um índice vazio para começar
            dummy_texts = ["initialization vector"]
            dummy_metadatas = [{"source_id": "init"}]
            db = FAISS.from_texts(dummy_texts, self.embeddings, metadatas=dummy_metadatas)
            db.save_local(str(LIVE_MEMORY_STORE_PATH))
            return db

    async def collect_and_index_wisdom(self):
        logger.info("--- [STARTING LIVE MEMORY COLLECTION CYCLE] ---")

        new_memories = self.db_repo.get_unindexed_live_memories()

        if not new_memories:
            logger.info("Collector: No new Live Memories to index. Cycle complete.")
            return

        logger.info(f"Collector: Found {len(new_memories)} new Live Memories to index.")

        texts_to_index = [mem.content for mem in new_memories]
        metadatas = [{"source_id": mem.id, "memory_type": mem.memory_type, "created_at": mem.created_at.isoformat()} for
                     mem in new_memories]

        if self.vector_store:
            self.vector_store.add_texts(texts=texts_to_index, metadatas=metadatas)
        else:  # Recria se não existir
            self.vector_store = FAISS.from_texts(texts=texts_to_index, embedding=self.embeddings, metadatas=metadatas)

        self.vector_store.save_local(str(LIVE_MEMORY_STORE_PATH))
        self.db_repo.mark_live_memories_as_indexed([mem.id for mem in new_memories])

        logger.info(f"Collector: Successfully indexed {len(new_memories)} memories. Vector store updated.")
        logger.info("--- [LIVE MEMORY COLLECTION CYCLE COMPLETE] ---")


async def run_collector_periodically(interval_minutes: int):
    collector = WisdomCollector()
    while True:
        try:
            await collector.collect_and_index_wisdom()
        except Exception as e:
            logger.error(f"Collector: Error during collection cycle: {e}", exc_info=True)

        logger.info(f"Collector: Sleeping for {interval_minutes} minutes.")
        await asyncio.sleep(interval_minutes * 60)