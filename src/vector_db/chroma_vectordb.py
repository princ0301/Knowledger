import logging
from typing import List, Dict, Any, Optional
import json
from pathlib import Path

import chromadb
from chromadb.config import Settings

from src.embeddings.embedding_generator import EmbeddedChunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChromaVectorDB:
    """
    Full replacement of MilvusVectorDB using ChromaDB
    NotebookLM-style vector store
    Windows compatible
    Persistent
    """

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "notebook_lm",
        embedding_dim: int = 384
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim

        self.client = None
        self.collection = None

        self._initialize_client()
        self._setup_collection()

    # ------------------------------------------------------------------
    # INIT
    # ------------------------------------------------------------------

    def _initialize_client(self):
        try:
            self.client = chromadb.Client(
                Settings(
                    persist_directory=self.persist_dir,
                    anonymized_telemetry=False
                )
            )
            logger.info(f"ChromaDB initialized at: {self.persist_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    def _setup_collection(self):
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Collection '{self.collection_name}' ready")
        except Exception as e:
            logger.error(f"Failed to setup collection: {e}")
            raise

    # ------------------------------------------------------------------
    # INSERT
    # ------------------------------------------------------------------

    def insert_embeddings(self, embedded_chunks: List[EmbeddedChunk]) -> List[str]:
        if not embedded_chunks:
            return []

        ids = []
        documents = []
        embeddings = []
        metadatas = []

        for chunk in embedded_chunks:
            data = chunk.to_vector_db_format()

            ids.append(data["id"])
            documents.append(data["content"])
            embeddings.append(data["vector"])

            metadata = {
                "source_file": data.get("source_file"),
                "source_type": data.get("source_type"),
                "page_number": data.get("page_number", -1),
                "chunk_index": data.get("chunk_index"),
                "start_char": data.get("start_char", -1),
                "end_char": data.get("end_char", -1),
                "embedding_model": data.get("embedding_model"),
                "metadata": json.dumps(data.get("metadata", {}))
            }

            metadatas.append(metadata)

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

        self.client.persist()

        logger.info(f"Inserted {len(ids)} chunks into ChromaDB")
        return ids

    # ------------------------------------------------------------------
    # SEARCH
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=limit,
            where=filter_metadata
        )

        formatted_results = []

        if not results or not results.get("ids"):
            return []

        for i in range(len(results["ids"][0])):
            metadata = results["metadatas"][0][i]

            raw_meta = metadata.get("metadata", "{}")
            try:
                raw_meta = json.loads(raw_meta)
            except:
                raw_meta = {}

            formatted_results.append({
                "id": results["ids"][0][i],
                "score": results["distances"][0][i],
                "content": results["documents"][0][i],
                "citation": {
                    "source_file": metadata.get("source_file"),
                    "source_type": metadata.get("source_type"),
                    "page_number": (
                        metadata.get("page_number")
                        if metadata.get("page_number", -1) != -1
                        else None
                    ),
                    "chunk_index": metadata.get("chunk_index"),
                    "start_char": (
                        metadata.get("start_char")
                        if metadata.get("start_char", -1) != -1
                        else None
                    ),
                    "end_char": (
                        metadata.get("end_char")
                        if metadata.get("end_char", -1) != -1
                        else None
                    ),
                },
                "metadata": raw_meta,
                "embedding_model": metadata.get("embedding_model"),
            })

        logger.info(f"Search returned {len(formatted_results)} results")
        return formatted_results

    # ------------------------------------------------------------------
    # GET CHUNK BY ID
    # ------------------------------------------------------------------

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        try:
            results = self.collection.get(ids=[chunk_id])

            if not results["ids"]:
                return None

            metadata = results["metadatas"][0]

            raw_meta = metadata.get("metadata", "{}")
            try:
                raw_meta = json.loads(raw_meta)
            except:
                raw_meta = {}

            return {
                "id": results["ids"][0],
                "content": results["documents"][0],
                "metadata": raw_meta,
                "source_file": metadata.get("source_file"),
                "source_type": metadata.get("source_type"),
                "page_number": metadata.get("page_number"),
                "chunk_index": metadata.get("chunk_index"),
            }

        except Exception as e:
            logger.error(f"Get chunk failed: {e}")
            return None

    # ------------------------------------------------------------------
    # DELETE
    # ------------------------------------------------------------------

    def delete_collection(self):
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' deleted")
        except Exception as e:
            logger.error(f"Delete failed: {e}")

    # ------------------------------------------------------------------
    # CLOSE
    # ------------------------------------------------------------------

    def close(self):
        try:
            self.client.persist()
            logger.info("ChromaDB persisted")
        except Exception:
            pass


# ----------------------------------------------------------------------
# TEST
# ----------------------------------------------------------------------

if __name__ == "__main__":
    from src.doc.doc_processor import DocumentProcessor
    from src.embeddings.embedding_generator import EmbeddingGenerator

    doc_processor = DocumentProcessor()
    embedding_generator = EmbeddingGenerator()
    vector_db = ChromaVectorDB()

    try:
        chunks = doc_processor.process_document(
            "D:/Project_new/notebook_lm/data/attn_is_all_you_need.pdf"
        )

        embedded_chunks = embedding_generator.generate_embeddings(chunks)

        ids = vector_db.insert_embeddings(embedded_chunks)
        print(f"Inserted {len(ids)} chunks")

        query = "What is the main idea of the paper?"
        query_vector = embedding_generator.generate_query_embedding(query)

        results = vector_db.search(query_vector.tolist(), limit=5)

        for i, r in enumerate(results, 1):
            print(f"\nResult {i}")
            print("Score:", r["score"])
            print("Text:", r["content"][:200])
            print("Citation:", r["citation"])

    finally:
        vector_db.close()
