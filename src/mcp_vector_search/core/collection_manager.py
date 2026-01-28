"""Collection management for ChromaDB vector database."""

from pathlib import Path
from typing import Any

from loguru import logger


class CollectionManager:
    """Manages ChromaDB collection lifecycle and operations.

    Handles collection creation, configuration, and access with
    optimized HNSW parameters for code search.
    """

    def __init__(self, collection_name: str = "code_search") -> None:
        """Initialize collection manager.

        Args:
            collection_name: Name of the collection to manage
        """
        self.collection_name = collection_name

    def get_or_create_collection(self, client: Any, embedding_function: Any) -> Any:
        """Get or create collection with optimized HNSW parameters.

        HNSW parameters are tuned for code search:
        - M=32: More connections per node (default: 16) for better recall
        - ef_construction=400: Higher construction quality (default: 200)
        - ef_search=75: Better search recall (default: 10)

        Args:
            client: ChromaDB client instance
            embedding_function: Embedding function for the collection

        Returns:
            ChromaDB collection instance
        """
        collection = client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=embedding_function,
            metadata={
                "description": "Semantic code search collection",
                "hnsw:space": "cosine",  # Cosine similarity for semantic search
                "hnsw:construction_ef": 400,  # Higher construction quality
                "hnsw:M": 32,  # More connections per node
                "hnsw:search_ef": 75,  # Better search recall
            },
        )

        logger.debug(
            f"Collection '{self.collection_name}' ready with optimized HNSW parameters"
        )
        return collection

    def configure_sqlite_timeout(self, persist_directory: Path) -> None:
        """Configure SQLite busy_timeout to prevent indefinite waits.

        Sets a 30-second timeout on database lock waits to prevent
        hanging on locked databases.

        Args:
            persist_directory: Path to ChromaDB persistence directory
        """
        try:
            import sqlite3

            chroma_db_path = persist_directory / "chroma.sqlite3"
            if chroma_db_path.exists():
                temp_conn = sqlite3.connect(str(chroma_db_path))
                temp_conn.execute("PRAGMA busy_timeout = 30000")  # 30 seconds
                temp_conn.close()
                logger.debug("Configured SQLite busy_timeout=30000ms")
        except Exception as e:
            logger.warning(f"Failed to configure SQLite busy_timeout: {e}")

    def reset_collection(self, client: Any) -> None:
        """Reset the collection by deleting all data.

        Args:
            client: ChromaDB client instance
        """
        try:
            client.reset()
            logger.info(f"Collection '{self.collection_name}' reset successfully")
        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
            raise
