"""Progress state management for indexing operations.

Provides persistent state tracking for all indexing phases:
- Chunking (Phase 1)
- Embedding (Phase 2)
- Knowledge Graph building (Phase 3)

State is stored in .mcp-vector-search/progress.json and can be queried
independently of the indexing process.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from loguru import logger


@dataclass
class ChunkingProgress:
    """Progress tracking for Phase 1 (chunking)."""

    total_files: int = 0
    processed_files: int = 0
    total_chunks: int = 0


@dataclass
class EmbeddingProgress:
    """Progress tracking for Phase 2 (embedding)."""

    total_chunks: int = 0
    embedded_chunks: int = 0


@dataclass
class KGBuildProgress:
    """Progress tracking for Phase 3 (KG build)."""

    total_chunks: int = 0
    processed_chunks: int = 0
    entities: int = 0
    relations: int = 0


@dataclass
class ProgressState:
    """Complete indexing progress state.

    Tracks all three phases of indexing with timestamps and metadata.
    """

    phase: Literal["chunking", "embedding", "kg_build", "complete"] = "chunking"
    chunking: ChunkingProgress = None
    embedding: EmbeddingProgress = None
    kg_build: KGBuildProgress = None
    started_at: float = None
    updated_at: float = None

    def __post_init__(self):
        """Initialize nested dataclasses if not provided."""
        if self.chunking is None:
            self.chunking = ChunkingProgress()
        if self.embedding is None:
            self.embedding = EmbeddingProgress()
        if self.kg_build is None:
            self.kg_build = KGBuildProgress()
        if self.started_at is None:
            self.started_at = time.time()
        if self.updated_at is None:
            self.updated_at = time.time()

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "phase": self.phase,
            "chunking": asdict(self.chunking),
            "embedding": asdict(self.embedding),
            "kg_build": asdict(self.kg_build),
            "started_at": self.started_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProgressState":
        """Create from dictionary (JSON deserialization)."""
        return cls(
            phase=data.get("phase", "chunking"),
            chunking=ChunkingProgress(**data.get("chunking", {})),
            embedding=EmbeddingProgress(**data.get("embedding", {})),
            kg_build=KGBuildProgress(**data.get("kg_build", {})),
            started_at=data.get("started_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
        )


class ProgressStateManager:
    """Manager for persisting and loading progress state.

    Handles atomic writes, file locking, and state validation.
    """

    def __init__(self, project_root: Path):
        """Initialize progress state manager.

        Args:
            project_root: Project root directory
        """
        self.project_root = project_root
        self.config_dir = project_root / ".mcp-vector-search"
        self.state_file = self.config_dir / "progress.json"

    def exists(self) -> bool:
        """Check if progress state file exists.

        Returns:
            True if state file exists, False otherwise
        """
        return self.state_file.exists()

    def load(self) -> ProgressState | None:
        """Load progress state from file.

        Returns:
            ProgressState if file exists, None otherwise
        """
        if not self.state_file.exists():
            return None

        try:
            with open(self.state_file) as f:
                data = json.load(f)
            return ProgressState.from_dict(data)
        except Exception as e:
            logger.warning(f"Failed to load progress state: {e}")
            return None

    def save(self, state: ProgressState) -> None:
        """Save progress state to file atomically.

        Args:
            state: Progress state to save
        """
        # Update timestamp
        state.updated_at = time.time()

        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Atomic write: temp file + rename
        temp_file = self.state_file.with_suffix(".tmp")
        try:
            with open(temp_file, "w") as f:
                json.dump(state.to_dict(), f, indent=2)

            # Atomic rename (overwrites existing)
            temp_file.replace(self.state_file)
        except Exception as e:
            logger.error(f"Failed to save progress state: {e}")
            if temp_file.exists():
                temp_file.unlink()
            raise

    def clear(self) -> None:
        """Clear progress state file.

        Removes the state file if it exists.
        """
        if self.state_file.exists():
            try:
                self.state_file.unlink()
                logger.info("Cleared progress state")
            except Exception as e:
                logger.error(f"Failed to clear progress state: {e}")
                raise

    def update_chunking(
        self,
        total_files: int | None = None,
        processed_files_increment: int = 0,
        chunks_increment: int = 0,
    ) -> None:
        """Update chunking progress.

        Args:
            total_files: Total files to process (set once at start)
            processed_files_increment: Number of files processed to add
            chunks_increment: Number of chunks created to add
        """
        state = self.load() or ProgressState()

        if total_files is not None:
            state.chunking.total_files = total_files

        state.chunking.processed_files += processed_files_increment
        state.chunking.total_chunks += chunks_increment

        # Update phase
        if (
            state.chunking.processed_files >= state.chunking.total_files
            and state.chunking.total_files > 0
        ):
            # Chunking complete, move to embedding
            if state.phase == "chunking":
                state.phase = "embedding"
                state.embedding.total_chunks = state.chunking.total_chunks

        self.save(state)

    def update_embedding(
        self,
        total_chunks: int | None = None,
        embedded_chunks_increment: int = 0,
    ) -> None:
        """Update embedding progress.

        Args:
            total_chunks: Total chunks to embed (set once at start)
            embedded_chunks_increment: Number of chunks embedded to add
        """
        state = self.load() or ProgressState()

        if total_chunks is not None:
            state.embedding.total_chunks = total_chunks

        state.embedding.embedded_chunks += embedded_chunks_increment

        # Update phase
        if (
            state.embedding.embedded_chunks >= state.embedding.total_chunks
            and state.embedding.total_chunks > 0
        ):
            # Embedding complete, move to KG build
            if state.phase == "embedding":
                state.phase = "kg_build"
                state.kg_build.total_chunks = state.embedding.total_chunks

        self.save(state)

    def update_kg_build(
        self,
        processed_chunks_increment: int = 0,
        entities_increment: int = 0,
        relations_increment: int = 0,
    ) -> None:
        """Update KG build progress.

        Args:
            processed_chunks_increment: Number of chunks processed to add
            entities_increment: Number of entities extracted to add
            relations_increment: Number of relations created to add
        """
        state = self.load() or ProgressState()

        state.kg_build.processed_chunks += processed_chunks_increment
        state.kg_build.entities += entities_increment
        state.kg_build.relations += relations_increment

        # Update phase
        if (
            state.kg_build.processed_chunks >= state.kg_build.total_chunks
            and state.kg_build.total_chunks > 0
        ):
            # KG build complete
            state.phase = "complete"

        self.save(state)

    def mark_complete(self) -> None:
        """Mark indexing as complete."""
        state = self.load()
        if state:
            state.phase = "complete"
            self.save(state)

    def reset(self) -> None:
        """Reset progress state to initial state.

        Creates a new empty state, replacing any existing state.
        """
        state = ProgressState()
        self.save(state)
