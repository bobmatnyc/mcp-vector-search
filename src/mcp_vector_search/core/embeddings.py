"""Embedding generation for MCP Vector Search."""

import contextlib
import hashlib
import json
import logging
import multiprocessing
import os
import sys
import warnings
from pathlib import Path

# Suppress verbose transformers/sentence-transformers output at module level
# These messages ("The following layers were not sharded...", progress bars) are noise
# Only INFO level and above from our code should show; transformers gets ERROR only
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# Suppress tqdm progress bars (used by transformers for model loading)
os.environ["TQDM_DISABLE"] = "1"

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*position_ids.*")
warnings.filterwarnings("ignore", message=".*not sharded.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")


@contextlib.contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout and stderr at OS level.

    Used to hide verbose model loading output like "BertModel LOAD REPORT"
    that is printed directly to file descriptors by native code (Rust/C),
    which bypasses Python's sys.stdout/stderr redirection.
    """
    # Save original file descriptors
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()

    # Duplicate original file descriptors
    stdout_dup = os.dup(stdout_fd)
    stderr_dup = os.dup(stderr_fd)

    # Open /dev/null for writing
    devnull = os.open(os.devnull, os.O_RDWR)

    try:
        # Redirect stdout/stderr to /dev/null at OS level
        os.dup2(devnull, stdout_fd)
        os.dup2(devnull, stderr_fd)
        yield
    finally:
        # Restore original file descriptors
        os.dup2(stdout_dup, stdout_fd)
        os.dup2(stderr_dup, stderr_fd)

        # Close duplicates and devnull
        os.close(stdout_dup)
        os.close(stderr_dup)
        os.close(devnull)


# Configure tokenizers parallelism based on process context
# Enable parallelism in main process for 2-4x speedup
# Disable in forked processes to avoid deadlock warnings
# See: https://github.com/huggingface/tokenizers/issues/1294
def _configure_tokenizers_parallelism() -> None:
    """Configure TOKENIZERS_PARALLELISM based on process context."""
    # Check if we're in the main process
    is_main_process = multiprocessing.current_process().name == "MainProcess"

    if is_main_process:
        # Enable parallelism in main process for better performance
        # This gives 2-4x speedup for embedding generation
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
    else:
        # Disable in forked processes to avoid deadlock
        os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Configure before importing sentence_transformers
_configure_tokenizers_parallelism()

import aiofiles
from loguru import logger
from sentence_transformers import SentenceTransformer

from ..config.defaults import get_model_dimensions, is_code_specific_model
from .exceptions import EmbeddingError


class EmbeddingCache:
    """LRU cache for embeddings with disk persistence."""

    def __init__(self, cache_dir: Path, max_size: int = 1000) -> None:
        """Initialize embedding cache.

        Args:
            cache_dir: Directory to store cached embeddings
            max_size: Maximum number of embeddings to keep in memory
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self._memory_cache: dict[str, list[float]] = {}
        self._access_order: list[str] = []  # For LRU eviction
        self._cache_hits = 0
        self._cache_misses = 0

    def _hash_content(self, content: str) -> str:
        """Generate cache key from content."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def get_embedding(self, content: str) -> list[float] | None:
        """Get cached embedding for content."""
        cache_key = self._hash_content(content)

        # Check memory cache first
        if cache_key in self._memory_cache:
            self._cache_hits += 1
            # Move to end for LRU
            self._access_order.remove(cache_key)
            self._access_order.append(cache_key)
            return self._memory_cache[cache_key]

        # Check disk cache
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                async with aiofiles.open(cache_file) as f:
                    content_str = await f.read()
                    embedding = json.loads(content_str)

                    # Add to memory cache with LRU management
                    self._add_to_memory_cache(cache_key, embedding)
                    self._cache_hits += 1
                    return embedding
            except Exception as e:
                logger.warning(f"Failed to load cached embedding: {e}")

        self._cache_misses += 1
        return None

    async def store_embedding(self, content: str, embedding: list[float]) -> None:
        """Store embedding in cache."""
        cache_key = self._hash_content(content)

        # Store in memory cache with LRU management
        self._add_to_memory_cache(cache_key, embedding)

        # Store in disk cache
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            async with aiofiles.open(cache_file, "w") as f:
                await f.write(json.dumps(embedding))
        except Exception as e:
            logger.warning(f"Failed to cache embedding: {e}")

    def _add_to_memory_cache(self, cache_key: str, embedding: list[float]) -> None:
        """Add embedding to memory cache with LRU eviction.

        Args:
            cache_key: Cache key for the embedding
            embedding: Embedding vector to cache
        """
        # If already in cache, update and move to end
        if cache_key in self._memory_cache:
            self._access_order.remove(cache_key)
            self._access_order.append(cache_key)
            self._memory_cache[cache_key] = embedding
            return

        # If cache is full, evict least recently used
        if len(self._memory_cache) >= self.max_size:
            lru_key = self._access_order.pop(0)
            del self._memory_cache[lru_key]

        # Add new embedding
        self._memory_cache[cache_key] = embedding
        self._access_order.append(cache_key)

    def clear_memory_cache(self) -> None:
        """Clear the in-memory cache."""
        self._memory_cache.clear()
        self._access_order.clear()

    def get_cache_stats(self) -> dict[str, any]:
        """Get cache performance statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0
        disk_files = (
            len(list(self.cache_dir.glob("*.json"))) if self.cache_dir.exists() else 0
        )

        return {
            "memory_cache_size": len(self._memory_cache),
            "memory_cached": len(self._memory_cache),  # Alias for compatibility
            "max_cache_size": self.max_size,
            "memory_limit": self.max_size,  # Alias for compatibility
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": round(hit_rate, 3),
            "disk_cache_files": disk_files,
            "disk_cached": disk_files,  # Alias for compatibility
        }


class CodeBERTEmbeddingFunction:
    """ChromaDB-compatible embedding function using CodeBERT."""

    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        timeout: float = 300.0,  # 5 minutes default timeout
    ) -> None:
        """Initialize CodeBERT embedding function.

        Args:
            model_name: Name of the sentence transformer model
            timeout: Timeout in seconds for embedding generation (default: 300s)
        """
        try:
            # Auto-detect CUDA availability for GPU acceleration
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Detect model dimensions and log info
            try:
                expected_dims = get_model_dimensions(model_name)
                is_code_model = is_code_specific_model(model_name)
                model_type = "code-specific" if is_code_model else "general-purpose"
            except ValueError:
                # Unknown model - will be logged as warning
                expected_dims = "unknown"
                model_type = "unknown"

            # Log model download for large models (CodeXEmbed is ~1.5GB)
            if "SFR-Embedding-Code" in model_name or "CodeXEmbed" in model_name:
                logger.info(
                    f"Loading {model_name} (~1.5GB download on first use)... "
                    f"This may take a few minutes."
                )

            # trust_remote_code=True needed for CodeXEmbed and other models with custom code
            # Suppress stdout to hide "BertModel LOAD REPORT" noise
            with suppress_stdout_stderr():
                self.model = SentenceTransformer(
                    model_name, device=device, trust_remote_code=True
                )
            self.model_name = model_name
            self.timeout = timeout

            # Get actual dimensions from loaded model
            actual_dims = self.model.get_sentence_embedding_dimension()

            # Log GPU/CPU usage and model details
            if device == "cuda":
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(
                    f"Loaded {model_type} embedding model: {model_name} "
                    f"on GPU ({gpu_name}) with {actual_dims} dimensions (timeout: {timeout}s)"
                )
            else:
                logger.info(
                    f"Loaded {model_type} embedding model: {model_name} "
                    f"on CPU with {actual_dims} dimensions (timeout: {timeout}s)"
                )

            # Validate dimensions match expected
            if expected_dims != "unknown" and actual_dims != expected_dims:
                logger.warning(
                    f"Model dimension mismatch: expected {expected_dims}, got {actual_dims}. "
                    f"Update MODEL_SPECIFICATIONS in defaults.py"
                )

        except Exception as e:
            logger.error(f"Failed to load embedding model {model_name}: {e}")
            raise EmbeddingError(f"Failed to load embedding model: {e}") from e

    def name(self) -> str:
        """Return embedding function name (ChromaDB requirement)."""
        return f"CodeBERTEmbeddingFunction:{self.model_name}"

    def __call__(self, input: list[str]) -> list[list[float]]:
        """Generate embeddings for input texts (ChromaDB interface)."""
        try:
            # Use ThreadPoolExecutor with timeout for embedding generation
            from concurrent.futures import ThreadPoolExecutor, TimeoutError

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._generate_embeddings, input)
                try:
                    embeddings = future.result(timeout=self.timeout)
                    return embeddings
                except TimeoutError:
                    logger.error(
                        f"Embedding generation timed out after {self.timeout}s for batch of {len(input)} texts"
                    )
                    raise EmbeddingError(
                        f"Embedding generation timed out after {self.timeout}s"
                    )
        except EmbeddingError:
            raise
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise EmbeddingError(f"Failed to generate embeddings: {e}") from e

    def _generate_embeddings(self, input: list[str]) -> list[list[float]]:
        """Internal method to generate embeddings (runs in thread pool)."""
        embeddings = self.model.encode(input, convert_to_numpy=True)
        return embeddings.tolist()


class BatchEmbeddingProcessor:
    """Batch processing for efficient embedding generation with caching."""

    def __init__(
        self,
        embedding_function: CodeBERTEmbeddingFunction,
        cache: EmbeddingCache | None = None,
        batch_size: int = 128,  # Increased from 32 to 128 for better throughput on modern hardware
    ) -> None:
        """Initialize batch embedding processor.

        Args:
            embedding_function: Function to generate embeddings
            cache: Optional embedding cache
            batch_size: Size of batches for processing (default: 128 for modern hardware)
        """
        self.embedding_function = embedding_function
        self.cache = cache
        self.batch_size = batch_size

    async def process_batch(self, contents: list[str]) -> list[list[float]]:
        """Process a batch of content for embeddings.

        Args:
            contents: List of text content to embed

        Returns:
            List of embeddings
        """
        if not contents:
            return []

        embeddings = []
        uncached_contents = []
        uncached_indices = []

        # Check cache for each content if cache is available
        if self.cache:
            for i, content in enumerate(contents):
                cached_embedding = await self.cache.get_embedding(content)
                if cached_embedding:
                    embeddings.append(cached_embedding)
                else:
                    embeddings.append(None)  # Placeholder
                    uncached_contents.append(content)
                    uncached_indices.append(i)
        else:
            # No cache, process all content
            uncached_contents = contents
            uncached_indices = list(range(len(contents)))
            embeddings = [None] * len(contents)

        # Generate embeddings for uncached content
        if uncached_contents:
            logger.debug(f"Generating {len(uncached_contents)} new embeddings")

            try:
                new_embeddings = []
                for i in range(0, len(uncached_contents), self.batch_size):
                    batch = uncached_contents[i : i + self.batch_size]
                    batch_embeddings = self.embedding_function(batch)
                    new_embeddings.extend(batch_embeddings)

                # Cache new embeddings and fill placeholders
                for i, (content, embedding) in enumerate(
                    zip(uncached_contents, new_embeddings, strict=False)
                ):
                    if self.cache:
                        await self.cache.store_embedding(content, embedding)
                    embeddings[uncached_indices[i]] = embedding

            except Exception as e:
                logger.error(f"Failed to generate embeddings: {e}")
                raise EmbeddingError(f"Failed to generate embeddings: {e}") from e

        return embeddings

    def get_stats(self) -> dict[str, any]:
        """Get processor statistics."""
        stats = {
            "model_name": self.embedding_function.model_name,
            "batch_size": self.batch_size,
            "cache_enabled": self.cache is not None,
        }

        if self.cache:
            stats.update(self.cache.get_cache_stats())

        return stats


def create_embedding_function(
    model_name: str = "microsoft/codebert-base",
    cache_dir: Path | None = None,
    cache_size: int = 1000,
):
    """Create embedding function and cache.

    Args:
        model_name: Name of the embedding model
        cache_dir: Directory for caching embeddings
        cache_size: Maximum cache size

    Returns:
        Tuple of (embedding_function, cache)
    """
    try:
        # Use ChromaDB's built-in sentence transformer function
        # Logging suppression is handled at module level
        from chromadb.utils import embedding_functions

        # Map legacy model names to current defaults
        # This ensures backward compatibility with old config files
        model_mapping = {
            # Legacy CodeBERT models (deprecated, never existed in sentence-transformers)
            "microsoft/codebert-base": "Salesforce/SFR-Embedding-Code-400M_R",
            "microsoft/unixcoder-base": "Salesforce/SFR-Embedding-Code-400M_R",
            # Maintain existing mappings for smooth migration
            "codebert": "Salesforce/SFR-Embedding-Code-400M_R",
            "unixcoder": "Salesforce/SFR-Embedding-Code-400M_R",
        }

        actual_model = model_mapping.get(model_name, model_name)

        # Log migration warning if model was remapped
        if actual_model != model_name:
            logger.warning(
                f"Model '{model_name}' is deprecated. Automatically using '{actual_model}' instead. "
                f"Please update your configuration to use the new model explicitly."
            )

        # Suppress stdout to hide "BertModel LOAD REPORT" noise
        with suppress_stdout_stderr():
            embedding_function = (
                embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=actual_model
                )
            )

        logger.debug(f"Created ChromaDB embedding function with model: {actual_model}")

    except Exception as e:
        logger.warning(f"Failed to create ChromaDB embedding function: {e}")
        # Fallback to our custom implementation
        embedding_function = CodeBERTEmbeddingFunction(model_name)

    cache = None
    if cache_dir:
        cache = EmbeddingCache(cache_dir, cache_size)

    return embedding_function, cache
