"""Embedding generation for MCP Vector Search."""

import contextlib
import hashlib
import logging
import multiprocessing
import os
import sys
import warnings
from pathlib import Path

import orjson

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
# Suppress Pydantic warnings from lancedb embeddings
warnings.filterwarnings("ignore", message=".*has conflict with protected namespace.*")


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


def _detect_device() -> str:
    """Detect optimal compute device (MPS > CUDA > CPU).

    Returns:
        Device string: "mps", "cuda", or "cpu"

    Environment Variables:
        MCP_VECTOR_SEARCH_DEVICE: Override device selection ("cpu", "cuda", or "mps")
    """
    import torch

    # Check environment variable override first
    env_device = os.environ.get("MCP_VECTOR_SEARCH_DEVICE", "").lower()
    if env_device in ("cpu", "cuda", "mps"):
        logger.info(f"Using device from environment override: {env_device}")
        return env_device

    # Apple Silicon MPS provides significant speedup for models >50M params on Apple Silicon
    # PyTorch 2.10.0 has a known MPS regression, so we fall back to CPU for that version
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        # Check for PyTorch 2.10.0 regression
        if torch.__version__.startswith("2.10.0"):
            logger.warning(
                "PyTorch 2.10.0 detected — falling back to CPU due to known MPS regression. "
                "Upgrade PyTorch to restore GPU acceleration."
            )
            return "cpu"

        logger.info("Apple Silicon detected. Using MPS for GPU-accelerated inference.")
        return "mps"

    # Check for NVIDIA CUDA with detailed diagnostics
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "unknown"
        logger.info(
            f"Using CUDA backend for GPU acceleration ({gpu_count} GPU(s): {gpu_name})"
        )
        return "cuda"

    # Log why CUDA isn't available (helps debug AWS/cloud issues)
    cuda_built = (
        torch.backends.cuda.is_built() if hasattr(torch.backends, "cuda") else False
    )
    if not cuda_built:
        logger.debug(
            "CUDA not available: PyTorch installed without CUDA support. "
            "Install with: pip install torch --index-url https://download.pytorch.org/whl/cu121"
        )
    else:
        logger.debug(
            "CUDA not available: PyTorch has CUDA support but no GPU detected. "
            "Check: nvidia-smi, NVIDIA drivers, and GPU instance type."
        )

    logger.info("Using CPU backend (no GPU acceleration)")
    return "cpu"


def _detect_optimal_batch_size() -> int:
    """Detect optimal batch size based on device and memory.

    Returns:
        Optimal batch size for embedding generation:
        - MPS (Apple Silicon):
          - 512 for M4 Max/Ultra with 64GB+ RAM
          - 384 for M4 Pro with 32GB+ RAM
          - 256 for M4 with 16GB+ RAM
        - CUDA (NVIDIA):
          - 512 for GPUs with 8GB+ VRAM (RTX 3070+, A100, etc.)
          - 256 for GPUs with 4-8GB VRAM (RTX 3060, etc.)
          - 128 for GPUs with <4GB VRAM
        - CPU: 128

    Environment Variables:
        MCP_VECTOR_SEARCH_BATCH_SIZE: Override auto-detection
    """
    import torch

    # Check environment override first
    env_batch_size = os.environ.get("MCP_VECTOR_SEARCH_BATCH_SIZE")
    if env_batch_size:
        try:
            return int(env_batch_size)
        except ValueError:
            logger.warning(
                f"Invalid MCP_VECTOR_SEARCH_BATCH_SIZE value: {env_batch_size}, using auto-detection"
            )

    # Check for Apple Silicon with unified memory
    if torch.backends.mps.is_available():
        try:
            import subprocess

            result = subprocess.run(  # nosec B607 - safe read-only sysctl
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                check=False,
            )
            total_ram_gb = int(result.stdout.strip()) / (1024**3)

            if total_ram_gb >= 64:
                batch_size = 512
                logger.info(
                    f"Apple Silicon detected ({total_ram_gb:.1f}GB RAM): using batch size {batch_size} (M4 Max/Ultra optimized)"
                )
                return batch_size
            elif total_ram_gb >= 32:
                batch_size = 384
                logger.info(
                    f"Apple Silicon detected ({total_ram_gb:.1f}GB RAM): using batch size {batch_size} (M4 Pro optimized)"
                )
                return batch_size
            else:
                batch_size = 256
                logger.info(
                    f"Apple Silicon detected ({total_ram_gb:.1f}GB RAM): using batch size {batch_size}"
                )
                return batch_size
        except Exception as e:
            logger.warning(
                f"Apple Silicon RAM detection failed: {e}, using default batch size 256"
            )
            return 256

    # Auto-detect based on CUDA GPU
    if torch.cuda.is_available():
        try:
            # Get GPU memory in GB
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_name = torch.cuda.get_device_name(0)

            # Choose batch size based on VRAM
            if gpu_memory_gb >= 8:
                batch_size = 512
                logger.info(
                    f"GPU detected ({gpu_name}, {gpu_memory_gb:.1f}GB VRAM): using batch size {batch_size}"
                )
                return batch_size
            elif gpu_memory_gb >= 4:
                batch_size = 256
                logger.info(
                    f"GPU detected ({gpu_name}, {gpu_memory_gb:.1f}GB VRAM): using batch size {batch_size}"
                )
                return batch_size
            else:
                batch_size = 128
                logger.info(
                    f"GPU detected ({gpu_name}, {gpu_memory_gb:.1f}GB VRAM): using batch size {batch_size}"
                )
                return batch_size
        except Exception as e:
            logger.warning(f"GPU detection failed: {e}, falling back to CPU batch size")

    # CPU fallback
    logger.info("No GPU detected: using CPU batch size 128")
    return 128


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
                async with aiofiles.open(cache_file, "rb") as f:
                    content_bytes = await f.read()
                    embedding = orjson.loads(content_bytes)

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
            async with aiofiles.open(cache_file, "wb") as f:
                await f.write(orjson.dumps(embedding))
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
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        timeout: float = 300.0,  # 5 minutes default timeout
    ) -> None:
        """Initialize embedding function.

        Args:
            model_name: Name of the sentence transformer model
            timeout: Timeout in seconds for embedding generation (default: 300s)

        Environment Variables:
            MCP_VECTOR_SEARCH_EMBEDDING_MODEL: Override embedding model (highest priority)
        """
        # Check environment variable override FIRST (highest priority)
        env_model = os.environ.get("MCP_VECTOR_SEARCH_EMBEDDING_MODEL")
        if env_model:
            model_name = env_model
            logger.info(f"Using embedding model from environment: {model_name}")

        try:
            # Auto-detect optimal device (MPS > CUDA > CPU)
            device = _detect_device()

            # Detect model dimensions and log info
            try:
                expected_dims = get_model_dimensions(model_name)
                is_code_model = is_code_specific_model(model_name)
                model_type = "code-specific" if is_code_model else "general-purpose"
            except ValueError:
                # Unknown model - will be logged as warning
                expected_dims = "unknown"
                model_type = "unknown"

            # Check if this is CodeT5+ model (needs special handling)
            self.is_codet5p = "codet5p" in model_name.lower()

            if self.is_codet5p:
                # CodeT5+ embedding model requires AutoModel, not SentenceTransformer
                # It's an encoder-decoder model with a projection head that outputs 256d
                logger.info(
                    f"Loading CodeT5+ embedding model: {model_name} "
                    f"(encoder-decoder with 256d projection head)"
                )
                import torch
                from transformers import AutoModel, AutoTokenizer

                with suppress_stdout_stderr():
                    self.model = AutoModel.from_pretrained(  # nosec B615
                        model_name, trust_remote_code=True
                    )
                    self.tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
                        model_name, trust_remote_code=True
                    )

                # Move model to device
                self.model = self.model.to(device)
                self.model.eval()  # Set to evaluation mode

                # CodeT5+ outputs 256d directly (has projection head)
                actual_dims = 256
            else:
                # Standard SentenceTransformer models
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

                # Get actual dimensions from loaded model
                actual_dims = self.model.get_sentence_embedding_dimension()

            self.model_name = model_name
            self.timeout = timeout
            self.device = device  # Store device for use in encode() calls

            # Log device usage and model details
            if device == "mps":
                import subprocess

                try:
                    # Get Apple Silicon chip info
                    result = subprocess.run(  # nosec B607 - safe read-only sysctl
                        ["sysctl", "-n", "machdep.cpu.brand_string"],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    chip_name = (
                        result.stdout.strip()
                        if result.returncode == 0
                        else "Apple Silicon"
                    )
                except Exception:
                    chip_name = "Apple Silicon"

                logger.info(
                    f"Loaded {model_type} embedding model: {model_name} "
                    f"on MPS ({chip_name}) with {actual_dims} dimensions (timeout: {timeout}s)"
                )
            elif device == "cuda":
                import torch

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
            # CUDA contexts are thread-bound, so run directly on CUDA to avoid
            # GPU operations silently falling back to CPU when run in thread pool.
            # For CPU/MPS, use ThreadPoolExecutor with timeout for safety.
            if self.device == "cuda":
                # Run directly on CUDA - no thread pool to avoid context issues
                return self._generate_embeddings(input)

            # For CPU/MPS, use thread pool with timeout
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
        """Internal method to generate embeddings (runs in thread pool).

        Uses optimal batch size for GPU (512 for M4 Max, detected automatically).
        """
        if self.is_codet5p:
            # CodeT5+ special handling: use AutoModel + tokenizer
            import torch

            # Use optimal batch size for GPU throughput
            batch_size = _detect_optimal_batch_size()

            all_embeddings = []
            with torch.no_grad():  # Disable gradient computation for inference
                for i in range(0, len(input), batch_size):
                    batch_texts = input[i : i + batch_size]

                    # Tokenize batch
                    inputs = self.tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt",
                    )

                    # Move inputs to device
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    # Get embeddings from model
                    # CodeT5+ embedding model outputs embeddings directly (256d)
                    outputs = self.model(**inputs)

                    # Extract embeddings (output is already 256d from projection head)
                    # The model outputs a tensor of shape [batch_size, 256]
                    batch_embeddings = outputs.cpu().numpy()

                    all_embeddings.extend(batch_embeddings.tolist())

            return all_embeddings
        else:
            # Standard SentenceTransformer models
            # Use optimal batch size for GPU throughput (5x faster with 512 vs 32)
            batch_size = _detect_optimal_batch_size()
            # CRITICAL: Pass device to ensure input tensors are moved to GPU
            # Without this, model weights are on GPU but inputs stay on CPU (0% GPU compute)
            embeddings = self.model.encode(
                input,
                convert_to_numpy=True,
                batch_size=batch_size,
                show_progress_bar=False,
                device=self.device,  # Ensure inputs go to GPU
            )
            return embeddings.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple documents (ChromaDB compatibility).

        Args:
            texts: List of text documents to embed

        Returns:
            List of embedding vectors, one per document
        """
        return self.__call__(input=texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text (ChromaDB compatibility).

        Args:
            text: Single text string to embed

        Returns:
            Embedding vector for the query text
        """
        return self.__call__(input=[text])[0]


class BatchEmbeddingProcessor:
    """Batch processing for efficient embedding generation with caching."""

    def __init__(
        self,
        embedding_function: CodeBERTEmbeddingFunction,
        cache: EmbeddingCache | None = None,
        batch_size: int | None = None,
    ) -> None:
        """Initialize batch embedding processor.

        Args:
            embedding_function: Function to generate embeddings
            cache: Optional embedding cache
            batch_size: Size of batches for processing (default: auto-detected based on GPU)
        """
        self.embedding_function = embedding_function
        self.cache = cache
        # Use GPU-aware auto-detection if batch size not explicitly provided
        if batch_size is None:
            batch_size = _detect_optimal_batch_size()
        self.batch_size = batch_size

    async def embed_batches_parallel(
        self, texts: list[str], batch_size: int = 32, max_concurrent: int | None = None
    ) -> list[list[float]]:
        """Generate embeddings in parallel batches for improved throughput.

        This method splits the input texts into batches and processes them
        concurrently using asyncio.to_thread() to avoid blocking the event loop.

        Args:
            texts: List of text content to embed
            batch_size: Size of each batch (default: 32)
            max_concurrent: Maximum number of concurrent batches (default: 8 for GPU, override with MCP_VECTOR_SEARCH_MAX_CONCURRENT)

        Returns:
            List of embeddings corresponding to input texts

        Environment Variables:
            MCP_VECTOR_SEARCH_MAX_CONCURRENT: Override max concurrent embedding batches (default: 8)

        Note:
            This method is most effective when:
            - Processing large numbers of texts (100+)
            - Using GPU for embeddings (parallel batches maximize GPU utilization)
            - Balancing batch size with max_concurrent to avoid OOM
        """
        # Auto-detect optimal max_concurrent if not specified
        if max_concurrent is None:
            # Check environment variable first
            env_max_concurrent = os.environ.get("MCP_VECTOR_SEARCH_MAX_CONCURRENT")
            if env_max_concurrent:
                try:
                    max_concurrent = int(env_max_concurrent)
                    logger.debug(
                        f"Using max_concurrent from environment: {max_concurrent}"
                    )
                except ValueError:
                    logger.warning(
                        f"Invalid MCP_VECTOR_SEARCH_MAX_CONCURRENT value: {env_max_concurrent}, using default 8"
                    )
                    max_concurrent = 8
            else:
                # Default to 16 for better GPU utilization (up from 8)
                # GPUs like M4 Max can handle much higher concurrency than CPUs
                # This allows more parallel batches to overlap, improving throughput
                max_concurrent = 16
        import asyncio

        if not texts:
            return []

        # Split texts into batches
        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

        # Semaphore to limit concurrent batch processing
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_batch(batch: list[str]) -> list[list[float]]:
            """Process a single batch in thread pool."""
            async with semaphore:
                # Run embedding generation in thread pool to avoid blocking
                try:
                    return await asyncio.to_thread(self.embedding_function, batch)
                except BaseException as e:
                    # PyO3 panics inherit from BaseException, not Exception
                    if "Python interpreter is not initialized" in str(e):
                        logger.warning("Embedding interrupted during shutdown")
                        raise RuntimeError(
                            "Embedding interrupted during Python shutdown"
                        ) from e
                    raise

        # Process all batches concurrently
        results = await asyncio.gather(*[process_batch(b) for b in batches])

        # Flatten results from all batches
        return [emb for batch_result in results for emb in batch_result]

    async def process_batch(self, contents: list[str]) -> list[list[float]]:
        """Process a batch of content for embeddings with parallel generation.

        Args:
            contents: List of text content to embed

        Returns:
            List of embeddings

        Environment Variables:
            MCP_VECTOR_SEARCH_PARALLEL_EMBEDDINGS: Enable parallel embedding (default: true)
        """
        import time

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
            start_time = time.perf_counter()
            logger.debug(f"Generating {len(uncached_contents)} new embeddings")

            try:
                # Check if parallel embeddings are enabled (default: true)
                use_parallel = os.environ.get(
                    "MCP_VECTOR_SEARCH_PARALLEL_EMBEDDINGS", "true"
                ).lower() in ("true", "1", "yes")

                if use_parallel and len(uncached_contents) >= 16:
                    # Use parallel embedding for moderate+ batches (16+ items)
                    try:
                        logger.debug(
                            f"Using parallel embedding generation for {len(uncached_contents)} items"
                        )
                        # max_concurrent now auto-detects from env var (defaults to 8)
                        new_embeddings = await self.embed_batches_parallel(
                            uncached_contents,
                            batch_size=self.batch_size,
                            max_concurrent=None,  # Auto-detect
                        )
                    except Exception as parallel_error:
                        # Graceful fallback to sequential if parallel fails
                        logger.warning(
                            f"Parallel embedding failed ({parallel_error}), falling back to sequential"
                        )
                        new_embeddings = await self._sequential_embed(uncached_contents)
                else:
                    # Use sequential for small batches or when parallel is disabled
                    new_embeddings = await self._sequential_embed(uncached_contents)

                # Calculate performance metrics
                elapsed_time = time.perf_counter() - start_time
                throughput = (
                    len(uncached_contents) / elapsed_time if elapsed_time > 0 else 0
                )
                logger.info(
                    f"Generated {len(uncached_contents)} embeddings in {elapsed_time:.2f}s "
                    f"({throughput:.1f} chunks/sec)"
                )

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

    async def _sequential_embed(self, contents: list[str]) -> list[list[float]]:
        """Sequential embedding generation (fallback method).

        Args:
            contents: List of text content to embed

        Returns:
            List of embeddings
        """
        import asyncio

        new_embeddings = []
        for i in range(0, len(contents), self.batch_size):
            batch = contents[i : i + self.batch_size]
            # Run in thread pool to avoid blocking
            try:
                batch_embeddings = await asyncio.to_thread(
                    self.embedding_function, batch
                )
                new_embeddings.extend(batch_embeddings)
            except BaseException as e:
                # PyO3 panics inherit from BaseException, not Exception
                if "Python interpreter is not initialized" in str(e):
                    logger.warning("Embedding interrupted during shutdown")
                    raise RuntimeError(
                        "Embedding interrupted during Python shutdown"
                    ) from e
                raise
        return new_embeddings

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


def _default_model_for_device() -> str:
    """Return the default embedding model.

    MiniLM provides fast, efficient embeddings with good quality.
    Use --model graphcodebert for code-specific understanding (data-flow aware).
    """
    return "sentence-transformers/all-MiniLM-L6-v2"


def get_model_dimension(model_name: str | None = None) -> int:
    """Return the embedding dimension for the given model.

    Args:
        model_name: Model name (None = use default)

    Returns:
        Embedding dimension (384 for MiniLM, 768 for GraphCodeBERT, 1024 for SFR)
    """
    if model_name is None:
        model_name = _default_model_for_device()
    if "minilm" in model_name.lower():
        return 384
    # GraphCodeBERT, CodeBERT, and most code models are 768d
    return 768


def create_embedding_function(
    model_name: str | None = None,
    cache_dir: Path | None = None,
    cache_size: int = 1000,
):
    """Create embedding function and cache.

    Args:
        model_name: Name of the embedding model (auto-selected by device if None)
        cache_dir: Directory for caching embeddings
        cache_size: Maximum cache size

    Returns:
        Tuple of (embedding_function, cache)

    Environment Variables:
        MCP_VECTOR_SEARCH_EMBEDDING_MODEL: Override embedding model (highest priority)
    """
    # Check environment variable override FIRST (highest priority)
    env_model = os.environ.get("MCP_VECTOR_SEARCH_EMBEDDING_MODEL")
    if env_model:
        model_name = env_model
        logger.info(f"Using embedding model from environment: {model_name}")
    elif model_name is None:
        model_name = _default_model_for_device()
        logger.info(f"Auto-selected embedding model for device: {model_name}")

    # Use our native CodeBERTEmbeddingFunction which supports GPU (MPS/CUDA)
    # and doesn't require ChromaDB (which has Python 3.14 compatibility issues)
    embedding_function = CodeBERTEmbeddingFunction(model_name)

    cache = None
    if cache_dir:
        cache = EmbeddingCache(cache_dir, cache_size)

    return embedding_function, cache
