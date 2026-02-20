# GPU Utilization Investigation: mcp-vector-search v2.5.48

**Investigation Date:** 2026-02-19
**Environment:** AWS g4dn.xlarge (Tesla T4 GPU, 16GB GPU memory)
**Issue:** 0% GPU utilization during reindex with GraphCodeBERT embedding model
**Version:** v2.5.48 (latest as of investigation)

---

## Executive Summary

**CRITICAL FINDING: The embedding model IS configured correctly for GPU usage, but may lose GPU context during multiprocess parsing or face CUDA initialization issues.**

The codebase has proper CUDA detection and device placement (`device=self.device` in `encode()` calls), but there are **two potential root causes** for 0% GPU utilization:

1. **Process Context Loss**: Embedding function created in main process, but multiprocessing parsing may spawn workers that don't inherit the GPU-initialized model
2. **PyTorch CUDA Initialization**: On AWS GPU instances, PyTorch may not detect CUDA properly without explicit environment setup or driver verification

**Key Evidence:**
- Device detection works: `_detect_device()` properly identifies CUDA at lines 97-149
- Device is passed to model: `SentenceTransformer(..., device=device)` at line 421
- Device is used in encoding: `model.encode(..., device=self.device)` at line 528
- **However**: Model is created once during database initialization, potentially before multiprocessing workers spawn

---

## Code Analysis Findings

### 1. Device Detection (`src/mcp_vector_search/core/embeddings.py`)

**Lines 97-149: `_detect_device()` function**

```python
def _detect_device() -> str:
    """Detect optimal compute device (MPS > CUDA > CPU)."""
    import torch

    # Check environment variable override first
    env_device = os.environ.get("MCP_VECTOR_SEARCH_DEVICE", "").lower()
    if env_device in ("cpu", "cuda", "mps"):
        logger.info(f"Using device from environment override: {env_device}")
        return env_device

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
```

**Analysis:**
- ✅ Properly detects CUDA availability with `torch.cuda.is_available()`
- ✅ Logs GPU name and count for verification
- ✅ Provides diagnostic info when CUDA is unavailable
- ✅ Supports environment override: `MCP_VECTOR_SEARCH_DEVICE=cuda`

**Potential Issue:**
- Detection runs at **module import time** (when `CodeBERTEmbeddingFunction` is instantiated)
- If PyTorch is imported before CUDA drivers are initialized, `torch.cuda.is_available()` may return `False`
- No retry mechanism if CUDA becomes available later

### 2. Embedding Model Initialization (`src/mcp_vector_search/core/embeddings.py`)

**Lines 373-477: `CodeBERTEmbeddingFunction.__init__()`**

```python
class CodeBERTEmbeddingFunction:
    def __init__(
        self,
        model_name: str = "microsoft/graphcodebert-base",
        timeout: float = 300.0,
    ) -> None:
        # Auto-detect optimal device (MPS > CUDA > CPU)
        device = _detect_device()  # Line 398

        # Load model with device placement
        with suppress_stdout_stderr():
            self.model = SentenceTransformer(
                model_name, device=device, trust_remote_code=True
            )  # Line 420-422

        self.model_name = model_name
        self.timeout = timeout
        self.device = device  # Store device for use in encode() calls (Line 425)
```

**Analysis:**
- ✅ Detects device correctly (`_detect_device()`)
- ✅ Passes device to `SentenceTransformer` constructor
- ✅ Stores device in `self.device` for later use
- ✅ Logs device usage and model details (lines 454-466)

**Potential Issue:**
- Model is created **once** during `create_embedding_function()` call
- This happens during database initialization in the **main process**
- Multiprocessing workers may not inherit this GPU-initialized model

### 3. Embedding Generation (`src/mcp_vector_search/core/embeddings.py`)

**Lines 514-530: `_generate_embeddings()` method**

```python
def _generate_embeddings(self, input: list[str]) -> list[list[float]]:
    """Internal method to generate embeddings (runs in thread pool)."""
    batch_size = _detect_optimal_batch_size()

    # CRITICAL: Pass device to ensure input tensors are moved to GPU
    # Without this, model weights are on GPU but inputs stay on CPU (0% GPU compute)
    embeddings = self.model.encode(
        input,
        convert_to_numpy=True,
        batch_size=batch_size,
        show_progress_bar=False,
        device=self.device,  # Ensure inputs go to GPU (Line 528)
    )
    return embeddings.tolist()
```

**Analysis:**
- ✅ **CRITICAL LINE 528**: Passes `device=self.device` to `model.encode()`
- ✅ This ensures input tensors are moved to GPU (without this, you'd see 0% GPU usage)
- ✅ Uses optimal batch size for GPU throughput (512 for 8GB+ VRAM)

**Verification:**
- This is **correct implementation** for GPU acceleration
- The comment at line 522-523 explicitly states why this is needed
- **If this code runs, GPU should be utilized**

### 4. Multiprocessing Interaction (`src/mcp_vector_search/core/indexer.py`)

**Lines 1189-1215: Embedding during Phase 2**

```python
async def _phase2_embed_chunks(self, batch_size: int = 10000, checkpoint_interval: int = 50000):
    """Phase 2: Embed pending chunks, store to vectors.lance."""

    # Inside the embedding loop:
    contents = [c["content"] for c in pending]

    # Method 1: Check for _embedding_function (ChromaDB)
    if hasattr(self.database, "_embedding_function"):
        vectors = self.database._embedding_function(contents)  # Line 1194

    # Method 3: Use database.embedding_function (proper API)
    elif hasattr(self.database, "embedding_function"):
        vectors = self.database.embedding_function.embed_documents(contents)  # Line 1204-1206
```

**Analysis:**
- ✅ Embedding happens in the **main async event loop**, not in multiprocess workers
- ✅ Uses the same `embedding_function` instance created during database initialization
- ❌ **Potential Issue**: If parsing happens in multiprocess workers, they may have spawned their own embedding instances

**Critical Question:**
- Does Phase 1 (parsing) run in multiprocess workers?
- Does Phase 2 (embedding) run in the main process or workers?

### 5. Multiprocessing Context (`src/mcp_vector_search/core/chunk_processor.py`)

**Lines 112-170: `_parse_file_standalone()` - multiprocessing target**

```python
def _parse_file_standalone(
    args: tuple[Path, str | None],
) -> tuple[Path, list[CodeChunk], Exception | None]:
    """Parse a single file - standalone function for multiprocessing.

    This function must be at module level (not a method) to be picklable for
    multiprocessing. It creates its own parser registry to avoid serialization issues.
    """
    file_path, subproject_info_json = args

    # Create parser registry in this process (lazy loading, only creates needed parser)
    parser_registry = get_parser_registry()  # Line 138

    # Parse file synchronously - no async/event loop overhead!
    chunks = parser.parse_file_sync(file_path)  # Line 145
```

**Analysis:**
- ✅ Parsing happens in multiprocess workers (lines 261-301)
- ✅ Each worker creates its own `parser_registry`
- ❌ **EMBEDDING DOES NOT HAPPEN HERE** - only parsing happens in workers
- ✅ Embedding happens later in Phase 2, in the main process (see indexer.py lines 1189-1215)

**Conclusion:**
- Multiprocessing is used for **parsing only**, not embedding
- Embedding happens in the **main process** during Phase 2
- This means the GPU-initialized model should be accessible during embedding

---

## Root Cause Analysis

### Hypothesis 1: PyTorch CUDA Not Properly Initialized on AWS ⭐ **MOST LIKELY**

**Evidence:**
1. `torch.cuda.is_available()` may return `False` on AWS GPU instances if:
   - PyTorch installed without CUDA support (`pip install torch` vs `pip install torch --index-url ...cu121`)
   - CUDA drivers not properly initialized before PyTorch import
   - NVIDIA driver version mismatch with PyTorch CUDA version

2. The code logs diagnostic info when CUDA is unavailable (lines 133-146)
   - Check logs for: "CUDA not available: PyTorch installed without CUDA support"
   - Check logs for: "Using CPU backend (no GPU acceleration)"

**Verification Steps:**
```bash
# On g4dn.xlarge instance:
nvidia-smi  # Should show Tesla T4 GPU

python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python3 -c "import torch; print('CUDA version:', torch.version.cuda)"
python3 -c "import torch; print('GPU count:', torch.cuda.device_count())"
python3 -c "import torch; print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

**Expected Output (if working):**
```
CUDA available: True
CUDA version: 12.1
GPU count: 1
GPU name: Tesla T4
```

**If Not Working:**
```
CUDA available: False
CUDA version: None
GPU count: 0
GPU name: N/A
```

### Hypothesis 2: Environment Variable Override

**Evidence:**
- Lines 109-112 in `embeddings.py` check for `MCP_VECTOR_SEARCH_DEVICE` override
- If set to `"cpu"`, it will bypass CUDA detection

**Verification:**
```bash
# Check if environment variable is set
echo $MCP_VECTOR_SEARCH_DEVICE

# Check in Python environment
python3 -c "import os; print('Device override:', os.environ.get('MCP_VECTOR_SEARCH_DEVICE', 'not set'))"
```

### Hypothesis 3: Model Created Before CUDA Initialization (Less Likely)

**Evidence:**
- Model is created during `ComponentFactory.create_embedding_function()` (factory.py line 159)
- This happens **before** indexing starts, during component initialization
- If CUDA drivers are not yet initialized, `torch.cuda.is_available()` may return `False`

**Less Likely Because:**
- CUDA drivers should be initialized at system boot on AWS GPU instances
- PyTorch's CUDA detection should work at any time after driver initialization

---

## Diagnostic Recommendations

### 1. Add GPU Utilization Logging

**File:** `src/mcp_vector_search/core/embeddings.py`
**Lines:** 514-530 (`_generate_embeddings` method)

Add GPU memory tracking before/after embedding:

```python
def _generate_embeddings(self, input: list[str]) -> list[list[float]]:
    """Internal method to generate embeddings (runs in thread pool)."""
    batch_size = _detect_optimal_batch_size()

    # LOG GPU MEMORY BEFORE EMBEDDING
    if self.device == "cuda":
        import torch
        logger.info(f"GPU memory before embedding: {torch.cuda.memory_allocated(0) / 1024**3:.2f}GB allocated, {torch.cuda.memory_reserved(0) / 1024**3:.2f}GB reserved")

    embeddings = self.model.encode(
        input,
        convert_to_numpy=True,
        batch_size=batch_size,
        show_progress_bar=False,
        device=self.device,
    )

    # LOG GPU MEMORY AFTER EMBEDDING
    if self.device == "cuda":
        logger.info(f"GPU memory after embedding: {torch.cuda.memory_allocated(0) / 1024**3:.2f}GB allocated, {torch.cuda.memory_reserved(0) / 1024**3:.2f}GB reserved")

    return embeddings.tolist()
```

### 2. Force CUDA Device at Runtime

**Environment Variable Override:**

```bash
# Force CUDA usage (bypasses auto-detection)
export MCP_VECTOR_SEARCH_DEVICE=cuda

# Verify it's set
mcp-vector-search index-project
```

**Expected Log Output:**
```
Using device from environment override: cuda
Loaded code-specific embedding model: microsoft/graphcodebert-base on GPU (Tesla T4) with 768 dimensions
```

### 3. Verify PyTorch CUDA Installation

**Check PyTorch Build:**

```bash
# Check if PyTorch has CUDA support
python3 -c "import torch; print('CUDA built:', torch.backends.cuda.is_built() if hasattr(torch.backends, 'cuda') else False)"

# Check PyTorch installation details
pip show torch | grep -E "Name|Version|Location"

# Check if torch was installed with CUDA wheels
pip list | grep torch
```

**If PyTorch is CPU-only, reinstall with CUDA:**

```bash
# Uninstall CPU-only PyTorch
pip uninstall torch torchvision torchaudio

# Install CUDA-enabled PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA is available
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### 4. Monitor GPU During Reindex

**Real-time GPU Monitoring:**

```bash
# In one terminal: monitor GPU utilization every second
watch -n 1 nvidia-smi

# In another terminal: run reindex
mcp-vector-search index-project --force
```

**Expected GPU Metrics During Embedding Phase:**
- **GPU Utilization:** 80-100% during embedding batches
- **GPU Memory Usage:** 2-4GB for GraphCodeBERT (768d embeddings, batch size 512)
- **Temperature:** Should increase from idle (~40°C) to active (~70-80°C)

**If GPU Utilization Stays at 0%:**
- Either PyTorch is using CPU
- Or device detection failed silently

---

## Code Paths to GPU Usage

### Successful GPU Path (when working correctly):

```
1. ComponentFactory.create_embedding_function() [factory.py:159]
   ↓
2. create_embedding_function(model_name) [embeddings.py:819]
   ↓
3. CodeBERTEmbeddingFunction.__init__() [embeddings.py:376]
   ↓
4. _detect_device() → "cuda" [embeddings.py:398]
   ↓
5. SentenceTransformer(model, device="cuda") [embeddings.py:420-422]
   ↓ (Model weights loaded onto GPU)

6. During Phase 2 embedding [indexer.py:1189-1215]:
   database.embedding_function.embed_documents(contents)
   ↓
7. CodeBERTEmbeddingFunction.__call__(contents) [embeddings.py:483]
   ↓
8. _generate_embeddings(contents) [embeddings.py:514]
   ↓
9. model.encode(..., device="cuda") [embeddings.py:523-529]
   ↓ (Input tensors moved to GPU, computation happens on GPU)

10. GPU shows 80-100% utilization during embedding
```

### Failed GPU Path (0% utilization):

```
1. ComponentFactory.create_embedding_function() [factory.py:159]
   ↓
2. create_embedding_function(model_name) [embeddings.py:819]
   ↓
3. CodeBERTEmbeddingFunction.__init__() [embeddings.py:376]
   ↓
4. _detect_device() → "cpu" ❌ [embeddings.py:398]
   ↓ (torch.cuda.is_available() returned False)

5. SentenceTransformer(model, device="cpu") [embeddings.py:420-422]
   ↓ (Model weights loaded onto CPU)

6. Logs show: "Using CPU backend (no GPU acceleration)" [embeddings.py:148]

7. During Phase 2 embedding:
   model.encode(..., device="cpu") [embeddings.py:528]
   ↓ (All computation happens on CPU)

8. GPU shows 0% utilization (not being used)
```

---

## Answers to Investigation Questions

### Q1: How does device detection work?

**Answer:** Device detection happens in `_detect_device()` (embeddings.py:97-149)

- **Priority 1:** Environment variable `MCP_VECTOR_SEARCH_DEVICE` (if set, bypasses auto-detection)
- **Priority 2:** Check `torch.cuda.is_available()` for CUDA support
- **Priority 3:** Fall back to CPU

**Key Finding:** Detection is solid, but relies on `torch.cuda.is_available()` returning `True`. If PyTorch is CPU-only or CUDA drivers aren't initialized, it will default to CPU.

### Q2: Does it properly detect CUDA and move the model to GPU?

**Answer:** YES, the code is correct ✅

- Line 398: `device = _detect_device()` detects CUDA
- Line 421: `SentenceTransformer(..., device=device)` moves model to GPU
- Line 425: `self.device = device` stores device for later use
- Line 528: `model.encode(..., device=self.device)` moves inputs to GPU

**The implementation is correct. If GPU isn't being used, it's because `_detect_device()` returned `"cpu"`.**

### Q3: Is there a fallback to CPU that might trigger unexpectedly?

**Answer:** YES, fallback triggers when: ✅

1. `torch.cuda.is_available()` returns `False` (lines 125-131)
2. Reasons this might happen on g4dn.xlarge:
   - PyTorch installed without CUDA support
   - NVIDIA drivers not properly initialized
   - Driver version mismatch with PyTorch CUDA version

**Diagnostic logs are provided** (lines 133-146) to help debug why CUDA is unavailable.

### Q4: Could multiprocessing cause the model to lose GPU context?

**Answer:** NO, multiprocessing should NOT affect GPU usage ❌

- Multiprocessing is used for **parsing only** (chunk_processor.py:261-301)
- **Embedding happens in the main process** (indexer.py:1189-1215)
- The embedding function is created once and reused (no re-initialization in workers)

**Conclusion:** Multiprocessing is not the root cause.

### Q5: Does `_default_model_for_device()` properly return GraphCodeBERT?

**Answer:** YES, but it doesn't consider device ✅

```python
def _default_model_for_device() -> str:
    """Return the default embedding model."""
    return "microsoft/graphcodebert-base"  # Line 799
```

- This function **always returns GraphCodeBERT** regardless of device
- The function name is misleading (doesn't actually use device info)
- GraphCodeBERT works on both CPU and GPU (768-dimensional embeddings)

**Conclusion:** Model selection is fine, but the function name is confusing.

### Q6: Does sentence-transformers properly use CUDA?

**Answer:** YES, when configured correctly ✅

- `SentenceTransformer` from `sentence-transformers` library properly supports CUDA
- The code passes `device="cuda"` to the constructor (line 421)
- The code passes `device=self.device` to `encode()` (line 528)

**This is the correct way to use GPU with sentence-transformers.**

---

## Key Question Answer

**On a g4dn.xlarge with Tesla T4 and CUDA drivers, would v2.5.48 actually use the GPU for GraphCodeBERT embeddings?**

**Answer: IT SHOULD, but there's likely a PyTorch installation or CUDA driver issue.**

**The code is correct:**
- ✅ Device detection logic is solid
- ✅ Device is passed to model constructor
- ✅ Device is passed to encode() method
- ✅ Multiprocessing doesn't interfere (embedding happens in main process)

**The problem is environmental:**
- ❌ PyTorch may be installed without CUDA support (`pip install torch` without CUDA wheels)
- ❌ CUDA drivers may not be properly initialized before PyTorch import
- ❌ Driver version may not match PyTorch CUDA version

**To fix:**

1. **Verify PyTorch CUDA support:**
   ```bash
   python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
   ```

2. **If False, reinstall PyTorch with CUDA:**
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Force CUDA device as workaround:**
   ```bash
   export MCP_VECTOR_SEARCH_DEVICE=cuda
   mcp-vector-search index-project
   ```

4. **Check logs for device usage:**
   - Look for: "Using CUDA backend for GPU acceleration (1 GPU(s): Tesla T4)"
   - NOT: "Using CPU backend (no GPU acceleration)"

---

## What Needs to Change

### Immediate Actions (To Diagnose Issue)

1. **Check PyTorch CUDA availability on AWS instance**
   ```bash
   python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
   ```

2. **Check mcp-vector-search logs during initialization**
   - Look for device detection logs
   - Look for "Using CUDA backend" vs "Using CPU backend"

3. **Force CUDA device and retry**
   ```bash
   export MCP_VECTOR_SEARCH_DEVICE=cuda
   mcp-vector-search index-project --force
   ```

### Code Improvements (For Better Diagnostics)

**File:** `src/mcp_vector_search/core/embeddings.py`

**1. Add GPU memory logging in `_generate_embeddings()`** (lines 514-530)

```python
def _generate_embeddings(self, input: list[str]) -> list[list[float]]:
    """Internal method to generate embeddings."""
    batch_size = _detect_optimal_batch_size()

    # Add GPU monitoring
    if self.device == "cuda":
        import torch
        mem_before = torch.cuda.memory_allocated(0) / 1024**3
        logger.debug(f"GPU memory before batch: {mem_before:.2f}GB")

    embeddings = self.model.encode(
        input,
        convert_to_numpy=True,
        batch_size=batch_size,
        show_progress_bar=False,
        device=self.device,
    )

    if self.device == "cuda":
        mem_after = torch.cuda.memory_allocated(0) / 1024**3
        logger.debug(f"GPU memory after batch: {mem_after:.2f}GB (delta: {mem_after - mem_before:.2f}GB)")

    return embeddings.tolist()
```

**2. Add warning when CUDA is unavailable but GPU instance detected** (lines 133-146)

```python
# After checking torch.cuda.is_available():
if not cuda_built:
    logger.warning(
        "⚠️  CUDA not available: PyTorch installed without CUDA support. "
        "On GPU instances, reinstall with: pip install torch --index-url https://download.pytorch.org/whl/cu121"
    )
else:
    # Check if we're on a GPU instance but CUDA isn't working
    try:
        import subprocess
        nvidia_result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
        if nvidia_result.returncode == 0:
            logger.error(
                "⚠️  GPU DETECTED but PyTorch CUDA is unavailable! "
                "GPU will not be used for embeddings. "
                "Check: nvidia-smi, PyTorch version, and CUDA driver compatibility."
            )
    except FileNotFoundError:
        pass  # nvidia-smi not available, not a GPU instance

    logger.debug(
        "CUDA not available: PyTorch has CUDA support but no GPU detected. "
        "Check: nvidia-smi, NVIDIA drivers, and GPU instance type."
    )
```

**3. Log device usage at start of Phase 2** (indexer.py:1100)

```python
logger.info(
    f"🧠 Phase 2: Embedding pending chunks (device: {self.database.embedding_function.device})..."
)
```

---

## File References

### Key Files Analyzed

1. **`src/mcp_vector_search/core/embeddings.py`**
   - Lines 97-149: Device detection (`_detect_device()`)
   - Lines 373-477: Model initialization (`CodeBERTEmbeddingFunction.__init__()`)
   - Lines 514-530: Embedding generation (`_generate_embeddings()`)
   - Lines 819-854: Embedding function factory (`create_embedding_function()`)

2. **`src/mcp_vector_search/core/indexer.py`**
   - Lines 1080-1270: Phase 2 embedding (`_phase2_embed_chunks()`)
   - Lines 1189-1215: Actual embedding call (uses `database.embedding_function`)

3. **`src/mcp_vector_search/core/chunk_processor.py`**
   - Lines 112-170: Multiprocess parsing worker (`_parse_file_standalone()`)
   - Lines 261-301: Multiprocess executor (`parse_files_multiprocess()`)

4. **`src/mcp_vector_search/core/factory.py`**
   - Lines 54-58: Embedding function factory method
   - Lines 159-161: Embedding function creation during initialization

---

## Conclusion

**The codebase is correctly implemented for GPU acceleration. The issue is almost certainly environmental:**

1. **Most Likely:** PyTorch installed without CUDA support on the AWS instance
2. **Second Most Likely:** CUDA drivers not properly initialized when PyTorch is imported
3. **Least Likely:** Environment variable `MCP_VECTOR_SEARCH_DEVICE` set to `"cpu"`

**Recommended Actions:**

1. **Immediate:** Check `torch.cuda.is_available()` on the AWS instance
2. **Immediate:** Check mcp-vector-search logs for device detection messages
3. **Workaround:** Force CUDA with `export MCP_VECTOR_SEARCH_DEVICE=cuda`
4. **Fix:** Reinstall PyTorch with CUDA wheels if needed
5. **Enhancement:** Add GPU memory logging to confirm GPU usage

**Expected Outcome After Fix:**
- Logs show: "Using CUDA backend for GPU acceleration (1 GPU(s): Tesla T4)"
- `nvidia-smi` shows 80-100% GPU utilization during Phase 2 (embedding)
- GPU memory usage: 2-4GB for GraphCodeBERT with batch size 512
- Embedding throughput: 500-1000+ chunks/sec (vs 50-100 on CPU)
