# Environment Detection and Adaptive Tuning Audit

**Date:** 2026-03-01
**Scope:** `src/mcp_vector_search/core/` — all environment detection and adaptive tuning paths
**Request:** Deep audit for five deployment environments: macOS local dev, AWS g4dn.xlarge (T4 GPU + EFS), large cloud (256 GB RAM + multi-GPU), Container/ECS (cgroup v2), CPU-only server

---

## Executive Summary

The codebase has solid foundations for macOS/Apple Silicon detection and basic cgroup memory detection, but contains **seven significant gaps** in environment detection that affect real-world deployments. The gaps are ranked below from highest to lowest impact.

---

## A. Storage Type Detection

### What IS detected
Nothing. There is no filesystem type detection anywhere in the codebase.

### What is NOT detected
- **EFS/NFS detection** — no calls to `statfs`, `os.statvfs()`, `/proc/mounts`, or `f_type` comparison against `NFS_SUPER_MAGIC` (0x6969) or the EFS-specific value.
- **SSD vs HDD vs network storage** — completely absent.
- **Write pattern adaptation based on filesystem** — the write batch size of 4096 chunks (indexer.py line 943-944) is hardcoded via env var default, not auto-tuned by filesystem type.

### Write batch size calibration

The `MCP_VECTOR_SEARCH_WRITE_BATCH_SIZE` default of 4096 is set at:
```
src/mcp_vector_search/core/indexer.py:943-944
write_batch_size = int(os.environ.get("MCP_VECTOR_SEARCH_WRITE_BATCH_SIZE", "4096"))
```

At approximately 2 KB per chunk × 4096 chunks = ~8 MB per LanceDB write. This is within EFS's "sweet spot" of 8–64 MB per write. However, it is not tuned automatically — a user on NVMe local storage could safely use a much smaller value (fewer fragments = less overhead), while an EFS deployment should use 16,384 or 32,768 chunks (16–32 MB) to maximize throughput and minimize the per-write round-trip tax.

**Gap #1 — HIGH IMPACT on AWS g4dn + EFS:**
No NFS/EFS detection. The write batch size, queue depth, and compaction strategy should all be auto-tuned when an NFS-type filesystem is detected.

Detection approach:
```python
# Linux: read /proc/mounts, check fstype column
# Or: use ctypes to call statfs() and compare f_type to NFS_SUPER_MAGIC
import struct, ctypes, os

def _detect_filesystem_type(path: str) -> str:
    """Return 'nfs', 'tmpfs', 'ext4', 'xfs', 'apfs', etc."""
    if os.path.exists("/proc/mounts"):
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                # parts: device mountpoint fstype options dump pass
                if len(parts) >= 3:
                    mountpoint = parts[1]
                    fstype = parts[2]
                    try:
                        if os.path.commonpath([path, mountpoint]) == mountpoint:
                            return fstype.lower()
                    except ValueError:
                        pass
    return "unknown"

NFS_TYPES = {"nfs", "nfs4", "nfsd", "efs"}

def _is_network_filesystem(path: str) -> bool:
    return _detect_filesystem_type(path) in NFS_TYPES
```

Behaviors to adapt when NFS detected:
- `write_batch_size` → 16384 (32 MB writes, reduces round trips by 4×)
- `MCP_VECTOR_SEARCH_QUEUE_DEPTH` → 2 (network latency adds to queue pressure)
- Skip `_compact_table()` during indexing (compaction amplifies NFS writes)
- Log a startup warning about EFS high-latency mode

---

## B. GPU Environment Detection

### What IS detected

`embeddings.py::_detect_device()` (lines 104-163):
- MPS availability and PyTorch 2.10.0 regression workaround
- CUDA availability
- GPU count via `torch.cuda.device_count()`
- GPU name via `torch.cuda.get_device_name(0)`

`embeddings.py::_detect_optimal_batch_size()` (lines 166-264):
- Apple Silicon RAM tiers (64 GB → 512, 32 GB → 384, default → 256)
- CUDA VRAM tiers (8 GB+ → 512, 4–8 GB → 256, <4 GB → 128)

### What is NOT detected

**Gap #2 — MEDIUM IMPACT: CUDA compute capability (sm version) not used**

The T4 GPU is sm_75 (Turing). The A100 is sm_80 (Ampere). These have different optimal batch sizes and support different precision modes (FP16, BF16, INT8). The batch size logic in `_detect_optimal_batch_size()` only checks VRAM in GB — not the compute capability — so sm_75 and sm_90 get identical treatment despite very different throughput characteristics.

The T4 has 16 GB VRAM, so it gets `batch_size = 512`. But the T4 is significantly slower than modern GPUs at FP32 inference; it would benefit more from a smaller batch (256–384) that keeps tensor operations in shared memory, and the T4's FP16 tensor cores are not enabled anywhere.

Detection approach (file: `embeddings.py`, after line 234):
```python
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(0)
    # sm_70 = V100, sm_75 = T4, sm_80 = A100, sm_86 = A10G, sm_89 = L4, sm_90 = H100
    compute_sm = major * 10 + minor
    # T4 / V100 (sm_70-75): conservative batch, enable FP16
    # A100+ (sm_80+): aggressive batch, enable BF16
```

**Gap #3 — MEDIUM IMPACT: Multi-GPU not exploited**

`_detect_device()` reports `gpu_count` but always returns `"cuda"` (device 0). On a multi-GPU instance, only one GPU is used. The batch embedding processor sends all work to device 0.

Detection is already present (`torch.cuda.device_count()` is called at line 140) but the result is discarded — it is logged only, not acted upon.

Adaptation for multi-GPU: distribute embedding batches across devices using `DataParallel` or manual device routing per batch. This is high complexity but relevant for "large cloud" environments with multiple GPUs.

**Gap #4 — LOW IMPACT: NVLink / peer-to-peer memory not checked**

Not detected. Not critical for current single-GPU workload.

---

## C. CPU and RAM Detection

### What IS detected

`resource_manager.py::calculate_optimal_workers()` (lines 39-95):
- Uses `psutil.virtual_memory()` for available (not total) RAM — this is correct.
- Uses `os.cpu_count()` as CPU upper bound.

`chunk_processor.py::_detect_optimal_workers()` (lines 38-90):
- Detects Apple Silicon via `platform.processor() == "arm"` and `platform.system() == "Darwin"`.
- Applies tier-based worker counts (16+ cores → 14 workers, etc.).

`hardware.py::detect_hardware_config()` (lines 12-140):
- Reads `/proc/meminfo` for Linux RAM.
- Reads `sysctl hw.memsize` for macOS RAM.

### What is NOT detected

**Gap #5 — HIGH IMPACT on containers and large cloud: CPU quotas not detected**

`os.cpu_count()` returns the number of logical CPUs on the host machine, not the CPU allocation granted to the process by cgroup. On a container with `--cpus=2` on a 64-core host, `os.cpu_count()` returns 64. The indexer will spawn 14–48 workers, all competing for 2 CPU cores. This causes massive context-switching overhead and is effectively a significant performance regression for container deployments.

cgroup v2 CPU quota path: `/sys/fs/cgroup/cpu.max` contains `quota_us period_us` (e.g., `200000 100000` = 2 CPUs). cgroup v1 path: `/sys/fs/cgroup/cpu/cpu.cfs_quota_us` and `cpu.cfs_period_us`.

Detection approach (file: `resource_manager.py`, new function before `calculate_optimal_workers`):
```python
def _detect_cpu_quota() -> int | None:
    """Detect container CPU quota from cgroup.

    Returns effective CPU count, or None if no quota / detection fails.
    cgroup v2: /sys/fs/cgroup/cpu.max  -> "quota_us period_us" or "max period_us"
    cgroup v1: /sys/fs/cgroup/cpu/cpu.cfs_quota_us (quota) and cfs_period_us (period)
    """
    import pathlib, math

    # cgroup v2
    try:
        cpu_max = pathlib.Path("/sys/fs/cgroup/cpu.max")
        if cpu_max.exists():
            parts = cpu_max.read_text().strip().split()
            if parts[0] != "max":
                quota_us = int(parts[0])
                period_us = int(parts[1])
                return max(1, math.ceil(quota_us / period_us))
    except (FileNotFoundError, ValueError, OSError):
        pass

    # cgroup v1
    try:
        quota_path = pathlib.Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        period_path = pathlib.Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if quota_path.exists() and period_path.exists():
            quota_us = int(quota_path.read_text().strip())
            if quota_us > 0:  # -1 means no quota
                period_us = int(period_path.read_text().strip())
                return max(1, math.ceil(quota_us / period_us))
    except (FileNotFoundError, ValueError, OSError):
        pass

    return None
```

Then in `calculate_optimal_workers`, use `min(os.cpu_count(), _detect_cpu_quota() or os.cpu_count())` as the CPU ceiling.

**Gap #6 — LOW IMPACT: Physical cores (not hyperthreads) not used**

`os.cpu_count()` returns logical CPUs (hyperthreads included). On an AWS g4dn.xlarge (4 vCPUs = 2 physical cores × 2 HT), the worker calculation produces `max_workers = 4`. For CPU-bound parsing workloads, using more than the physical core count gives no throughput gain (hyperthreads share execution units). Physical core count on Linux is in `/proc/cpuinfo` as unique `core id` + `physical id` combinations. This is a minor improvement for parsing-heavy workloads.

**Gap #7 — LOW IMPACT: NUMA topology not considered**

Not detected. Relevant for 256 GB+ multi-socket servers where memory access latency differs significantly across NUMA nodes. Not critical for current workloads but would matter for very large embedding batches on dual-socket instances.

---

## D. Container / cgroup Detection

### What IS detected

`memory_monitor.py::_detect_memory_limit_gb()` (lines 14-49) — this is the post-#122 fix:
- cgroup v2: reads `/sys/fs/cgroup/memory.max` (returns `max` if unlimited)
- cgroup v1: reads `/sys/fs/cgroup/memory/memory.limit_in_bytes` (checks `< (1 << 62)` for unlimited sentinel)
- Falls back to `psutil.virtual_memory().total`

This is **correctly implemented** for memory limits.

### What is NOT detected

CPU quotas — covered in Gap #5 above.

The indexer has no awareness that it is inside a container at all. It does not read `/.dockerenv`, check `CONTAINER` env vars, or adjust any non-memory parameters for containerized operation. The memory detection is the only container-aware code path.

---

## E. Network Filesystem Adaptation (write_batch_size)

### Current state

`indexer.py` lines 942-945:
```python
write_batch_size = int(
    os.environ.get("MCP_VECTOR_SEARCH_WRITE_BATCH_SIZE", "4096")
)
```

This is a static default. The comment on line 1084 says "reduces LanceDB fragment count and EFS I/O amplification" — so EFS is a known concern, but detection and auto-tuning are not implemented.

### Calibration analysis

- 4096 chunks × ~2 KB/chunk = ~8 MB per LanceDB write → matches EFS sweet spot lower bound.
- Increasing to 16384 (32 MB) would provide more headroom and fewer write round-trips.
- On NVMe local storage, smaller values (512–1024) produce fewer fragments without network penalty.

The write batch for chunk phase (file batching) at `indexer.py:1498` is hardcoded at 256 files:
```python
write_batch_files = 256
```
This is never auto-tuned.

**Recommendation:** When NFS/EFS detected, set `write_batch_size` to 16384 and `write_batch_files` to 512. When local NVMe detected, use 1024 and 128 respectively.

---

## F. macOS-Specific Guards

### What IS guarded

`lancedb_backend.py::optimize()` (lines 459-514):
- Full guard: `if platform.system() == "Darwin": return` — skips compaction entirely on macOS.

`vectors_backend.py::_compact_table()` (lines 652-704):
- Full guard: `if platform.system() == "Darwin": return` — skips `compact_files()` on macOS.

`chunk_processor.py::_get_mp_context()` (lines 20-35):
- Uses `spawn` instead of `fork` on macOS to avoid SIGILL from fork + PyTorch.

### What is NOT guarded

**Potential remaining SIGBUS risk:** The macOS guards in `_compact_table()` and `optimize()` prevent LanceDB compaction (which calls `compact_files()` internally, which uses mmap). However, the periodic compaction in `vectors_backend.py` is triggered every 500 appends (`_append_count % 500 == 0`, line 643). The guard is applied inside `_compact_table()`, so the macOS check is correctly in place. All paths are guarded.

However, the `lancedb.connect()` call itself opens the database with memory-mapped I/O. On a full disk or a file that is truncated by another process, LanceDB's Rust backend mmap operations could SIGBUS outside the guarded compaction path. This is an edge case, not a practical daily risk.

**Overall assessment:** macOS SIGBUS guards are adequate for the identified risk (compaction + MPS). No critical unguarded paths found.

---

## G. systemd / Service Environment

### What IS detected
Nothing. The indexer has no awareness of systemd.

### What is NOT detected

**Gap (LOW IMPACT):** No systemd integration:
- `INVOCATION_ID` env var is set by systemd — not read.
- `JOURNAL_STREAM` env var indicates journald is the log sink — not detected.
- Log format is not adjusted for journald (no ISO timestamp stripping since journald adds its own).
- Process priority (`nice`/`ionice`) is not adjusted.
- The OOM score (`/proc/self/oom_score_adj`) is not set to protect the service process.

**Detection approach:**
```python
def _is_systemd_managed() -> bool:
    """Detect if running as a systemd service."""
    return bool(os.environ.get("INVOCATION_ID"))

def _setup_journald_logging() -> None:
    """Switch to plain format when running under journald."""
    if os.environ.get("JOURNAL_STREAM"):
        # journald adds timestamps — suppress ours to avoid duplication
        logger.remove()
        logger.add(sys.stderr, level="INFO", format="{level}: {message}", colorize=False)
```

**Process priority:** For a background indexing service, setting `os.nice(10)` or `os.setpriority(os.PRIO_PROCESS, 0, 10)` at startup would prevent the indexer from starving interactive processes on shared machines. This is recommended but optional.

---

## Ranked Gap Summary

| Rank | Gap | Impact | Environment | File(s) | Detection Method |
|------|-----|--------|-------------|---------|-----------------|
| 1 | NFS/EFS filesystem detection — write patterns not adapted | HIGH | AWS g4dn + EFS | `indexer.py:943`, `vectors_backend.py:652` | `/proc/mounts` fstype column |
| 2 | CPU cgroup quota not detected — over-parallelizes in containers | HIGH | ECS/Container | `resource_manager.py:71`, `chunk_processor.py:55` | `/sys/fs/cgroup/cpu.max` (v2) or `cpu.cfs_quota_us` (v1) |
| 3 | CUDA compute capability (sm version) not used for batch sizing | MEDIUM | AWS g4dn T4 | `embeddings.py:234` | `torch.cuda.get_device_capability(0)` |
| 4 | Multi-GPU not exploited — all work goes to device 0 | MEDIUM | Large cloud | `embeddings.py:140-145` | `torch.cuda.device_count()` already called, result ignored |
| 5 | Physical core count vs hyperthreads not distinguished | LOW | All Linux | `resource_manager.py:71`, `chunk_processor.py:55` | `/proc/cpuinfo` unique `core id` + `physical id` pairs |
| 6 | write_batch_size not auto-tuned by filesystem type | LOW | AWS g4dn + EFS | `indexer.py:943-944` | Follows from Gap #1 |
| 7 | systemd/journald not detected — log format not adapted, no nice() | LOW | AWS systemd service | `output.py:69`, startup | `INVOCATION_ID`, `JOURNAL_STREAM` env vars |

---

## Existing Correct Detections (Not Gaps)

- cgroup v2 memory limit: correctly reads `/sys/fs/cgroup/memory.max` (`memory_monitor.py:30`)
- cgroup v1 memory limit: correctly reads `memory.limit_in_bytes` with unlimited sentinel (`memory_monitor.py:40`)
- CUDA VRAM-based batch size: `torch.cuda.get_device_properties(0).total_memory` (`embeddings.py:237`)
- Available (not total) RAM for worker calculation: `psutil.virtual_memory().available` (`resource_manager.py:34`)
- macOS SIGBUS guard on compaction: `platform.system() == "Darwin"` (`lancedb_backend.py:472`, `vectors_backend.py:665`)
- Apple Silicon multiprocessing context (spawn not fork): `sys.platform == "darwin"` (`chunk_processor.py:31`)
- MPS PyTorch 2.10.0 regression: version string check (`embeddings.py:125`)
- macOS available RAM via `sysctl hw.memsize` (`embeddings.py:201`)
- Linux RAM via `/proc/meminfo` MemAvailable (`index.py:127`)

---

## Files and Line Numbers for Each Gap

### Gap #1 — NFS/EFS detection
- **Detection to add:** New function `_detect_filesystem_type(path)` in `resource_manager.py` or `memory_monitor.py`
- **Adaptation point:** `indexer.py:943-944` — make `write_batch_size` dynamic
- **Adaptation point:** `indexer.py:1498` — make `write_batch_files` dynamic
- **Adaptation point:** `vectors_backend.py:643` — skip periodic compaction on NFS

### Gap #2 — CPU cgroup quota
- **Detection to add:** New function `_detect_cpu_quota()` in `resource_manager.py` (before line 39)
- **Adaptation point:** `resource_manager.py:71` — replace `os.cpu_count()` with `min(os.cpu_count(), cpu_quota or os.cpu_count())`
- **Adaptation point:** `chunk_processor.py:55` — same fix in `_detect_optimal_workers()`

### Gap #3 — CUDA compute capability
- **Detection to add:** In `embeddings.py::_detect_optimal_batch_size()`, after line 234
- **Adaptation:** Add sm-based batch size tiers: sm_75 (T4) → 256, sm_80+ (A100) → 512, sm_90 (H100) → 1024

### Gap #4 — Multi-GPU
- **Detection:** `torch.cuda.device_count()` is already called at `embeddings.py:140` — result is logged but unused
- **Adaptation point:** `embeddings.py::BatchEmbeddingProcessor.embed_batches_parallel()` — distribute batches across GPUs

### Gap #5 — Physical cores
- **Detection to add:** Parse `/proc/cpuinfo` for unique `(core id, physical id)` pairs in `resource_manager.py`
- **Adaptation point:** `resource_manager.py:71` and `chunk_processor.py:55`

### Gap #6 — write_batch_size auto-tuning
- Depends on Gap #1 (NFS detection)
- **Adaptation point:** `indexer.py:943-944` — conditional on filesystem type

### Gap #7 — systemd/journald
- **Detection to add:** Check `os.environ.get("INVOCATION_ID")` and `os.environ.get("JOURNAL_STREAM")`
- **Adaptation point:** `output.py::setup_logging()` (line 69) — switch to plain format for journald
- **Optional:** `os.nice(10)` at process start in MCP server main or CLI main
