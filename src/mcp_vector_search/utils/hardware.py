"""Hardware detection and logging utilities for Apple Silicon M4 Max optimizations."""

import multiprocessing
import os
import platform
import subprocess
from typing import Any

from loguru import logger


def detect_hardware_config() -> dict[str, Any]:
    """Detect hardware configuration for optimal performance settings.

    Returns:
        Dictionary with hardware information:
        - device: Compute device (mps, cuda, cpu)
        - ram_gb: Total RAM in GB
        - cpu_cores: Number of CPU cores
        - cpu_arch: CPU architecture (arm, x86_64, etc.)
        - system: Operating system (Darwin, Linux, Windows)
        - chip_name: Chip name (Apple M4 Max, Intel Core i9, etc.)
        - batch_size: Recommended batch size for embeddings
        - write_buffer_size: Recommended write buffer size
        - max_workers: Recommended number of worker processes
        - cache_size: Recommended cache size
    """
    config = {
        "device": "cpu",
        "ram_gb": 0,
        "cpu_cores": multiprocessing.cpu_count(),
        "cpu_arch": platform.processor() or platform.machine(),
        "system": platform.system(),
        "chip_name": "Unknown",
        "batch_size": 128,
        "write_buffer_size": 1000,
        "max_workers": max(1, multiprocessing.cpu_count() * 3 // 4),
        "cache_size": 100,
    }

    # Detect device (MPS > CUDA > CPU)
    try:
        import torch

        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            config["device"] = "mps"
        elif torch.cuda.is_available():
            config["device"] = "cuda"
    except ImportError:
        pass

    # Detect RAM on macOS
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                config["ram_gb"] = int(result.stdout.strip()) / (1024**3)

            # Get chip name
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                config["chip_name"] = result.stdout.strip()
        except Exception:
            pass

    # Detect RAM on Linux
    elif platform.system() == "Linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        # MemTotal is in kB
                        kb = int(line.split()[1])
                        config["ram_gb"] = kb / (1024**2)
                        break

            # Get CPU info
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        config["chip_name"] = line.split(":")[1].strip()
                        break
        except Exception:
            pass

    # Calculate optimal settings based on detected hardware
    ram_gb = config["ram_gb"]

    # Batch size (for MPS/Apple Silicon)
    if config["device"] == "mps":
        if ram_gb >= 64:
            config["batch_size"] = 512
        elif ram_gb >= 32:
            config["batch_size"] = 384
        elif ram_gb >= 16:
            config["batch_size"] = 256
        else:
            config["batch_size"] = 128

    # Write buffer size
    if ram_gb >= 64:
        config["write_buffer_size"] = 10000
    elif ram_gb >= 32:
        config["write_buffer_size"] = 5000
    elif ram_gb >= 16:
        config["write_buffer_size"] = 2000
    else:
        config["write_buffer_size"] = 1000

    # Max workers (for Apple Silicon)
    cpu_cores = config["cpu_cores"]
    if config["cpu_arch"] == "arm" and config["system"] == "Darwin":
        if cpu_cores >= 16:
            config["max_workers"] = min(14, cpu_cores - 2)
        elif cpu_cores >= 10:
            config["max_workers"] = min(10, cpu_cores - 2)
        else:
            config["max_workers"] = max(4, cpu_cores - 1)

    # Cache size
    if ram_gb >= 64:
        config["cache_size"] = 10000
    elif ram_gb >= 32:
        config["cache_size"] = 5000
    elif ram_gb >= 16:
        config["cache_size"] = 1000
    else:
        config["cache_size"] = 100

    return config


def log_hardware_config() -> None:
    """Log detected hardware configuration at startup.

    This provides visibility into the auto-detected optimizations
    for Apple Silicon M4 Max and other hardware.
    """
    config = detect_hardware_config()

    logger.info("=" * 70)
    logger.info("Hardware Configuration Detected:")
    logger.info("-" * 70)
    logger.info(f"System:            {config['system']}")
    logger.info(f"CPU Architecture:  {config['cpu_arch']}")
    logger.info(f"Chip:              {config['chip_name']}")
    logger.info(f"CPU Cores:         {config['cpu_cores']}")
    logger.info(f"Total RAM:         {config['ram_gb']:.1f} GB")
    logger.info(f"Compute Device:    {config['device'].upper()}")
    logger.info("-" * 70)
    logger.info("Optimized Settings:")
    logger.info(f"  Batch Size:        {config['batch_size']}")
    logger.info(f"  Write Buffer:      {config['write_buffer_size']}")
    logger.info(f"  Max Workers:       {config['max_workers']}")
    logger.info(f"  Cache Size:        {config['cache_size']}")
    logger.info("-" * 70)

    # Show environment variable overrides if set
    overrides = []
    if os.environ.get("MCP_VECTOR_SEARCH_BATCH_SIZE"):
        overrides.append(f"BATCH_SIZE={os.environ['MCP_VECTOR_SEARCH_BATCH_SIZE']}")
    if os.environ.get("MCP_VECTOR_SEARCH_WRITE_BUFFER_SIZE"):
        overrides.append(
            f"WRITE_BUFFER_SIZE={os.environ['MCP_VECTOR_SEARCH_WRITE_BUFFER_SIZE']}"
        )
    if os.environ.get("MCP_VECTOR_SEARCH_MAX_WORKERS"):
        overrides.append(f"MAX_WORKERS={os.environ['MCP_VECTOR_SEARCH_MAX_WORKERS']}")
    if os.environ.get("MCP_VECTOR_SEARCH_CACHE_SIZE"):
        overrides.append(f"CACHE_SIZE={os.environ['MCP_VECTOR_SEARCH_CACHE_SIZE']}")

    if overrides:
        logger.info("Environment Overrides:")
        for override in overrides:
            logger.info(f"  {override}")
        logger.info("-" * 70)

    logger.info("=" * 70)
