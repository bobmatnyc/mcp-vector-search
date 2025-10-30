#!/usr/bin/env python3
"""Generate favicon images using Stable Diffusion XL.

This script creates a graph visualization icon for the mcp-vector-search project
using SDXL with Apple Silicon MPS acceleration.
"""

import sys
from pathlib import Path
from typing import Final

try:
    import torch
    from diffusers import DiffusionPipeline
    from PIL import Image
except ImportError as e:
    print(f"Error: Missing required library: {e}")
    print("Install with: pip install diffusers torch pillow transformers accelerate")
    sys.exit(1)


# Configuration
MODEL_ID: Final[str] = "stabilityai/stable-diffusion-xl-base-1.0"
OUTPUT_DIR: Final[Path] = Path(__file__).parent.parent / "src" / "mcp_vector_search" / "visualization"
ORIGINAL_SIZE: Final[int] = 1024
FAVICON_SIZES: Final[list[int]] = [512, 256, 128, 64, 32, 16]

# Generation parameters
PROMPT: Final[str] = (
    "minimalist graph visualization icon, connected nodes and links, "
    "network diagram, semantic search symbol, clean geometric shapes, "
    "tech logo style, flat design, simple, professional, white background, "
    "vector art style"
)

NEGATIVE_PROMPT: Final[str] = (
    "text, letters, words, realistic, photographic, gradients, shadows, "
    "3d, complex, busy, detailed, watermark, blurry, low quality"
)

# Seeds for variations
SEEDS: Final[list[int]] = [42, 123, 456]


def check_mps_availability() -> str:
    """Check if MPS is available and return the device string.

    Returns:
        Device string ("mps" or "cpu")
    """
    if torch.backends.mps.is_available():
        if torch.backends.mps.is_built():
            print("✓ MPS (Metal Performance Shaders) acceleration available")
            return "mps"
        else:
            print("⚠ MPS not built, falling back to CPU")
            return "cpu"
    else:
        print("⚠ MPS not available, falling back to CPU")
        return "cpu"


def load_sdxl_pipeline(device: str) -> DiffusionPipeline:
    """Load SDXL pipeline with appropriate settings.

    Args:
        device: Device to use ("mps" or "cpu")

    Returns:
        Loaded diffusion pipeline

    Raises:
        RuntimeError: If pipeline loading fails
    """
    print(f"\nLoading SDXL model ({MODEL_ID})...")
    print("This may take a minute if it's the first run...")

    try:
        # Load pipeline with float16 for memory efficiency
        pipeline = DiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if device == "mps" else torch.float32,
            use_safetensors=True,
            variant="fp16" if device == "mps" else None,
        )

        # Move to device
        pipeline = pipeline.to(device)

        # Enable attention slicing for memory efficiency
        pipeline.enable_attention_slicing()

        print("✓ Model loaded successfully")
        return pipeline

    except Exception as e:
        raise RuntimeError(f"Failed to load SDXL pipeline: {e}") from e


def generate_image(
    pipeline: DiffusionPipeline,
    seed: int,
    device: str
) -> Image.Image:
    """Generate a single image using SDXL.

    Args:
        pipeline: Loaded diffusion pipeline
        seed: Random seed for reproducibility
        device: Device being used

    Returns:
        Generated PIL Image
    """
    print(f"\nGenerating image with seed {seed}...")

    # Set random seed
    generator = torch.Generator(device=device).manual_seed(seed)

    # Generate image
    result = pipeline(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=40,
        guidance_scale=7.5,
        generator=generator,
        height=ORIGINAL_SIZE,
        width=ORIGINAL_SIZE,
    )

    print("✓ Image generated")
    return result.images[0]


def create_favicon_sizes(
    image: Image.Image,
    output_dir: Path,
    variant_suffix: str = ""
) -> list[Path]:
    """Resize image to all favicon sizes and save.

    Args:
        image: Source image at original resolution
        output_dir: Directory to save resized images
        variant_suffix: Optional suffix for filename (e.g., "-v1")

    Returns:
        List of saved file paths
    """
    print(f"\nCreating favicon sizes{' (' + variant_suffix + ')' if variant_suffix else ''}...")

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files: list[Path] = []

    # Save original
    original_path = output_dir / f"favicon{variant_suffix}-{ORIGINAL_SIZE}.png"
    image.save(original_path, "PNG", quality=100)
    saved_files.append(original_path)
    print(f"  ✓ Saved {ORIGINAL_SIZE}x{ORIGINAL_SIZE}")

    # Create and save each size
    for size in FAVICON_SIZES:
        resized = image.resize((size, size), Image.LANCZOS)
        filename = output_dir / f"favicon{variant_suffix}-{size}.png"
        resized.save(filename, "PNG", quality=100)
        saved_files.append(filename)
        print(f"  ✓ Saved {size}x{size}")

    return saved_files


def create_ico_file(
    image: Image.Image,
    output_dir: Path,
    variant_suffix: str = ""
) -> Path:
    """Create .ico file with multiple sizes embedded.

    Args:
        image: Source image at original resolution
        output_dir: Directory to save .ico file
        variant_suffix: Optional suffix for filename

    Returns:
        Path to saved .ico file
    """
    print(f"\nCreating .ico file{' (' + variant_suffix + ')' if variant_suffix else ''}...")

    # Create different sizes for .ico
    sizes = [(16, 16), (32, 32), (64, 64), (128, 128), (256, 256)]
    resized_images = [image.resize(size, Image.LANCZOS) for size in sizes]

    ico_path = output_dir / f"favicon{variant_suffix}.ico"
    resized_images[0].save(
        ico_path,
        format="ICO",
        sizes=sizes,
        append_images=resized_images[1:]
    )

    print(f"  ✓ Saved {ico_path.name} with {len(sizes)} embedded sizes")
    return ico_path


def main() -> int:
    """Main execution function.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    print("=" * 70)
    print("MCP Vector Search - Favicon Generator")
    print("=" * 70)

    try:
        # Check device availability
        device = check_mps_availability()

        # Load pipeline
        pipeline = load_sdxl_pipeline(device)

        # Generate multiple variations
        print(f"\nGenerating {len(SEEDS)} variations...")

        all_saved_files: list[Path] = []

        for i, seed in enumerate(SEEDS, 1):
            variant_suffix = f"-v{i}"

            # Generate image
            image = generate_image(pipeline, seed, device)

            # Save all sizes
            saved_files = create_favicon_sizes(image, OUTPUT_DIR, variant_suffix)
            all_saved_files.extend(saved_files)

            # Create .ico file
            ico_file = create_ico_file(image, OUTPUT_DIR, variant_suffix)
            all_saved_files.append(ico_file)

        # Summary
        print("\n" + "=" * 70)
        print("✓ Generation complete!")
        print("=" * 70)
        print(f"\nGenerated {len(SEEDS)} variations with {len(FAVICON_SIZES) + 2} files each")
        print(f"Total files created: {len(all_saved_files)}")
        print(f"\nOutput directory: {OUTPUT_DIR}")
        print("\nVariations:")
        for i in range(1, len(SEEDS) + 1):
            print(f"  - favicon-v{i}-*.png (all sizes)")
            print(f"  - favicon-v{i}.ico")

        print("\n💡 Tip: Review all variations and choose your favorite!")
        print("   Then rename it to 'favicon.png' and 'favicon.ico' for use.")

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠ Generation interrupted by user")
        return 1

    except Exception as e:
        print(f"\n\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    finally:
        # Cleanup
        if 'pipeline' in locals():
            print("\nCleaning up...")
            del pipeline
            if device == "mps":
                torch.mps.empty_cache()
            else:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == "__main__":
    sys.exit(main())
