# Favicon Generation Script

Generate professional graph visualization favicons for the mcp-vector-search project using Stable Diffusion XL.

## Prerequisites

- Python 3.10+
- Apple Silicon Mac (for MPS acceleration) or CPU fallback
- SDXL model downloaded (~13GB)

## Required Libraries

```bash
pip install diffusers torch pillow transformers accelerate
```

## Usage

### Basic Generation

Run the script to generate 3 variations:

```bash
cd /Users/masa/Projects/mcp-vector-search
python scripts/generate_favicon.py
```

### What It Does

1. **Loads SDXL Model**: Uses `stabilityai/stable-diffusion-xl-base-1.0` with MPS acceleration
2. **Generates Images**: Creates 3 variations (seeds: 42, 123, 456) at 1024x1024 resolution
3. **Creates Favicon Sizes**: Resizes each variation to:
   - 1024x1024 (original)
   - 512x512
   - 256x256
   - 128x128
   - 64x64
   - 32x32
   - 16x16
4. **Creates .ico Files**: Embeds multiple sizes in .ico format

### Output

Files are saved to: `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/visualization/`

```
favicon-v1-1024.png
favicon-v1-512.png
favicon-v1-256.png
... (all sizes for v1)
favicon-v1.ico

favicon-v2-1024.png
... (all sizes for v2)
favicon-v2.ico

favicon-v3-1024.png
... (all sizes for v3)
favicon-v3.ico
```

### Choosing Your Favorite

1. Review all 3 variations
2. Choose your favorite (e.g., `favicon-v2-*.png`)
3. Rename/copy to `favicon.png` and `favicon.ico` for production use

## Generation Parameters

- **Prompt**: Minimalist graph visualization, connected nodes, network diagram, semantic search symbol
- **Negative Prompt**: No text, letters, realistic photos, gradients, shadows
- **Steps**: 40 inference steps
- **Guidance Scale**: 7.5
- **Resolution**: 1024x1024 (SDXL native)

## Performance

- **Runtime**: ~2-3 minutes on Apple Silicon (M1/M2/M3)
- **Memory**: ~8GB with float16 precision
- **Device**: MPS (GPU) or CPU fallback

## Customization

Edit the script to customize:

- `PROMPT`: Change the image description
- `NEGATIVE_PROMPT`: Add unwanted elements
- `SEEDS`: Generate different variations (line 39)
- `FAVICON_SIZES`: Add/remove output sizes (line 29)

## Troubleshooting

### MPS Not Available

If MPS fails, the script automatically falls back to CPU (slower but works).

### Out of Memory

Reduce batch size or use CPU:
```python
# In the script, change:
torch_dtype=torch.float32  # Use full precision
```

### Model Not Found

Download SDXL manually:
```bash
python -c "from diffusers import DiffusionPipeline; DiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0')"
```

## Example Output

The script generates clean, minimalist icons showing:
- Connected nodes (graph vertices)
- Network links (edges)
- Semantic search visualization
- Tech logo aesthetic
- Flat design, white background

Perfect for browser tabs, app icons, and documentation!
