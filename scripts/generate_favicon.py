#!/usr/bin/env python3
"""Generate favicon.ico for code graph visualization.

Creates a simple network graph icon with connected nodes using PIL.
Fast, lightweight alternative to AI-generated favicons.
"""

from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ImportError:
    print("Error: Pillow is required. Install with: pip install pillow")
    import sys

    sys.exit(1)


def create_network_graph_icon(size: int = 32) -> Image.Image:
    """Create a favicon image with a simple network graph design.

    Design: Three nodes arranged in a triangle with a center node,
    all connected with edges. Uses GitHub blue color scheme.

    Args:
        size: Size of the square icon (width and height)

    Returns:
        PIL Image object with RGBA mode
    """
    # Colors matching GitHub dark theme
    bg_color = (13, 17, 23, 255)  # #0d1117 - GitHub dark background
    node_color = (88, 166, 255, 255)  # #58a6ff - GitHub blue
    node_highlight = (108, 186, 255, 255)  # Lighter blue for centers
    edge_color = (88, 166, 255, 180)  # Semi-transparent blue for edges

    # Create image with dark background
    img = Image.new("RGBA", (size, size), bg_color)
    draw = ImageDraw.Draw(img)

    # Scale all dimensions based on size
    scale = size / 32
    padding = int(6 * scale)
    node_radius = max(2, int(3 * scale))
    edge_width = max(1, int(1.5 * scale))

    # Define node positions - triangle with center node
    nodes = [
        (padding + int(8 * scale), padding + int(4 * scale)),  # Top-left
        (size - padding - int(8 * scale), padding + int(4 * scale)),  # Top-right
        (size // 2, size - padding - int(4 * scale)),  # Bottom-center
        (size // 2, size // 2 - int(2 * scale)),  # Center node
    ]

    # Define edges (node indices to connect)
    edges = [
        (0, 3),  # Top-left to center
        (1, 3),  # Top-right to center
        (2, 3),  # Bottom to center
        (0, 1),  # Top-left to top-right
    ]

    # Draw edges first (behind nodes)
    for start_idx, end_idx in edges:
        start = nodes[start_idx]
        end = nodes[end_idx]
        draw.line([start, end], fill=edge_color, width=edge_width)

    # Draw nodes
    for x, y in nodes:
        # Outer circle (main node color)
        draw.ellipse(
            [
                (x - node_radius - 1, y - node_radius - 1),
                (x + node_radius + 1, y + node_radius + 1),
            ],
            fill=node_color,
            outline=node_color,
        )
        # Inner circle (highlight)
        if node_radius > 1:
            inner_radius = node_radius - 1
            draw.ellipse(
                [
                    (x - inner_radius, y - inner_radius),
                    (x + inner_radius, y + inner_radius),
                ],
                fill=node_highlight,
                outline=node_highlight,
            )

    return img


def main():
    """Generate favicon.ico with multiple sizes embedded."""
    output_dir = Path(
        "/Users/masa/Projects/mcp-vector-search/.mcp-vector-search/visualization"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Generating Favicon for Code Graph Visualization")
    print("=" * 70)

    # Create icons in different sizes for .ico file
    sizes = [16, 32, 48, 64]
    print(f"\nCreating {len(sizes)} icon sizes: {', '.join(f'{s}x{s}' for s in sizes)}")

    images = [create_network_graph_icon(size) for size in sizes]

    # Save as .ico file with multiple sizes embedded
    ico_path = output_dir / "favicon.ico"
    images[0].save(
        ico_path,
        format="ICO",
        sizes=[(img.width, img.height) for img in images],
        append_images=images[1:],
    )

    print(f"\n✅ Favicon created: {ico_path}")
    print(f"   Embedded sizes: {', '.join(f'{s}x{s}' for s in sizes)}")

    # Also create a larger PNG preview for verification
    preview_size = 128
    preview_path = output_dir / "favicon_preview.png"
    create_network_graph_icon(preview_size).save(preview_path, format="PNG")
    print(f"   Preview created: {preview_path} ({preview_size}x{preview_size})")

    print("\n✅ Done! Favicon ready for use in visualization.")
    print(f"   The favicon.ico file contains {len(sizes)} embedded sizes")
    print("   for optimal display across different contexts.")


if __name__ == "__main__":
    main()
