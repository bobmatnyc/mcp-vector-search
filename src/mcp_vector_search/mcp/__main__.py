"""Entry point for running the MCP server."""

import asyncio
import sys
from pathlib import Path

from .server import run_mcp_server


def main():
    """Main entry point for the MCP server."""
    # Allow specifying project root as command line argument
    project_root = Path(sys.argv[1]) if len(sys.argv) > 1 else None

    try:
        asyncio.run(run_mcp_server(project_root))
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"MCP server error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
