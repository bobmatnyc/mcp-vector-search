#!/usr/bin/env bash
# MCP Vector Search - Shell Aliases and Functions
# Add this to your ~/.zshrc or ~/.bashrc, or source it directly

# Development build path
MCP_DEV_PATH="/Users/masa/Projects/managed/mcp-vector-search"

# Main alias - run mcp-vector-search from development build (mimics PyPI installation)
mcp-vector-search() {
    # Use the development build but run it in the current directory context
    # This avoids directory switching and operates on the current directory as project root
    "$MCP_DEV_PATH/.venv/bin/python3" -m mcp_vector_search.cli.main "$@"
}

# Convenience functions
mcp-install() {
    # Install in current directory using development build
    "$MCP_DEV_PATH/.venv/bin/python3" -m mcp_vector_search.cli.main install . "$@"
}

mcp-demo() {
    # Run demo from development build
    "$MCP_DEV_PATH/.venv/bin/python3" -m mcp_vector_search.cli.main demo
}

mcp-dev() {
    # Development alias - same as main function
    "$MCP_DEV_PATH/.venv/bin/python3" -m mcp_vector_search.cli.main "$@"
}

mcp-setup() {
    # Setup alias - same as mcp-install
    "$MCP_DEV_PATH/.venv/bin/python3" -m mcp_vector_search.cli.main install . "$@"
}

# Helper function to show available commands
mcp-help() {
    echo "🚀 MCP Vector Search - Development Build Commands"
    echo "================================================="
    echo
    echo "Main Commands:"
    echo "  mcp-vector-search [args...]     # Run mcp-vector-search from dev build"
    echo
    echo "Project Setup:"
    echo "  mcp-install [options...]        # Install in current directory"
    echo "  mcp-setup [options...]          # Alias for mcp-install"
    echo "  mcp-demo                        # Run installation demo"
    echo
    echo "Development:"
    echo "  mcp-dev [args...]               # Run from dev environment"
    echo "  mcp-help                        # Show this help"
    echo
    echo "Examples:"
    echo "  mcp-vector-search --help        # Show help"
    echo "  mcp-vector-search search 'query' # Search current project"
    echo "  mcp-vector-search status main   # Check project status"
    echo "  mcp-install                     # Install in current directory"
    echo "  mcp-install --no-mcp            # Install without MCP integration"
    echo "  mcp-demo                        # See installation demo"
    echo
}

# Show confirmation that aliases are loaded
echo "✅ MCP Vector Search aliases loaded! Run 'mcp-help' for available commands."
