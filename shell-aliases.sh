#!/usr/bin/env bash
# MCP Vector Search - Shell Aliases and Functions
# Add this to your ~/.zshrc or ~/.bashrc, or source it directly

# Development build path
MCP_DEV_PATH="/Users/masa/Projects/managed/mcp-vector-search"

# Main alias - run mcp-vector-search from development build (mimics PyPI installation)
mcp-vector-search() {
    local current_dir="$(pwd)"
    cd "$MCP_DEV_PATH" && uv run mcp-vector-search --project-root "$current_dir" "$@"
}

# Convenience functions
mcp-install() {
    local current_dir="$(pwd)"
    cd "$MCP_DEV_PATH" && uv run mcp-vector-search install "$current_dir" "$@"
}

mcp-demo() {
    cd "$MCP_DEV_PATH" && uv run mcp-vector-search demo
}

mcp-dev() {
    local current_dir="$(pwd)"
    cd "$MCP_DEV_PATH" && uv run mcp-vector-search --project-root "$current_dir" "$@"
}

mcp-setup() {
    local current_dir="$(pwd)"
    cd "$MCP_DEV_PATH" && uv run mcp-vector-search install "$current_dir" "$@"
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
