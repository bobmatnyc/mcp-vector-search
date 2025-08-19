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

# Installer alias
alias mcp-vector-search-install="$MCP_DEV_PATH/mcp-vector-search-install"

# Convenience functions
mcp-install() {
    "$MCP_DEV_PATH/mcp-vector-search-install" "$@"
}

mcp-demo() {
    "$MCP_DEV_PATH/mcp-vector-search-install" demo
}

mcp-dev() {
    local current_dir="$(pwd)"
    cd "$MCP_DEV_PATH" && uv run mcp-vector-search --project-root "$current_dir" "$@"
}

mcp-setup() {
    "$MCP_DEV_PATH/mcp-vector-search-install" "$@"
}

# Helper function to show available commands
mcp-help() {
    echo "🚀 MCP Vector Search - Development Build Commands"
    echo "================================================="
    echo
    echo "Main Commands:"
    echo "  mcp-vector-search [args...]     # Run mcp-vector-search from dev build"
    echo "  mcp-vector-search-install       # Run the installer script"
    echo
    echo "Project Setup:"
    echo "  mcp-install [dir] [options...]  # Install in project (default: current dir)"
    echo "  mcp-setup [dir] [options...]    # Alias for mcp-install"
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
    echo "  mcp-install ~/my-project        # Install in specific directory"
    echo "  mcp-demo                        # See installation demo"
    echo
}

# Show confirmation that aliases are loaded
echo "✅ MCP Vector Search aliases loaded! Run 'mcp-help' for available commands."
