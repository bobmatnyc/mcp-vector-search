#!/bin/bash
# Setup alias for mcp-vector-search development

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "# MCP Vector Search Development Alias"
echo "# Add this to your ~/.zshrc or ~/.bashrc:"
echo ""
echo "alias mcp-dev='${PROJECT_DIR}/scripts/mcp-dev'"
echo ""
echo "# Or create a function for better integration:"
echo "mcp-vector-search() {"
echo "    ${PROJECT_DIR}/scripts/mcp-dev \"\$@\""
echo "}"
echo ""
echo "# After adding to your shell config, reload with:"
echo "#   source ~/.zshrc  (or ~/.bashrc)"