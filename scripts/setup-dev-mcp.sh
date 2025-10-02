#!/bin/bash
# Setup script to configure mcp-vector-search to run from source like pipx installation

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"
BIN_DIR="$HOME/.local/bin"
SCRIPT_NAME="mcp-vector-search"

echo -e "${GREEN}MCP Vector Search Development Setup${NC}"
echo "======================================="
echo "Project directory: ${PROJECT_DIR}"
echo ""

# Step 1: Create/update virtual environment
echo -e "${YELLOW}Step 1: Setting up virtual environment...${NC}"
if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "${VENV_DIR}"
else
    echo "Virtual environment already exists."
fi

# Step 2: Activate venv and install dependencies
echo -e "${YELLOW}Step 2: Installing dependencies...${NC}"
source "${VENV_DIR}/bin/activate"

# Upgrade pip first
pip install --upgrade pip setuptools wheel

# Install the project in editable mode with all dependencies
pip install -e "${PROJECT_DIR}"

echo -e "${GREEN}✓ Dependencies installed${NC}"

# Step 3: Create wrapper script
echo -e "${YELLOW}Step 3: Creating wrapper script...${NC}"

# Create .local/bin if it doesn't exist
mkdir -p "${BIN_DIR}"

# Create wrapper script
cat > "${BIN_DIR}/${SCRIPT_NAME}" << 'EOF'
#!/bin/bash
# Wrapper script for mcp-vector-search running from source

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find the project directory (assuming standard setup)
PROJECT_DIR="__PROJECT_DIR__"
VENV_DIR="${PROJECT_DIR}/.venv"

# Check if virtual environment exists
if [ ! -d "${VENV_DIR}" ]; then
    echo "Error: Virtual environment not found at ${VENV_DIR}"
    echo "Please run the setup script first."
    exit 1
fi

# Activate virtual environment and run the command
source "${VENV_DIR}/bin/activate"
exec python -m mcp_vector_search.cli.main "$@"
EOF

# Replace placeholder with actual project directory
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS requires different sed syntax
    sed -i.bak "s|__PROJECT_DIR__|${PROJECT_DIR}|g" "${BIN_DIR}/${SCRIPT_NAME}"
    rm -f "${BIN_DIR}/${SCRIPT_NAME}.bak"
else
    sed -i "s|__PROJECT_DIR__|${PROJECT_DIR}|g" "${BIN_DIR}/${SCRIPT_NAME}"
fi

# Make wrapper script executable
chmod +x "${BIN_DIR}/${SCRIPT_NAME}"

echo -e "${GREEN}✓ Wrapper script created at ${BIN_DIR}/${SCRIPT_NAME}${NC}"

# Step 4: Add to PATH if necessary
if ! echo "$PATH" | grep -q "$HOME/.local/bin"; then
    echo -e "${YELLOW}Step 4: Adding ~/.local/bin to PATH...${NC}"

    # Detect shell and add to appropriate config file
    SHELL_CONFIG=""
    if [ -n "$ZSH_VERSION" ]; then
        SHELL_CONFIG="$HOME/.zshrc"
    elif [ -n "$BASH_VERSION" ]; then
        SHELL_CONFIG="$HOME/.bashrc"
        if [ -f "$HOME/.bash_profile" ]; then
            SHELL_CONFIG="$HOME/.bash_profile"
        fi
    fi

    if [ -n "$SHELL_CONFIG" ]; then
        echo "" >> "$SHELL_CONFIG"
        echo "# Added by mcp-vector-search setup" >> "$SHELL_CONFIG"
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_CONFIG"
        echo -e "${GREEN}✓ Added ~/.local/bin to PATH in ${SHELL_CONFIG}${NC}"
        echo -e "${YELLOW}  Please run: source ${SHELL_CONFIG}${NC}"
    else
        echo -e "${YELLOW}Please add the following to your shell configuration:${NC}"
        echo '  export PATH="$HOME/.local/bin:$PATH"'
    fi
else
    echo -e "${GREEN}✓ ~/.local/bin already in PATH${NC}"
fi

# Step 5: Test the installation
echo ""
echo -e "${YELLOW}Step 5: Testing installation...${NC}"

# Test if the command is available
if command -v mcp-vector-search &> /dev/null; then
    echo -e "${GREEN}✓ mcp-vector-search command is available${NC}"

    # Show version
    echo -n "Version: "
    mcp-vector-search --version || echo "(version command failed)"
else
    echo -e "${YELLOW}⚠ mcp-vector-search command not found in PATH${NC}"
    echo "You may need to:"
    echo "  1. Reload your shell configuration"
    echo "  2. Or run: export PATH=\"\$HOME/.local/bin:\$PATH\""
fi

# Step 6: Create MCP configuration
echo ""
echo -e "${YELLOW}Step 6: Configuring Claude Desktop MCP integration...${NC}"

# Create Claude Desktop config directory if it doesn't exist
CLAUDE_CONFIG_DIR="$HOME/Library/Application Support/Claude"
CLAUDE_CONFIG_FILE="${CLAUDE_CONFIG_DIR}/claude_desktop_config.json"

mkdir -p "${CLAUDE_CONFIG_DIR}"

# Check if config file exists
if [ -f "${CLAUDE_CONFIG_FILE}" ]; then
    echo "Claude Desktop config already exists. Here's the MCP configuration to add:"
else
    echo "Creating Claude Desktop config with MCP server configuration..."
    cat > "${CLAUDE_CONFIG_FILE}" << EOF
{
  "mcpServers": {
    "mcp-vector-search": {
      "command": "${BIN_DIR}/mcp-vector-search",
      "args": ["mcp"],
      "cwd": "${PROJECT_DIR}"
    }
  }
}
EOF
    echo -e "${GREEN}✓ Created Claude Desktop config at ${CLAUDE_CONFIG_FILE}${NC}"
fi

# Show the configuration
echo ""
echo "MCP Server Configuration for Claude Desktop:"
echo "--------------------------------------------"
cat << EOF
{
  "mcpServers": {
    "mcp-vector-search": {
      "command": "${BIN_DIR}/mcp-vector-search",
      "args": ["mcp"],
      "cwd": "${PROJECT_DIR}"
    }
  }
}
EOF

echo ""
echo -e "${GREEN}Setup Complete!${NC}"
echo ""
echo "Next steps:"
echo "1. If needed, reload your shell or run: export PATH=\"\$HOME/.local/bin:\$PATH\""
echo "2. Test the command: mcp-vector-search --version"
echo "3. Initialize a project: cd <your-project> && mcp-vector-search init"
echo "4. Restart Claude Desktop to load the MCP server"
echo "5. Verify MCP integration: claude mcp list"
echo ""
echo "The mcp-vector-search command is now available system-wide and will run from:"
echo "  ${PROJECT_DIR}"