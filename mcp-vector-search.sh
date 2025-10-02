#!/usr/bin/env bash

# MCP Vector Search Local Development Script
# This script sets up and runs mcp-vector-search from a local venv in this directory

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_DIR="$SCRIPT_DIR/.venv-mcp"

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the correct directory
if [ "$PWD" != "$SCRIPT_DIR" ]; then
    print_error "This script must be run from the mcp-vector-search directory"
    print_info "Please run: cd $SCRIPT_DIR && ./mcp-vector-search.sh"
    exit 1
fi

# Function to create/update virtual environment
setup_venv() {
    print_info "Setting up virtual environment at $VENV_DIR"

    # Create venv if it doesn't exist
    if [ ! -d "$VENV_DIR" ]; then
        print_info "Creating new virtual environment..."
        python3 -m venv "$VENV_DIR"
    else
        print_info "Virtual environment already exists"
    fi

    # Activate venv
    source "$VENV_DIR/bin/activate"

    # Upgrade pip
    print_info "Upgrading pip..."
    pip install --upgrade pip --quiet

    # Install the package in editable mode with dependencies
    print_info "Installing mcp-vector-search in editable mode..."
    pip install -e . --quiet

    print_info "Setup complete!"
}

# Function to run mcp-vector-search with arguments
run_mcp() {
    # Ensure venv exists and is activated
    if [ ! -d "$VENV_DIR" ]; then
        print_warn "Virtual environment not found. Setting up..."
        setup_venv
    fi

    # Activate venv if not already activated
    if [ -z "$VIRTUAL_ENV" ] || [ "$VIRTUAL_ENV" != "$VENV_DIR" ]; then
        source "$VENV_DIR/bin/activate"
    fi

    # Run mcp-vector-search with all passed arguments
    exec python -m mcp_vector_search "$@"
}

# Main script logic
if [ $# -eq 0 ]; then
    # No arguments - show help
    print_info "MCP Vector Search Local Runner"
    echo ""
    echo "Usage:"
    echo "  ./mcp-vector-search.sh [command] [options]"
    echo ""
    echo "Commands:"
    echo "  setup     - Set up the virtual environment"
    echo "  update    - Update dependencies and reinstall"
    echo "  clean     - Remove virtual environment"
    echo "  *         - Any mcp-vector-search command"
    echo ""
    echo "Examples:"
    echo "  ./mcp-vector-search.sh setup           # Set up venv"
    echo "  ./mcp-vector-search.sh init            # Initialize project"
    echo "  ./mcp-vector-search.sh index           # Index codebase"
    echo "  ./mcp-vector-search.sh search 'query'  # Search code"
    echo "  ./mcp-vector-search.sh mcp             # Start MCP server"
    echo ""
    exit 0
fi

# Handle special commands
case "$1" in
    setup)
        setup_venv
        print_info "You can now run: ./mcp-vector-search.sh [command]"
        ;;
    update)
        print_info "Updating dependencies..."
        setup_venv
        source "$VENV_DIR/bin/activate"
        pip install --upgrade -e . --quiet
        print_info "Update complete!"
        ;;
    clean)
        print_warn "Removing virtual environment at $VENV_DIR"
        rm -rf "$VENV_DIR"
        print_info "Virtual environment removed"
        ;;
    *)
        # Pass through to mcp-vector-search
        run_mcp "$@"
        ;;
esac