#!/bin/bash
# Simple wrapper for the Python Homebrew formula updater
# Called by Makefile targets

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Pass all arguments to the Python script
exec python3 "$SCRIPT_DIR/update_homebrew_formula.py" "$@"
