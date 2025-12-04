#!/bin/bash

# Exit on error
set -e

echo "Initializing submodules..."
git submodule update --init --recursive

# Fix missing __init__.py in LIBERO
if [ ! -f "LIBERO/libero/__init__.py" ]; then
    echo "Creating missing __init__.py in LIBERO/libero..."
    touch LIBERO/libero/__init__.py
fi

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "uv could not be found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
fi

echo "Creating environment and installing dependencies with uv..."
uv sync --python 3.10

# Install the current package in editable mode explicitly to ensure robomimic is found
uv pip install -e .

echo "Installation complete!"
echo "Activate the environment with: source .venv/bin/activate"