#!/bin/bash
set -e

# Install Poetry if not already installed
if ! command -v poetry &> /dev/null; then
  echo "Installing Poetry..."
  curl -sSL https://install.python-poetry.org | python3 -
else
  echo "Poetry already installed"
fi

# Ensure Poetry is on PATH for current shell
export PATH="$HOME/.local/bin:$PATH"
echo "Poetry version: $(poetry --version)"