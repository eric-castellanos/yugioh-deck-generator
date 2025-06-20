#!/bin/bash
set -e

# Ensure Poetry is on PATH for current shell, included for running locally
export PATH="$HOME/.local/bin:$PATH"

# Install Poetry if not already installed
if ! command -v poetry &> /dev/null; then
  echo "Installing Poetry..."
  wget -qO- https://install.python-poetry.org | POETRY_VERSION=1.8.2 python3 -
else
  echo "Poetry already installed"
fi

echo "Poetry version: $(poetry --version)"