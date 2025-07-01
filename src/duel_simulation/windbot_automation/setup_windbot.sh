#!/bin/bash
"""
WindBot Setup Script

This script downloads, builds, and configures WindBot for automated duel simulation.
"""

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
WINDBOT_DIR="$PROJECT_ROOT/Applications/WindBot"

echo "Setting up WindBot for automated duel simulation..."
echo "Project root: $PROJECT_ROOT"
echo "WindBot directory: $WINDBOT_DIR"

# Create WindBot directory
mkdir -p "$WINDBOT_DIR"
cd "$WINDBOT_DIR"

# Check if WindBot already exists
if [ -f "WindBot.exe" ]; then
    echo "WindBot already exists at $WINDBOT_DIR/WindBot.exe"
    exit 0
fi

# Install dependencies
echo "Installing dependencies..."

# Install Mono (for running C# applications on Linux)
if ! command -v mono &> /dev/null; then
    echo "Installing Mono..."
    sudo apt-get update
    sudo apt-get install -y mono-complete
else
    echo "Mono already installed"
fi

# Install git if not present
if ! command -v git &> /dev/null; then
    echo "Installing git..."
    sudo apt-get install -y git
fi

# Install MSBuild or build tools
if ! command -v msbuild &> /dev/null; then
    echo "Installing MSBuild..."
    sudo apt-get install -y msbuild
fi

# Clone WindBot repository
echo "Cloning WindBot repository..."
if [ ! -d "windbot-source" ]; then
    git clone https://github.com/edo9300/windbot.git windbot-source
else
    echo "WindBot source already cloned"
fi

cd windbot-source

# Build WindBot
echo "Building WindBot..."
if command -v msbuild &> /dev/null; then
    # Use MSBuild if available
    msbuild WindBot.sln /p:Configuration=Release /p:Platform="Any CPU"
    
    # Copy built files
    if [ -f "bin/Release/WindBot.exe" ]; then
        cp -r bin/Release/* ../
        echo "WindBot built successfully using MSBuild"
    else
        echo "MSBuild failed, trying xbuild..."
        xbuild WindBot.sln /p:Configuration=Release
        if [ -f "bin/Release/WindBot.exe" ]; then
            cp -r bin/Release/* ../
            echo "WindBot built successfully using xbuild"
        else
            echo "Build failed with xbuild as well"
            exit 1
        fi
    fi
elif command -v xbuild &> /dev/null; then
    # Use xbuild as fallback
    xbuild WindBot.sln /p:Configuration=Release
    
    # Copy built files
    if [ -f "bin/Release/WindBot.exe" ]; then
        cp -r bin/Release/* ../
        echo "WindBot built successfully using xbuild"
    else
        echo "Build failed"
        exit 1
    fi
else
    echo "Neither MSBuild nor xbuild found. Cannot build WindBot."
    echo "You may need to build it manually on Windows and copy the binaries."
    exit 1
fi

cd ..

# Copy cards database from EDOPro
echo "Setting up cards database..."
EDOPRO_PATH="/home/ecast229/Applications/EDOPro"
if [ -f "$EDOPRO_PATH/cards.cdb" ]; then
    cp "$EDOPRO_PATH/cards.cdb" .
    echo "Copied cards.cdb from EDOPro"
elif [ -f "$EDOPRO_PATH/databases/cards.cdb" ]; then
    cp "$EDOPRO_PATH/databases/cards.cdb" .
    echo "Copied cards.cdb from EDOPro databases folder"
else
    echo "Warning: cards.cdb not found in EDOPro. WindBot may not work correctly."
    echo "You may need to download cards.cdb manually."
fi

# Create decks directory
mkdir -p Decks

# Create default configuration
echo "Creating default configuration..."
cat > config.txt << 'EOF'
# WindBot Configuration
# This file can be used to set default parameters

Name=WindBot
Host=127.0.0.1
Port=7911
Dialog=default
Chat=false
Debug=false
EOF

# Test WindBot
echo "Testing WindBot..."
if mono WindBot.exe --help &> /dev/null; then
    echo "WindBot test successful!"
elif timeout 5 mono WindBot.exe &> /dev/null; then
    echo "WindBot appears to be working (timeout expected for test)"
else
    echo "Warning: WindBot test failed. It may still work in actual usage."
fi

echo ""
echo "WindBot setup complete!"
echo "Location: $WINDBOT_DIR"
echo "Executable: $WINDBOT_DIR/WindBot.exe"
echo ""
echo "To run WindBot manually:"
echo "  cd $WINDBOT_DIR"
echo "  mono WindBot.exe Name=TestBot Deck=Blue-Eyes Host=127.0.0.1 Port=7911"
echo ""
echo "The Python wrapper will use this installation automatically."
