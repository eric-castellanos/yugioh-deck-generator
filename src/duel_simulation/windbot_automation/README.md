# WindBot Automation for YGOPro

This directory contains a complete automation system for YGOPro duel simulation using WindBot. It enables automated .ydk vs .ydk duels for ML/AI research and deck evaluation.

## Overview

The system consists of three main components:

1. **WindBot Wrapper** (`windbot_wrapper.py`) - Python interface to the WindBot C# application
2. **YGOPro Server** (`ygopro_server.py`) - Server management and duel monitoring  
3. **Integration Layer** (`automation_integration.py`) - Complete automation combining server and bots

## Features

- **Automated Duel Simulation**: Run .ydk vs .ydk duels automatically
- **Tournament Support**: Round-robin and other tournament formats
- **Result Analysis**: Statistical analysis of duel outcomes
- **Replay Support**: Save and manage replay files
- **Error Handling**: Robust error detection and recovery
- **Configurable**: Flexible configuration for different use cases

## Installation

### Prerequisites

1. **EDOPro** - YGOPro client/server
   ```bash
   # Download from https://projectignis.github.io/download.html
   # Extract to /home/ecast229/Applications/EDOPro (or configure path)
   ```

2. **Mono** - For running C# applications on Linux
   ```bash
   sudo apt-get update
   sudo apt-get install mono-complete
   ```

3. **Python Dependencies**
   ```bash
   pip install dataclasses pathlib
   ```

### Setup WindBot

Run the setup script to download and build WindBot:

```bash
./setup_windbot.sh
```

This will:
- Clone the WindBot repository
- Build WindBot using Mono/MSBuild
- Copy necessary files (cards.cdb, etc.)
- Test the installation

## Usage

### Basic Usage

```python
from windbot_automation import YGOProAutomation, AutomationConfig

# Configure automation
config = AutomationConfig(
    edopro_path="/path/to/edopro",
    debug=True,
    save_replays=True
)

# Initialize automation system
automation = YGOProAutomation(config)

# Run a single duel
result = automation.simulate_single_duel(
    deck1_path="deck1.ydk",
    deck2_path="deck2.ydk",
    deck1_name="BlueEyes",
    deck2_name="DarkMagician"
)

print(f"Winner: {result.deck1_name if result.winner == 0 else result.deck2_name}")
print(f"Turns: {result.turns}")
print(f"Duration: {result.duration:.2f}s")

# Cleanup
automation.cleanup()
```

### Tournament Simulation

```python
# Run a tournament between multiple decks
deck_paths = [
    "deck1.ydk",
    "deck2.ydk", 
    "deck3.ydk"
]

results = automation.simulate_tournament(
    deck_paths=deck_paths,
    rounds=3  # 3 rounds per matchup
)

# Analyze results
analysis = automation.analyze_results(results)
print(f"Tournament completed: {analysis['total_duels']} duels")

# Save results
automation.save_results(results, "tournament_results.json")
```

### Integration with ML Pipeline

```python
# Use in ML evaluation pipeline
def evaluate_deck_performance(deck_path, opponent_decks):
    results = []
    
    for opponent in opponent_decks:
        result = automation.simulate_single_duel(
            deck1_path=deck_path,
            deck2_path=opponent,
            deck1_name="TestDeck",
            deck2_name=f"Opponent_{Path(opponent).stem}"
        )
        results.append(result)
    
    # Calculate win rate
    wins = sum(1 for r in results if r.winner == 0 and r.error is None)
    total = len([r for r in results if r.error is None])
    win_rate = wins / total if total > 0 else 0
    
    return win_rate, results

# Example usage in deck generation pipeline
generated_deck = "generated_deck.ydk"
meta_decks = ["meta1.ydk", "meta2.ydk", "meta3.ydk"]

win_rate, duel_results = evaluate_deck_performance(generated_deck, meta_decks)
print(f"Generated deck win rate: {win_rate:.2%}")
```

## Configuration

### AutomationConfig Options

```python
config = AutomationConfig(
    edopro_path="/path/to/edopro",        # EDOPro installation path
    windbot_path="/path/to/windbot.exe",  # WindBot executable (auto-detected if None)
    work_dir="/tmp/automation",           # Working directory for temp files
    server_port=7911,                     # YGOPro server port
    duel_timeout=300,                     # Duel timeout in seconds
    debug=False,                          # Enable debug logging
    save_replays=True                     # Save replay files
)
```

### ServerConfig Options

```python
server_config = ServerConfig(
    port=7911,           # Server port
    host="127.0.0.1",    # Server host
    banlist=0,           # Banlist (0=TCG, 1=OCG)
    rule=5,              # Master Rule version
    start_lp=8000,       # Starting Life Points
    start_hand=5,        # Starting hand size
    time_limit=300       # Time limit per duel
)
```

## Architecture

### Component Interaction

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ML Pipeline   │ -> │ Automation      │ -> │ YGOPro Server   │
│                 │    │ Integration     │    │ Manager         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                v                       v
                       ┌─────────────────┐    ┌─────────────────┐
                       │ WindBot         │ -> │ Duel Monitoring │
                       │ Wrapper         │    │ & Results       │
                       └─────────────────┘    └─────────────────┘
                                │                       │
                                v                       v
                       ┌─────────────────┐    ┌─────────────────┐
                       │ WindBot AI      │    │ Replay Files    │
                       │ (C# Process)    │    │ & Statistics    │
                       └─────────────────┘    └─────────────────┘
```

### Data Flow

1. **Input**: .ydk deck files
2. **Processing**: WindBot automation with YGOPro server
3. **Monitoring**: Real-time duel progress tracking
4. **Output**: DuelResult objects with statistics
5. **Analysis**: Win rates, turn counts, duration metrics
6. **Storage**: JSON results and replay files

## Limitations and Considerations

### Current Limitations

1. **Server Dependency**: Requires a working YGOPro server implementation
2. **WindBot AI**: Limited to pre-programmed AI strategies
3. **Platform**: Primarily tested on Linux (may work on Windows with modifications)
4. **Performance**: Sequential duel execution (no parallel duels yet)

### Planned Improvements

1. **Parallel Execution**: Run multiple duels simultaneously
2. **Custom AI**: Integration with custom AI strategies
3. **Advanced Monitoring**: Real-time game state analysis
4. **Cloud Support**: Deployment on cloud infrastructure
5. **Web Interface**: Browser-based tournament management

## Troubleshooting

### Common Issues

1. **WindBot not found**
   ```bash
   # Run the setup script
   ./setup_windbot.sh
   ```

2. **Mono not installed**
   ```bash
   sudo apt-get install mono-complete
   ```

3. **EDOPro path incorrect**
   ```python
   # Update config with correct path
   config.edopro_path = "/correct/path/to/edopro"
   ```

4. **Port conflicts**
   ```python
   # Use a different port
   config.server_port = 7912
   ```

5. **Permission errors**
   ```bash
   # Ensure directories are writable
   chmod +w /path/to/work/dir
   ```

### Debug Mode

Enable debug mode for detailed logging:

```python
config = AutomationConfig(debug=True)
```

This will show:
- Detailed command execution
- Server startup/shutdown logs
- WindBot communication
- Duel progress monitoring

## Performance Metrics

Typical performance on a modern system:

- **Duel Duration**: 30-180 seconds per duel
- **Memory Usage**: ~500MB per WindBot instance
- **CPU Usage**: Moderate (depends on AI complexity)
- **Disk Space**: ~1MB per replay file

## Security Considerations

- **Local Network**: Server binds to localhost by default
- **Temporary Files**: Cleaned up automatically
- **Process Management**: Proper cleanup of subprocesses
- **File Permissions**: Restrictive permissions on work directories

## Contributing

To contribute to this automation system:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

### Development Setup

```bash
# Clone the repository
git clone <repository-url>
cd yugioh-deck-generator

# Setup development environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run tests
python -m pytest src/duel_simulation/windbot_automation/tests/
```

## License

This automation system is part of the YuGiOh Deck Generator project and follows the same license terms.

## Contact

For questions or support regarding this automation system, please open an issue in the main repository.
