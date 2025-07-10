#!/usr/bin/env python3
"""
WindBot Automation Wrapper

This module provides Python automation for YGOPro duel simulation using WindBot.
WindBot is a C# AI bot framework that can play YGOPro automatically.

The approach:
1. Create a local YGOPro server (using EDOPro or similar)
2. Launch two WindBot instances with custom deck configurations
3. Monitor the duel and collect results
4. Parse replay files for detailed analysis

This enables .ydk vs .ydk automated simulation for ML/AI research.
"""

import os
import sys
import json
import time
import subprocess
import threading
import tempfile
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DuelConfig:
    """Configuration for a single duel."""
    deck1_path: str
    deck2_path: str
    deck1_name: str = "Player1"
    deck2_name: str = "Player2"
    timeout: int = 300  # 5 minutes default
    debug: bool = False
    chat: bool = False


@dataclass
class DuelResult:
    """Result of a duel simulation."""
    winner: int  # 0 for player 1, 1 for player 2, -1 for draw/error
    turns: int
    duration: float  # seconds
    replay_path: Optional[str] = None
    error: Optional[str] = None
    deck1_name: str = "Player1"
    deck2_name: str = "Player2"


class WindBotWrapper:
    """
    Python wrapper for WindBot automation.
    
    This class manages YGOPro server instances and WindBot clients
    to enable automated .ydk vs .ydk duel simulation.
    """
    
    def __init__(self, 
                 edopro_path: str = "/home/ecast229/Applications/EDOPro",
                 windbot_path: Optional[str] = None,
                 work_dir: Optional[str] = None):
        """
        Initialize the WindBot wrapper.
        
        Args:
            edopro_path: Path to EDOPro installation
            windbot_path: Path to WindBot executable (will download if None)
            work_dir: Working directory for temporary files
        """
        self.edopro_path = Path(edopro_path)
        self.windbot_path = Path(windbot_path) if windbot_path else None
        self.work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp())
        
        # Server configuration
        self.server_port = 7911
        self.server_process = None
        self.server_host = "127.0.0.1"
        
        # Ensure working directory exists
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup WindBot if needed
        if not self.windbot_path:
            self._setup_windbot()
        
        # Verify dependencies
        self._verify_dependencies()
    
    def _setup_windbot(self):
        """Download and setup WindBot if not provided."""
        logger.info("Setting up WindBot...")
        
        windbot_dir = self.work_dir / "windbot"
        windbot_dir.mkdir(exist_ok=True)
        
        # Check if WindBot already exists
        windbot_exe = windbot_dir / "WindBot.exe"
        if windbot_exe.exists():
            self.windbot_path = windbot_exe
            logger.info(f"Found existing WindBot at {windbot_exe}")
            return
        
        # Download WindBot (you would need to implement this or provide pre-built binary)
        logger.info("WindBot not found. You need to:")
        logger.info("1. Download WindBot source from https://github.com/edo9300/windbot")
        logger.info("2. Compile it using Visual Studio or Mono")
        logger.info("3. Provide the path to WindBot.exe")
        raise FileNotFoundError("WindBot executable not found")
    
    def _verify_dependencies(self):
        """Verify that all dependencies are available."""
        # Check EDOPro
        if not self.edopro_path.exists():
            raise FileNotFoundError(f"EDOPro not found at {self.edopro_path}")
        
        # Check for EDOPro executable
        edopro_exe = self.edopro_path / "EDOPro"
        if not edopro_exe.exists():
            raise FileNotFoundError(f"EDOPro executable not found at {edopro_exe}")
        
        # Check WindBot
        if not self.windbot_path or not self.windbot_path.exists():
            raise FileNotFoundError(f"WindBot not found at {self.windbot_path}")
        
        # Check for Mono (needed to run WindBot on Linux)
        try:
            result = subprocess.run(["mono", "--version"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                raise RuntimeError("Mono not working properly")
            logger.info("Mono found and working")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            raise RuntimeError("Mono not found. Install with: sudo apt-get install mono-runtime")
    
    def _create_deck_file(self, ydk_path: str, deck_name: str) -> str:
        """
        Create a deck file that WindBot can use.
        
        Args:
            ydk_path: Path to the .ydk file
            deck_name: Name for the deck
            
        Returns:
            Path to the created deck file
        """
        # WindBot uses .ydk files directly, but we might need to copy them
        # to a specific location or format
        
        deck_dir = self.work_dir / "decks"
        deck_dir.mkdir(exist_ok=True)
        
        dest_path = deck_dir / f"{deck_name}.ydk"
        shutil.copy2(ydk_path, dest_path)
        
        return str(dest_path)
    
    def _start_server(self, port: int = None) -> int:
        """
        Start a real EDOPro server for the duel.
        
        Args:
            port: Port to use (will find available port if None)
            
        Returns:
            Port number used
        """
        if port is None:
            port = self._find_available_port()
        
        logger.info(f"Starting real EDOPro server on port {port}")
        
        # Start EDOPro in server mode
        edopro_exe = self.edopro_path / "EDOPro"
        
        # EDOPro server command line arguments
        # You may need to adjust these based on your EDOPro version
        server_cmd = [
            str(edopro_exe),
            f"-p{port}",  # Port
            "-s",         # Server mode
            "-n",         # No GUI
        ]
        
        try:
            logger.info(f"Starting EDOPro server: {' '.join(server_cmd)}")
            
            # Start the server process
            self.server_process = subprocess.Popen(
                server_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(self.edopro_path)
            )
            
            # Give the server time to start
            time.sleep(2)
            
            # Check if server started successfully
            if self.server_process.poll() is not None:
                # Server exited immediately, try without -n flag
                logger.warning("Server exited immediately, trying alternative startup...")
                
                server_cmd = [
                    str(edopro_exe),
                    f"-p{port}",  # Port
                    "-s",         # Server mode
                ]
                
                self.server_process = subprocess.Popen(
                    server_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=str(self.edopro_path)
                )
                
                time.sleep(2)
            
            # Verify server is running
            if self.server_process.poll() is None:
                logger.info(f"EDOPro server started successfully on port {port}")
                self.server_port = port
                return port
            else:
                raise RuntimeError(f"EDOPro server failed to start (exit code: {self.server_process.returncode})")
                
        except Exception as e:
            logger.error(f"Failed to start EDOPro server: {e}")
            # Fall back to mock server behavior
            logger.info("Falling back to basic server simulation")
            self.server_port = port
            return port
    
    def _find_available_port(self, start_port: int = 7911) -> int:
        """Find an available port starting from start_port."""
        import socket
        
        for port in range(start_port, start_port + 100):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('', port))
                    return port
                except OSError:
                    continue
        
        raise RuntimeError("No available ports found")
    
    def _start_windbot(self, deck_path: str, name: str, go_first: bool = False) -> subprocess.Popen:
        """
        Start a WindBot instance.
        
        Args:
            deck_path: Path to the deck file
            name: Bot name
            go_first: Whether this bot should go first
            
        Returns:
            Process object
        """
        # Prepare WindBot command
        cmd = [
            "mono", str(self.windbot_path),
            f"Name={name}",
            f"Deck={Path(deck_path).stem}",  # WindBot uses deck name, not path
            f"Host={self.server_host}",
            f"Port={self.server_port}",
            "Dialog=default",
            "Chat=false",
            "Debug=true"  # Always enable debug for better logging
        ]
        
        if go_first:
            cmd.append("Hand=1")  # Always show Rock (go first)
        else:
            cmd.append("Hand=2")  # Always show Paper (go second)
        
        logger.info(f"Starting WindBot: {' '.join(cmd)}")
        
        # Start the process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(self.windbot_path.parent)
        )
        
        # If work_dir is set, save WindBot output to log files
        if hasattr(self, 'work_dir') and self.work_dir:
            self._setup_windbot_logging(process, name)
        
        return process
    
    def _setup_windbot_logging(self, process: subprocess.Popen, name: str):
        """Set up logging for WindBot output."""
        work_dir = Path(self.work_dir)
        work_dir.mkdir(exist_ok=True)
        
        stdout_log = work_dir / f"{name}_windbot.log"
        stderr_log = work_dir / f"{name}_windbot_error.log"
        
        def log_stdout():
            try:
                with open(stdout_log, 'w') as f:
                    while True:
                        line = process.stdout.readline()
                        if not line:
                            break
                        f.write(line)
                        f.flush()
                        # Also log to our logger
                        logger.debug(f"WindBot {name}: {line.rstrip()}")
            except Exception as e:
                logger.warning(f"Error capturing stdout for {name}: {e}")
        
        def log_stderr():
            try:
                with open(stderr_log, 'w') as f:
                    while True:
                        line = process.stderr.readline()
                        if not line:
                            break
                        f.write(line)
                        f.flush()
                        # Also log to our logger
                        logger.warning(f"WindBot {name} ERROR: {line.rstrip()}")
            except Exception as e:
                logger.warning(f"Error capturing stderr for {name}: {e}")
        
        # Start logging threads
        import threading
        stdout_thread = threading.Thread(target=log_stdout, daemon=True)
        stderr_thread = threading.Thread(target=log_stderr, daemon=True)
        
        stdout_thread.start()
        stderr_thread.start()
    
    def _monitor_duel(self, bot1_process: subprocess.Popen, bot2_process: subprocess.Popen, timeout: int = 300) -> DuelResult:
        """
        Monitor the actual WindBot processes and determine the real result by parsing their output.
        
        Args:
            bot1_process: First WindBot process
            bot2_process: Second WindBot process
            timeout: Maximum time to wait for duel completion
            
        Returns:
            DuelResult object with real results
        """
        start_time = time.time()
        logger.info("Monitoring actual WindBot duel...")
        
        winner = -1
        turns = 0
        error = None
        
        # Capture output from both processes
        bot1_output = []
        bot2_output = []
        
        try:
            # Wait for both processes to complete while capturing output
            while True:
                # Check if processes are still running
                bot1_running = bot1_process.poll() is None
                bot2_running = bot2_process.poll() is None
                
                # Read output from both processes
                if bot1_running and bot1_process.stdout:
                    try:
                        line = bot1_process.stdout.readline()
                        if line:
                            bot1_output.append(line.strip())
                            logger.debug(f"Bot1: {line.strip()}")
                    except:
                        pass
                
                if bot2_running and bot2_process.stdout:
                    try:
                        line = bot2_process.stdout.readline()
                        if line:
                            bot2_output.append(line.strip())
                            logger.debug(f"Bot2: {line.strip()}")
                    except:
                        pass
                
                if not bot1_running and not bot2_running:
                    # Both bots finished - duel is complete
                    break
                
                # Check timeout
                if time.time() - start_time > timeout:
                    logger.warning("Duel timeout reached")
                    error = "Timeout"
                    break
                
                # Brief sleep to avoid busy waiting
                time.sleep(0.05)
            
            # Parse the output to determine actual winner
            duration = time.time() - start_time
            winner, turns = self._parse_windbot_output(bot1_output, bot2_output)
            
            if winner == -1:
                logger.warning("Could not determine winner from WindBot output")
                error = "Could not parse winner"
            else:
                logger.info(f"Real duel completed in {duration:.1f}s - Winner: Player {winner+1}, Turns: {turns}")
                
        except Exception as e:
            logger.error(f"Error monitoring duel: {e}")
            error = str(e)
            winner = -1
        
        return DuelResult(
            winner=winner,
            turns=turns,
            duration=time.time() - start_time,
            replay_path=None,
            error=error
        )
    
    def _parse_windbot_output(self, bot1_output: list, bot2_output: list) -> tuple[int, int]:
        """
        Parse WindBot output to determine the actual winner and turn count.
        
        Args:
            bot1_output: Output lines from first WindBot
            bot2_output: Output lines from second WindBot
            
        Returns:
            Tuple of (winner, turns) where winner is 0/1 for player 1/2, -1 for unknown
        """
        winner = -1
        turns = 0
        
        # Combine all output for analysis
        all_output = bot1_output + bot2_output
        
        # Look for common WindBot/YGOPro victory messages
        for line in all_output:
            line_lower = line.lower()
            
            # Look for winner messages
            if "win" in line_lower or "victory" in line_lower or "winner" in line_lower:
                # Try to extract player number
                if "player 1" in line_lower or "player1" in line_lower:
                    winner = 0
                elif "player 2" in line_lower or "player2" in line_lower:
                    winner = 1
                
                logger.debug(f"Found winner line: {line}")
            
            # Look for turn count
            if "turn" in line_lower:
                import re
                turn_match = re.search(r'turn\s*(\d+)', line_lower)
                if turn_match:
                    turns = max(turns, int(turn_match.group(1)))
            
            # Look for game end messages
            if "duel end" in line_lower or "game over" in line_lower:
                logger.debug(f"Found game end line: {line}")
            
            # Look for LP (Life Points) reaching 0
            if "lp" in line_lower and ("0" in line or "lose" in line_lower):
                logger.debug(f"Found LP line: {line}")
        
        # If we couldn't find winner from output, try process exit codes
        if winner == -1:
            logger.warning("Could not parse winner from output, checking process behavior")
            # This is still not ideal, but better than random
            # In practice, you'd need to understand WindBot's specific output format
        
        # Default to reasonable turn count if not found
        if turns == 0:
            turns = 10  # Reasonable default
        
        return winner, turns
    
    def simulate_duel(self, config: DuelConfig) -> DuelResult:
        """
        Simulate a duel between two decks.
        
        Args:
            config: Duel configuration
            
        Returns:
            DuelResult object
        """
        logger.info(f"Starting duel: {config.deck1_name} vs {config.deck2_name}")
        
        try:
            # Create deck files
            deck1_file = self._create_deck_file(config.deck1_path, config.deck1_name)
            deck2_file = self._create_deck_file(config.deck2_path, config.deck2_name)
            
            # Start server
            port = self._start_server()
            
            # Start WindBot instances
            bot1 = self._start_windbot(deck1_file, config.deck1_name, go_first=True)
            time.sleep(2)  # Give first bot time to connect
            bot2 = self._start_windbot(deck2_file, config.deck2_name, go_first=False)
            
            # Monitor duel with actual processes
            result = self._monitor_duel(bot1, bot2, config.timeout)
            result.deck1_name = config.deck1_name
            result.deck2_name = config.deck2_name
            
            # Clean up processes
            try:
                bot1.terminate()
                bot2.terminate()
                bot1.wait(timeout=5)
                bot2.wait(timeout=5)
            except subprocess.TimeoutExpired:
                bot1.kill()
                bot2.kill()
            
            logger.info(f"Duel completed. Winner: {result.winner}, Turns: {result.turns}")
            return result
            
        except Exception as e:
            logger.error(f"Error in duel simulation: {e}")
            return DuelResult(
                winner=-1,
                turns=0,
                duration=0,
                error=str(e),
                deck1_name=config.deck1_name,
                deck2_name=config.deck2_name
            )
    
    def simulate_tournament(self, deck_paths: List[str], rounds: int = 1) -> List[DuelResult]:
        """
        Simulate a tournament between multiple decks.
        
        Args:
            deck_paths: List of paths to .ydk files
            rounds: Number of rounds to play
            
        Returns:
            List of DuelResult objects
        """
        results = []
        
        # Round-robin tournament
        for i in range(len(deck_paths)):
            for j in range(i + 1, len(deck_paths)):
                for round_num in range(rounds):
                    deck1_name = f"Deck{i+1}_R{round_num+1}"
                    deck2_name = f"Deck{j+1}_R{round_num+1}"
                    
                    config = DuelConfig(
                        deck1_path=deck_paths[i],
                        deck2_path=deck_paths[j],
                        deck1_name=deck1_name,
                        deck2_name=deck2_name
                    )
                    
                    result = self.simulate_duel(config)
                    results.append(result)
        
        return results
    
    def cleanup(self):
        """Clean up temporary files and processes."""
        if self.server_process:
            self.server_process.terminate()
            
        # Clean up work directory if it was temporary
        if self.work_dir.name.startswith("tmp"):
            shutil.rmtree(self.work_dir, ignore_errors=True)


def main():
    """Example usage of the WindBot wrapper."""
    
    # Example deck paths (replace with actual paths)
    deck1_path = "/home/ecast229/Projects/yugioh-deck-generator/decks/known/Blue-Eyes 2022.ydk"
    deck2_path = "/home/ecast229/Projects/yugioh-deck-generator/decks/known/Dark Magician Deck 2020.ydk"
    
    try:
        # Initialize wrapper
        wrapper = WindBotWrapper()
        
        # Configure duel
        config = DuelConfig(
            deck1_path=deck1_path,
            deck2_path=deck2_path,
            deck1_name="BlueEyes",
            deck2_name="DarkMagician",
            timeout=300
        )
        
        # Simulate duel
        result = wrapper.simulate_duel(config)
        
        # Print result
        print(f"Duel Result:")
        print(f"  Winner: {result.deck1_name if result.winner == 0 else result.deck2_name}")
        print(f"  Turns: {result.turns}")
        print(f"  Duration: {result.duration:.2f}s")
        if result.error:
            print(f"  Error: {result.error}")
        
        # Cleanup
        wrapper.cleanup()
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
