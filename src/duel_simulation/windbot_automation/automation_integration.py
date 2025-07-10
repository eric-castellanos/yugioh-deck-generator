#!/usr/bin/env python3
"""
Complete YGOPro Automation Integration

This module integrates the YGOPro server and WindBot automation
to provide complete .ydk vs .ydk duel simulation capabilities.
"""

import os
import sys
import time
import json
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

# Add the current directory to Python path to import our modules
sys.path.insert(0, str(Path(__file__).parent))

from windbot_wrapper import WindBotWrapper, DuelConfig, DuelResult

logger = logging.getLogger(__name__)


@dataclass
class AutomationConfig:
    """Configuration for complete automation."""
    edopro_path: str = "/home/ecast229/Applications/EDOPro"
    windbot_path: Optional[str] = None
    work_dir: Optional[str] = None
    server_port: int = 7911
    duel_timeout: int = 300
    debug: bool = False
    save_replays: bool = True
    

class YGOProAutomation:
    """
    Complete YGOPro automation system.
    
    This class combines server management and WindBot automation
    to provide seamless .ydk vs .ydk duel simulation.
    """
    
    def __init__(self, config: AutomationConfig):
        """
        Initialize the automation system.
        
        Args:
            config: Automation configuration
        """
        self.config = config
        self.windbot_wrapper = None
        
        # Setup logging
        log_level = logging.DEBUG if config.debug else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize WindBot wrapper."""
        logger.info("Initializing automation components...")
        
        # Initialize WindBot wrapper
        try:
            self.windbot_wrapper = WindBotWrapper(
                edopro_path=self.config.edopro_path,
                windbot_path=self.config.windbot_path,
                work_dir=self.config.work_dir
            )
        except Exception as e:
            logger.warning(f"WindBot wrapper initialization failed: {e}")
            logger.info("You may need to run setup_windbot.sh first")
            raise
        
        logger.info("Automation components initialized successfully")
    
    def simulate_single_duel(self, 
                           deck1_path: str, 
                           deck2_path: str,
                           deck1_name: str = "Player1",
                           deck2_name: str = "Player2") -> DuelResult:
        """
        Simulate a single duel between two decks.
        
        Args:
            deck1_path: Path to first deck (.ydk file)
            deck2_path: Path to second deck (.ydk file)
            deck1_name: Name for first player
            deck2_name: Name for second player
            
        Returns:
            DuelResult object containing the outcome
        """
        logger.info(f"Starting duel: {deck1_name} vs {deck2_name}")
        
        try:
            # Configure duel
            duel_config = DuelConfig(
                deck1_path=deck1_path,
                deck2_path=deck2_path,
                deck1_name=deck1_name,
                deck2_name=deck2_name,
                timeout=self.config.duel_timeout,
                debug=self.config.debug
            )
            
            # Simulate duel using WindBot directly (no separate server needed)
            result = self.windbot_wrapper.simulate_duel(duel_config)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in duel simulation: {e}")
            
            return DuelResult(
                winner=-1,
                turns=0,
                duration=0,
                error=str(e),
                deck1_name=deck1_name,
                deck2_name=deck2_name
            )
    
    def simulate_tournament(self, 
                          deck_paths: List[str],
                          rounds: int = 1,
                          round_robin: bool = True) -> List[DuelResult]:
        """
        Simulate a tournament between multiple decks.
        
        Args:
            deck_paths: List of paths to .ydk files
            rounds: Number of rounds per matchup
            round_robin: If True, every deck plays every other deck
            
        Returns:
            List of DuelResult objects
        """
        logger.info(f"Starting tournament with {len(deck_paths)} decks, {rounds} rounds")
        
        results = []
        
        if round_robin:
            # Round-robin format
            for i in range(len(deck_paths)):
                for j in range(i + 1, len(deck_paths)):
                    deck1_path = deck_paths[i]
                    deck2_path = deck_paths[j]
                    
                    deck1_name = Path(deck1_path).stem
                    deck2_name = Path(deck2_path).stem
                    
                    for round_num in range(rounds):
                        logger.info(f"Round {round_num + 1}: {deck1_name} vs {deck2_name}")
                        
                        result = self.simulate_single_duel(
                            deck1_path=deck1_path,
                            deck2_path=deck2_path,
                            deck1_name=f"{deck1_name}_R{round_num + 1}",
                            deck2_name=f"{deck2_name}_R{round_num + 1}"
                        )
                        
                        results.append(result)
                        
                        # Brief pause between duels
                        time.sleep(2)
        
        else:
            # Single elimination or other format
            # (Implementation would depend on specific tournament format)
            pass
        
        logger.info(f"Tournament completed. {len(results)} duels simulated.")
        return results
    
    def analyze_results(self, results: List[DuelResult]) -> Dict[str, Any]:
        """
        Analyze tournament results and generate statistics.
        
        Args:
            results: List of DuelResult objects
            
        Returns:
            Dictionary containing analysis results
        """
        if not results:
            return {'error': 'No results to analyze'}
        
        # Extract deck names
        all_decks = set()
        for result in results:
            all_decks.add(result.deck1_name.split('_R')[0])  # Remove round suffix
            all_decks.add(result.deck2_name.split('_R')[0])
        
        # Calculate statistics per deck
        deck_stats = {}
        for deck in all_decks:
            wins = 0
            losses = 0
            total_turns = 0
            total_duration = 0
            games = 0
            
            for result in results:
                deck1_base = result.deck1_name.split('_R')[0]
                deck2_base = result.deck2_name.split('_R')[0]
                
                if deck1_base == deck:
                    games += 1
                    total_turns += result.turns
                    total_duration += result.duration
                    if result.winner == 0:
                        wins += 1
                    else:
                        losses += 1
                        
                elif deck2_base == deck:
                    games += 1
                    total_turns += result.turns
                    total_duration += result.duration
                    if result.winner == 1:
                        wins += 1
                    else:
                        losses += 1
            
            if games > 0:
                deck_stats[deck] = {
                    'wins': wins,
                    'losses': losses,
                    'win_rate': wins / games,
                    'avg_turns': total_turns / games,
                    'avg_duration': total_duration / games,
                    'total_games': games
                }
        
        # Overall statistics
        total_duels = len(results)
        completed_duels = len([r for r in results if r.error is None])
        error_rate = (total_duels - completed_duels) / total_duels if total_duels > 0 else 0
        
        avg_turns = sum(r.turns for r in results if r.error is None) / completed_duels if completed_duels > 0 else 0
        avg_duration = sum(r.duration for r in results if r.error is None) / completed_duels if completed_duels > 0 else 0
        
        analysis = {
            'total_duels': total_duels,
            'completed_duels': completed_duels,
            'error_rate': error_rate,
            'avg_turns_per_duel': avg_turns,
            'avg_duration_per_duel': avg_duration,
            'deck_statistics': deck_stats,
            'top_decks': sorted(deck_stats.items(), key=lambda x: x[1]['win_rate'], reverse=True)[:5]
        }
        
        return analysis
    
    def save_results(self, results: List[DuelResult], output_path: str):
        """
        Save results to a JSON file.
        
        Args:
            results: List of DuelResult objects
            output_path: Path to save the results
        """
        # Convert results to serializable format
        serializable_results = []
        for result in results:
            result_dict = asdict(result)
            serializable_results.append(result_dict)
        
        # Add metadata
        output_data = {
            'metadata': {
                'total_duels': len(results),
                'generated_at': time.time(),
                'config': asdict(self.config)
            },
            'results': serializable_results,
            'analysis': self.analyze_results(results)
        }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    def cleanup(self):
        """Clean up automation system resources."""
        logger.info("Cleaning up automation system...")
        
        if self.windbot_wrapper:
            self.windbot_wrapper.cleanup()


def main():
    """Example usage of the complete automation system."""
    
    # Example configuration
    config = AutomationConfig(
        edopro_path="/home/ecast229/Applications/EDOPro",
        windbot_path=None,  # Will auto-detect
        debug=True,
        save_replays=True
    )
    
    # Example deck paths
    deck_dir = Path("/home/ecast229/Projects/yugioh-deck-generator/decks/known")
    deck_files = [
        deck_dir / "Blue-Eyes 2022.ydk",
        deck_dir / "Dark Magician Deck 2020.ydk",
        deck_dir / "Blackwing Deck 2021.ydk"
    ]
    
    # Filter to existing files
    deck_files = [f for f in deck_files if f.exists()]
    
    if len(deck_files) < 2:
        logger.error("Need at least 2 deck files for testing")
        return 1
    
    try:
        # Initialize automation
        automation = YGOProAutomation(config)
        
        # Run a simple test duel
        logger.info("Running test duel...")
        result = automation.simulate_single_duel(
            deck1_path=str(deck_files[0]),
            deck2_path=str(deck_files[1]),
            deck1_name="TestDeck1",
            deck2_name="TestDeck2"
        )
        
        print("\nDuel Result:")
        print(f"  Winner: {result.deck1_name if result.winner == 0 else result.deck2_name}")
        print(f"  Turns: {result.turns}")
        print(f"  Duration: {result.duration:.2f}s")
        if result.error:
            print(f"  Error: {result.error}")
        if result.replay_path:
            print(f"  Replay: {result.replay_path}")
        
        # Run a small tournament if we have enough decks
        if len(deck_files) >= 3:
            logger.info("Running mini tournament...")
            tournament_results = automation.simulate_tournament(
                deck_paths=[str(f) for f in deck_files[:3]],
                rounds=1
            )
            
            # Analyze and save results
            analysis = automation.analyze_results(tournament_results)
            print(f"\nTournament Analysis:")
            print(f"  Total duels: {analysis['total_duels']}")
            print(f"  Completion rate: {(1 - analysis['error_rate']) * 100:.1f}%")
            print(f"  Average turns: {analysis['avg_turns_per_duel']:.1f}")
            print(f"  Average duration: {analysis['avg_duration_per_duel']:.1f}s")
            
            # Save results
            output_path = "tournament_results.json"
            automation.save_results(tournament_results, output_path)
            print(f"  Results saved to: {output_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in automation: {e}")
        return 1
    
    finally:
        if 'automation' in locals():
            automation.cleanup()


if __name__ == "__main__":
    sys.exit(main())
