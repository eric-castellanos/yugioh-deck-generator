#!/usr/bin/env python3
"""
ML Pipeline Integration for WindBot Automation

This module provides high-level functions for integrating the WindBot automation
system into machine learning pipelines for deck evaluation and analysis.
"""

import json
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import asdict

from .automation_integration import YGOProAutomation, AutomationConfig

logger = logging.getLogger(__name__)

class MLDuelEvaluator:
    """High-level interface for ML-driven deck evaluation using automated duels."""
    
    def __init__(self, 
                 edopro_path: str = "/home/ecast229/Applications/EDOPro",
                 windbot_path: str = "/home/ecast229/Projects/yugioh-deck-generator/Applications/WindBot/WindBot.exe",
                 work_dir: Optional[str] = None,
                 debug: bool = False,
                 base_port: int = 7911):
        """
        Initialize the ML duel evaluator.
        
        Args:
            edopro_path: Path to EDOPro installation directory
            windbot_path: Path to WindBot executable
            work_dir: Working directory for temporary files
            debug: Enable debug logging
            base_port: Base port for server (for parallel execution support)
        """
        self.config = AutomationConfig(
            edopro_path=edopro_path,
            windbot_path=windbot_path,
            work_dir=work_dir,
            debug=debug,
            save_replays=True,
            duel_timeout=120,  # 2 minutes per duel
            server_port=base_port  # Use base_port as server_port
        )
        
        self.automation = YGOProAutomation(self.config)
        logger.info("ML Duel Evaluator initialized")
    
    def evaluate_deck_against_meta(self, 
                                   deck_path: str, 
                                   meta_deck_paths: List[str],
                                   rounds_per_matchup: int = 3) -> Dict[str, Any]:
        """
        Evaluate a single deck against a meta of known strong decks.
        
        Args:
            deck_path: Path to the deck to evaluate (.ydk file)
            meta_deck_paths: List of paths to meta decks
            rounds_per_matchup: Number of duels per matchup
            
        Returns:
            Dict with evaluation metrics and detailed results
        """
        deck_name = Path(deck_path).stem
        total_expected_duels = len(meta_deck_paths) * rounds_per_matchup
        logger.info(f"Evaluating deck '{deck_name}' against {len(meta_deck_paths)} meta decks ({total_expected_duels} duels)")
        
        results = []
        total_wins = 0
        total_duels = 0
        completed_duels = 0
        
        for meta_idx, meta_deck_path in enumerate(meta_deck_paths):
            meta_deck_name = Path(meta_deck_path).stem
            meta_progress = ((meta_idx + 1) / len(meta_deck_paths)) * 100
            logger.info(f"  🎯 vs {meta_deck_name} ({meta_idx+1}/{len(meta_deck_paths)}, {meta_progress:.1f}%)")
            
            # Run multiple duels against this meta deck
            for round_num in range(rounds_per_matchup):
                try:
                    result = self.automation.simulate_single_duel(
                        deck1_path=deck_path,
                        deck2_path=meta_deck_path,
                        deck1_name=f"{deck_name}_R{round_num+1}",
                        deck2_name=f"{meta_deck_name}_R{round_num+1}"
                    )
                    
                    completed_duels += 1
                    duel_progress = (completed_duels / total_expected_duels) * 100
                    
                    if result.error is None:
                        total_duels += 1
                        if result.winner == 0:  # Our deck won
                            total_wins += 1
                        
                        win_status = "WON" if result.winner == 0 else "LOST"
                        logger.info(f"    Round {round_num+1}: {win_status} in {result.turns} turns ({duel_progress:.1f}%)")
                        
                        results.append({
                            'deck_name': deck_name,
                            'opponent': meta_deck_name,
                            'round': round_num + 1,
                            'won': result.winner == 0,
                            'turns': result.turns,
                            'duration': result.duration,
                            'replay_path': result.replay_path
                        })
                    else:
                        logger.warning(f"    Round {round_num+1}: FAILED - {result.error} ({duel_progress:.1f}%)")
                        
                except Exception as e:
                    completed_duels += 1
                    duel_progress = (completed_duels / total_expected_duels) * 100
                    logger.error(f"    Round {round_num+1}: ERROR - {e} ({duel_progress:.1f}%)")
        
        # Calculate metrics
        win_rate = total_wins / total_duels if total_duels > 0 else 0.0
        avg_turns = sum(r['turns'] for r in results) / len(results) if results else 0
        avg_duration = sum(r['duration'] for r in results) / len(results) if results else 0
        
        evaluation = {
            'deck_name': deck_name,
            'deck_path': deck_path,
            'meta_decks_tested': len(meta_deck_paths),
            'total_duels': total_duels,
            'wins': total_wins,
            'losses': total_duels - total_wins,
            'win_rate': win_rate,
            'avg_turns_per_duel': avg_turns,
            'avg_duration_per_duel': avg_duration,
            'detailed_results': results
        }
        
        logger.info(f"Evaluation complete: {deck_name} win rate = {win_rate:.2%}")
        return evaluation
    
    def batch_evaluate_decks(self, 
                            deck_paths: List[str], 
                            meta_deck_paths: List[str],
                            rounds_per_matchup: int = 3,
                            save_results: bool = True,
                            output_dir: str = "outputs/deck_analysis") -> pd.DataFrame:
        """
        Evaluate multiple decks against a meta and return results as DataFrame.
        
        Args:
            deck_paths: List of deck paths to evaluate
            meta_deck_paths: List of meta deck paths
            rounds_per_matchup: Number of duels per matchup
            save_results: Whether to save results to disk
            output_dir: Directory to save results
            
        Returns:
            DataFrame with evaluation results for all decks
        """
        logger.info(f"Starting batch evaluation of {len(deck_paths)} decks")
        
        all_evaluations = []
        
        for i, deck_path in enumerate(deck_paths):
            logger.info(f"Evaluating deck {i+1}/{len(deck_paths)}: {Path(deck_path).stem}")
            
            try:
                evaluation = self.evaluate_deck_against_meta(
                    deck_path=deck_path,
                    meta_deck_paths=meta_deck_paths,
                    rounds_per_matchup=rounds_per_matchup
                )
                all_evaluations.append(evaluation)
                
            except Exception as e:
                logger.error(f"Failed to evaluate {deck_path}: {e}")
                # Add failed evaluation record
                all_evaluations.append({
                    'deck_name': Path(deck_path).stem,
                    'deck_path': deck_path,
                    'meta_decks_tested': len(meta_deck_paths),
                    'total_duels': 0,
                    'wins': 0,
                    'losses': 0,
                    'win_rate': 0.0,
                    'avg_turns_per_duel': 0.0,
                    'avg_duration_per_duel': 0.0,
                    'error': str(e)
                })
        
        # Create DataFrame
        df = pd.DataFrame([
            {
                'deck_name': eval_result['deck_name'],
                'deck_path': eval_result['deck_path'],
                'win_rate': eval_result['win_rate'],
                'total_duels': eval_result['total_duels'],
                'wins': eval_result['wins'],
                'losses': eval_result['losses'],
                'avg_turns': eval_result['avg_turns_per_duel'],
                'avg_duration': eval_result['avg_duration_per_duel'],
                'error': eval_result.get('error', None)
            }
            for eval_result in all_evaluations
        ])
        
        if save_results:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save DataFrame as CSV
            csv_path = output_path / "batch_evaluation_results.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"Results saved to: {csv_path}")
            
            # Save detailed results as JSON
            json_path = output_path / "detailed_evaluation_results.json"
            with open(json_path, 'w') as f:
                json.dump(all_evaluations, f, indent=2)
            logger.info(f"Detailed results saved to: {json_path}")
        
        return df
    
    def find_optimal_meta_decks(self, 
                               all_deck_paths: List[str],
                               num_meta_decks: int = 10,
                               evaluation_rounds: int = 2) -> List[str]:
        """
        Find the strongest decks to use as a meta for evaluation.
        
        This runs a tournament to identify the top-performing decks.
        
        Args:
            all_deck_paths: All available deck paths
            num_meta_decks: Number of top decks to return
            evaluation_rounds: Rounds per matchup in meta tournament
            
        Returns:
            List of paths to the strongest decks
        """
        logger.info(f"Finding optimal meta from {len(all_deck_paths)} decks")
        
        # Run tournament to find strongest decks
        tournament_results = self.automation.simulate_tournament(
            deck_paths=all_deck_paths,
            rounds=evaluation_rounds
        )
        
        # Count wins per deck
        deck_wins = {}
        deck_games = {}
        
        for result in tournament_results:
            if result.error is None:
                deck1_name = result.deck1_name.split('_R')[0]  # Remove round suffix
                deck2_name = result.deck2_name.split('_R')[0]
                
                # Initialize if not seen
                for deck_name in [deck1_name, deck2_name]:
                    if deck_name not in deck_wins:
                        deck_wins[deck_name] = 0
                        deck_games[deck_name] = 0
                
                # Count games
                deck_games[deck1_name] += 1
                deck_games[deck2_name] += 1
                
                # Count wins
                if result.winner == 0:
                    deck_wins[deck1_name] += 1
                else:
                    deck_wins[deck2_name] += 1
        
        # Calculate win rates and sort
        deck_win_rates = {}
        for deck_name in deck_wins:
            if deck_games[deck_name] > 0:
                deck_win_rates[deck_name] = deck_wins[deck_name] / deck_games[deck_name]
            else:
                deck_win_rates[deck_name] = 0.0
        
        # Sort by win rate and take top decks
        top_decks = sorted(deck_win_rates.items(), key=lambda x: x[1], reverse=True)
        top_deck_names = [deck[0] for deck in top_decks[:num_meta_decks]]
        
        # Map back to file paths
        name_to_path = {Path(path).stem: path for path in all_deck_paths}
        meta_deck_paths = [name_to_path[name] for name in top_deck_names if name in name_to_path]
        
        logger.info(f"Identified {len(meta_deck_paths)} meta decks with win rates:")
        for i, deck_name in enumerate(top_deck_names[:len(meta_deck_paths)]):
            win_rate = deck_win_rates[deck_name]
            logger.info(f"  {i+1}. {deck_name}: {win_rate:.2%}")
        
        return meta_deck_paths
    
    def cleanup(self):
        """Clean up automation resources."""
        self.automation.cleanup()
        logger.info("ML Duel Evaluator cleaned up")


# Example usage functions for ML pipeline integration
def quick_deck_evaluation(deck_path: str, 
                         meta_deck_dir: str = "/home/ecast229/Projects/yugioh-deck-generator/decks/known",
                         num_meta_decks: int = 10,
                         base_port: int = 7911) -> Dict[str, Any]:
    """
    Quick evaluation of a single deck against top meta decks.
    
    Args:
        deck_path: Path to deck to evaluate
        meta_deck_dir: Directory containing meta decks
        num_meta_decks: Number of meta decks to test against
        base_port: Base port for server (for parallel execution support)
        
    Returns:
        Evaluation results dictionary
    """
    evaluator = MLDuelEvaluator(base_port=base_port)
    
    try:
        # Get meta decks
        meta_deck_paths = list(Path(meta_deck_dir).glob("*.ydk"))[:num_meta_decks]
        
        # Evaluate
        result = evaluator.evaluate_deck_against_meta(
            deck_path=deck_path,
            meta_deck_paths=[str(p) for p in meta_deck_paths],
            rounds_per_matchup=2
        )
        
        return result
        
    finally:
        evaluator.cleanup()


def ml_training_data_generation(deck_dir: str,
                               output_dir: str = "outputs/ml_training_data",
                               sample_size: int = 50,
                               base_port: int = 7911) -> str:
    """
    Generate training data for ML models by running automated duels.
    
    Args:
        deck_dir: Directory containing deck files
        output_dir: Output directory for training data
        sample_size: Number of decks to sample for training data
        base_port: Base port for server (for parallel execution support)
        
    Returns:
        Path to generated training data CSV
    """
    evaluator = MLDuelEvaluator(debug=False, base_port=base_port)  # Disable debug for speed
    
    try:
        # Get deck files
        all_deck_paths = list(Path(deck_dir).glob("*.ydk"))
        
        # Sample decks if we have too many
        if len(all_deck_paths) > sample_size:
            import random
            all_deck_paths = random.sample(all_deck_paths, sample_size)
        
        # Find meta decks (top 20%)
        num_meta = max(3, len(all_deck_paths) // 5)
        meta_deck_paths = evaluator.find_optimal_meta_decks(
            all_deck_paths=[str(p) for p in all_deck_paths],
            num_meta_decks=num_meta,
            evaluation_rounds=1
        )
        
        # Evaluate all decks against meta
        df = evaluator.batch_evaluate_decks(
            deck_paths=[str(p) for p in all_deck_paths],
            meta_deck_paths=meta_deck_paths,
            rounds_per_matchup=2,
            save_results=True,
            output_dir=output_dir
        )
        
        training_data_path = Path(output_dir) / "batch_evaluation_results.csv"
        logger.info(f"ML training data generated: {training_data_path}")
        
        return str(training_data_path)
        
    finally:
        evaluator.cleanup()


if __name__ == "__main__":
    # Example: Quick evaluation of a single deck
    deck_path = "/home/ecast229/Projects/yugioh-deck-generator/decks/known/Blue-Eyes 2022.ydk"
    
    print("🤖 ML Integration Demo")
    print("=" * 40)
    
    result = quick_deck_evaluation(deck_path, num_meta_decks=5)
    
    print(f"Deck: {result['deck_name']}")
    print(f"Win Rate: {result['win_rate']:.2%}")
    print(f"Total Duels: {result['total_duels']}")
    print(f"Average Turns: {result['avg_turns_per_duel']:.1f}")
    print(f"Average Duration: {result['avg_duration_per_duel']:.1f}s")
