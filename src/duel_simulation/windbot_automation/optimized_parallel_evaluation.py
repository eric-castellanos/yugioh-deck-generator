#!/usr/bin/env python3
"""
Optimized Parallel Yu-Gi-Oh! Deck Evaluation System

This module provides highly optimized parallel simulation with:
- Persistent YGOPro servers per worker (no restart overhead)
- Configurable worker count (use all CPU cores)
- Port pool management to avoid collisions
- Batch processing for maximum efficiency
"""

import csv
import json
import logging
import random
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import threading
import queue
import signal
import os

from .automation_integration import YGOProAutomation, AutomationConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class OptimizedEvaluationConfig:
    """Configuration for optimized parallel deck evaluation."""
    known_decks_dir: str = "/home/ecast229/Projects/yugioh-deck-generator/decks/known"
    novel_decks_dir: str = "/home/ecast229/Projects/yugioh-deck-generator/decks/novel"
    output_file: str = "outputs/deck_evaluation_results.csv"
    k_known_decks: int = 10  # Number of known decks to sample per novel deck
    n_duels_per_pair: int = 20  # Number of duels per novel-known pair
    parallel_workers: int = 0  # Number of parallel workers (0 = auto-detect all CPUs)
    base_port: int = 8000  # Base port for YGOPro servers
    port_range: int = 100  # Port range per worker (worker 0: 8000-8099, worker 1: 8100-8199, etc.)
    duel_timeout: int = 180  # Timeout per duel in seconds
    batch_size: int = 50  # Number of duels per batch (for progress tracking)
    save_replays: bool = False  # Set to True to save all replay files
    debug: bool = False  # Enable debug logging
    
    # WindBot/EDOPro paths
    edopro_path: str = "/home/ecast229/Applications/EDOPro"
    windbot_path: str = "/home/ecast229/Projects/yugioh-deck-generator/Applications/WindBot/WindBot.exe"

@dataclass
class DuelTask:
    """Represents a single duel to be executed."""
    novel_deck_path: str
    known_deck_path: str
    novel_goes_first: bool
    task_id: str
    batch_id: int

@dataclass
class DuelOutcome:
    """Result of a single duel."""
    novel_deck: str
    known_deck: str
    result: str  # "win" or "loss" for the novel deck
    who_went_first: str  # "novel" or "known"
    turns: int
    duration: float
    task_id: str
    worker_id: int
    error: Optional[str] = None

@dataclass
class WorkerBatch:
    """Batch of tasks for a worker."""
    worker_id: int
    tasks: List[DuelTask]
    base_port: int

class PersistentWorker:
    """
    A worker that maintains a persistent YGOPro server across multiple duels.
    This eliminates server startup/shutdown overhead.
    """
    
    def __init__(self, worker_id: int, config: OptimizedEvaluationConfig):
        self.worker_id = worker_id
        self.config = config
        self.server_port = config.base_port + (worker_id * config.port_range)
        self.automation = None
        self.is_initialized = False
        
    def initialize(self):
        """Initialize the persistent automation system."""
        try:
            automation_config = AutomationConfig(
                edopro_path=self.config.edopro_path,
                windbot_path=self.config.windbot_path,
                server_port=self.server_port,
                duel_timeout=self.config.duel_timeout,
                save_replays=self.config.save_replays,
                debug=self.config.debug
            )
            
            self.automation = YGOProAutomation(automation_config)
            self.is_initialized = True
            logger.info(f"Worker {self.worker_id} initialized with port {self.server_port}")
            
        except Exception as e:
            logger.error(f"Failed to initialize worker {self.worker_id}: {e}")
            self.is_initialized = False
            raise
    
    def process_batch(self, tasks: List[DuelTask]) -> List[DuelOutcome]:
        """Process a batch of duel tasks using the persistent server."""
        if not self.is_initialized:
            raise RuntimeError(f"Worker {self.worker_id} not initialized")
        
        results = []
        
        for task in tasks:
            try:
                result = self._process_single_duel(task)
                results.append(result)
                
                # Small delay between duels to ensure clean state
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Worker {self.worker_id} failed task {task.task_id}: {e}")
                results.append(DuelOutcome(
                    novel_deck=Path(task.novel_deck_path).stem,
                    known_deck=Path(task.known_deck_path).stem,
                    result="error",
                    who_went_first="novel" if task.novel_goes_first else "known",
                    turns=0,
                    duration=0.0,
                    task_id=task.task_id,
                    worker_id=self.worker_id,
                    error=str(e)
                ))
        
        return results
    
    def _process_single_duel(self, task: DuelTask) -> DuelOutcome:
        """Process a single duel task."""
        # Determine deck names and roles
        novel_deck_name = Path(task.novel_deck_path).stem
        known_deck_name = Path(task.known_deck_path).stem
        
        # Set up the duel based on who goes first
        if task.novel_goes_first:
            deck1_path = task.novel_deck_path
            deck2_path = task.known_deck_path
            deck1_name = f"{novel_deck_name}_P1"
            deck2_name = f"{known_deck_name}_P2"
            who_went_first = "novel"
        else:
            deck1_path = task.known_deck_path
            deck2_path = task.novel_deck_path
            deck1_name = f"{known_deck_name}_P1"
            deck2_name = f"{novel_deck_name}_P2"
            who_went_first = "known"
        
        # Simulate the duel using persistent automation
        result = self.automation.simulate_single_duel(
            deck1_path=deck1_path,
            deck2_path=deck2_path,
            deck1_name=deck1_name,
            deck2_name=deck2_name
        )
        
        if result.error is not None:
            return DuelOutcome(
                novel_deck=novel_deck_name,
                known_deck=known_deck_name,
                result="error",
                who_went_first=who_went_first,
                turns=0,
                duration=0.0,
                task_id=task.task_id,
                worker_id=self.worker_id,
                error=result.error
            )
        
        # Determine if novel deck won
        if task.novel_goes_first:
            novel_won = (result.winner == 0)  # Novel deck was player 1
        else:
            novel_won = (result.winner == 1)  # Novel deck was player 2
        
        return DuelOutcome(
            novel_deck=novel_deck_name,
            known_deck=known_deck_name,
            result="win" if novel_won else "loss",
            who_went_first=who_went_first,
            turns=result.turns,
            duration=result.duration,
            task_id=task.task_id,
            worker_id=self.worker_id
        )
    
    def cleanup(self):
        """Clean up the persistent automation system."""
        if self.automation:
            try:
                self.automation.cleanup()
                logger.info(f"Worker {self.worker_id} cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up worker {self.worker_id}: {e}")
        self.is_initialized = False

def worker_process_function(worker_batch: WorkerBatch, config: OptimizedEvaluationConfig) -> List[DuelOutcome]:
    """
    Function that runs in a separate process to handle a batch of duels.
    This function initializes a persistent worker and processes all tasks in the batch.
    """
    worker = None
    try:
        # Set up signal handling for graceful shutdown
        def signal_handler(signum, frame):
            logger.info(f"Worker {worker_batch.worker_id} received shutdown signal")
            if worker:
                worker.cleanup()
            os._exit(0)
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        # Initialize persistent worker
        worker = PersistentWorker(worker_batch.worker_id, config)
        worker.initialize()
        
        # Process all tasks in the batch
        results = worker.process_batch(worker_batch.tasks)
        
        return results
        
    except Exception as e:
        logger.error(f"Worker process {worker_batch.worker_id} failed: {e}")
        return []
        
    finally:
        if worker:
            worker.cleanup()

class OptimizedParallelDeckEvaluator:
    """
    Optimized parallel deck evaluator with persistent servers and configurable workers.
    """
    
    def __init__(self, config: OptimizedEvaluationConfig):
        self.config = config
        self.results: List[DuelOutcome] = []
        
        # Auto-detect CPU count if not specified
        if config.parallel_workers <= 0:
            self.config.parallel_workers = mp.cpu_count()
            logger.info(f"Auto-detected {self.config.parallel_workers} CPU cores")
        
        # Ensure output directory exists
        output_path = Path(config.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Optimized Parallel Deck Evaluator initialized")
        logger.info(f"Known decks dir: {config.known_decks_dir}")
        logger.info(f"Novel decks dir: {config.novel_decks_dir}")
        logger.info(f"Output file: {config.output_file}")
        logger.info(f"Parallel workers: {config.parallel_workers}")
        logger.info(f"Base port: {config.base_port} (range: {config.port_range} per worker)")
        logger.info(f"Batch size: {config.batch_size} duels per batch")
    
    def load_deck_files(self) -> Tuple[List[Path], List[Path]]:
        """Load all deck files from known and novel directories."""
        known_dir = Path(self.config.known_decks_dir)
        novel_dir = Path(self.config.novel_decks_dir)
        
        known_decks = list(known_dir.glob("*.ydk"))
        novel_decks = list(novel_dir.glob("*.ydk"))
        
        logger.info(f"Found {len(known_decks)} known decks")
        logger.info(f"Found {len(novel_decks)} novel decks")
        
        if len(known_decks) == 0:
            raise ValueError(f"No known decks found in {known_dir}")
        if len(novel_decks) == 0:
            raise ValueError(f"No novel decks found in {novel_dir}")
        
        return known_decks, novel_decks
    
    def generate_duel_tasks(self, known_decks: List[Path], novel_decks: List[Path]) -> List[DuelTask]:
        """Generate all duel tasks to be executed."""
        tasks = []
        task_counter = 0
        
        for novel_deck in novel_decks:
            # Sample k known decks for this novel deck
            k = min(self.config.k_known_decks, len(known_decks))
            sampled_known = random.sample(known_decks, k)
            
            logger.info(f"Novel deck '{novel_deck.stem}' will face {k} known decks")
            
            for known_deck in sampled_known:
                # Generate n duels per pair, alternating who goes first
                for duel_num in range(self.config.n_duels_per_pair):
                    novel_goes_first = (duel_num % 2 == 0)  # Alternate who goes first
                    
                    task = DuelTask(
                        novel_deck_path=str(novel_deck),
                        known_deck_path=str(known_deck),
                        novel_goes_first=novel_goes_first,
                        task_id=f"task_{task_counter:06d}",
                        batch_id=task_counter // self.config.batch_size
                    )
                    tasks.append(task)
                    task_counter += 1
        
        logger.info(f"Generated {len(tasks)} duel tasks")
        return tasks
    
    def create_worker_batches(self, tasks: List[DuelTask]) -> List[WorkerBatch]:
        """Distribute tasks across workers in balanced batches."""
        num_workers = self.config.parallel_workers
        
        # Distribute tasks evenly across workers
        worker_tasks = [[] for _ in range(num_workers)]
        
        for i, task in enumerate(tasks):
            worker_id = i % num_workers
            worker_tasks[worker_id].append(task)
        
        # Create worker batches
        worker_batches = []
        for worker_id in range(num_workers):
            if worker_tasks[worker_id]:  # Only create batch if there are tasks
                batch = WorkerBatch(
                    worker_id=worker_id,
                    tasks=worker_tasks[worker_id],
                    base_port=self.config.base_port + (worker_id * self.config.port_range)
                )
                worker_batches.append(batch)
        
        logger.info(f"Created {len(worker_batches)} worker batches")
        for batch in worker_batches:
            logger.info(f"Worker {batch.worker_id}: {len(batch.tasks)} tasks, port {batch.base_port}")
        
        return worker_batches
    
    def run_optimized_evaluation(self) -> List[DuelOutcome]:
        """Run the optimized parallel evaluation with persistent servers."""
        # Load deck files
        known_decks, novel_decks = self.load_deck_files()
        
        # Generate all tasks
        tasks = self.generate_duel_tasks(known_decks, novel_decks)
        
        if not tasks:
            logger.warning("No tasks generated!")
            return []
        
        # Create worker batches
        worker_batches = self.create_worker_batches(tasks)
        
        logger.info(f"Starting optimized parallel execution with {len(worker_batches)} workers")
        logger.info(f"Total duels: {len(tasks)}")
        
        start_time = time.time()
        
        # Initialize CSV file with headers
        self._initialize_csv_file()
        
        completed_tasks = 0
        
        # Execute worker batches in parallel
        with ProcessPoolExecutor(max_workers=len(worker_batches)) as executor:
            # Submit all worker batches
            future_to_batch = {
                executor.submit(worker_process_function, batch, self.config): batch 
                for batch in worker_batches
            }
            
            # Process completed batches as they finish
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                
                try:
                    batch_results = future.result()
                    self.results.extend(batch_results)
                    completed_tasks += len(batch_results)
                    
                    # Write results to CSV immediately
                    for result in batch_results:
                        self._append_result_to_csv(result)
                    
                    # Log progress
                    elapsed = time.time() - start_time
                    rate = completed_tasks / elapsed if elapsed > 0 else 0
                    eta = (len(tasks) - completed_tasks) / rate if rate > 0 else 0
                    
                    logger.info(f"Worker {batch.worker_id} completed {len(batch_results)} duels")
                    logger.info(f"Progress: {completed_tasks}/{len(tasks)} "
                              f"({completed_tasks/len(tasks)*100:.1f}%) "
                              f"Rate: {rate:.1f} duels/sec "
                              f"ETA: {eta/60:.1f} min")
                    
                except Exception as e:
                    logger.error(f"Worker batch {batch.worker_id} failed: {e}")
        
        total_time = time.time() - start_time
        logger.info(f"Optimized parallel evaluation completed in {total_time/60:.1f} minutes")
        logger.info(f"Total duels: {len(self.results)}")
        logger.info(f"Average rate: {len(self.results)/total_time:.1f} duels/second")
        
        # Generate summary
        self._generate_summary()
        
        return self.results
    
    def _initialize_csv_file(self):
        """Initialize the CSV file with headers."""
        with open(self.config.output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'novel_deck', 'known_deck', 'result', 'who_went_first',
                'turns', 'duration', 'task_id', 'worker_id', 'error'
            ])
    
    def _append_result_to_csv(self, outcome: DuelOutcome):
        """Append a single result to the CSV file."""
        with open(self.config.output_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                outcome.novel_deck,
                outcome.known_deck,
                outcome.result,
                outcome.who_went_first,
                outcome.turns,
                outcome.duration,
                outcome.task_id,
                outcome.worker_id,
                outcome.error or ""
            ])
    
    def _generate_summary(self):
        """Generate and save a summary of the evaluation results."""
        if not self.results:
            return
        
        # Calculate statistics
        total_duels = len(self.results)
        successful_duels = len([r for r in self.results if r.error is None])
        error_duels = total_duels - successful_duels
        
        # Worker performance analysis
        worker_stats = {}
        for result in self.results:
            worker_id = result.worker_id
            if worker_id not in worker_stats:
                worker_stats[worker_id] = {'total': 0, 'successful': 0, 'avg_duration': 0}
            
            worker_stats[worker_id]['total'] += 1
            if result.error is None:
                worker_stats[worker_id]['successful'] += 1
                worker_stats[worker_id]['avg_duration'] += result.duration
        
        # Calculate average durations per worker
        for worker_id in worker_stats:
            if worker_stats[worker_id]['successful'] > 0:
                worker_stats[worker_id]['avg_duration'] /= worker_stats[worker_id]['successful']
        
        # Calculate win rates per novel deck
        novel_deck_stats = {}
        for result in self.results:
            if result.error is None and result.result in ['win', 'loss']:
                deck = result.novel_deck
                if deck not in novel_deck_stats:
                    novel_deck_stats[deck] = {'wins': 0, 'total': 0}
                
                novel_deck_stats[deck]['total'] += 1
                if result.result == 'win':
                    novel_deck_stats[deck]['wins'] += 1
        
        # Calculate overall statistics
        total_wins = sum(stats['wins'] for stats in novel_deck_stats.values())
        total_games = sum(stats['total'] for stats in novel_deck_stats.values())
        overall_win_rate = total_wins / total_games if total_games > 0 else 0
        
        # Create summary
        summary = {
            'evaluation_config': {
                'k_known_decks': self.config.k_known_decks,
                'n_duels_per_pair': self.config.n_duels_per_pair,
                'parallel_workers': self.config.parallel_workers,
                'batch_size': self.config.batch_size
            },
            'statistics': {
                'total_duels': total_duels,
                'successful_duels': successful_duels,
                'error_duels': error_duels,
                'success_rate': successful_duels / total_duels if total_duels > 0 else 0,
                'overall_novel_win_rate': overall_win_rate
            },
            'worker_performance': worker_stats,
            'novel_deck_performance': {
                deck: {
                    'wins': stats['wins'],
                    'total': stats['total'],
                    'win_rate': stats['wins'] / stats['total'] if stats['total'] > 0 else 0
                }
                for deck, stats in novel_deck_stats.items()
            }
        }
        
        # Save summary
        summary_file = Path(self.config.output_file).with_suffix('.summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary saved to: {summary_file}")
        logger.info(f"Overall novel deck win rate: {overall_win_rate:.2%}")
        logger.info(f"Success rate: {successful_duels}/{total_duels} "
                   f"({successful_duels/total_duels*100:.1f}%)")

def main():
    """Main function to run optimized parallel deck evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimized Parallel Yu-Gi-Oh! Deck Evaluation")
    parser.add_argument("--known-dir", default="/home/ecast229/Projects/yugioh-deck-generator/decks/known",
                      help="Directory containing known deck files")
    parser.add_argument("--novel-dir", default="/home/ecast229/Projects/yugioh-deck-generator/decks/novel", 
                      help="Directory containing novel deck files")
    parser.add_argument("--output", default="outputs/optimized_deck_evaluation_results.csv",
                      help="Output CSV file path")
    parser.add_argument("--k", type=int, default=10,
                      help="Number of known decks to sample per novel deck")
    parser.add_argument("--n", type=int, default=20,
                      help="Number of duels per novel-known pair")
    parser.add_argument("--parallel-workers", type=int, default=0,
                      help="Number of parallel workers (0 = auto-detect all CPUs)")
    parser.add_argument("--base-port", type=int, default=8000,
                      help="Base port for YGOPro servers")
    parser.add_argument("--port-range", type=int, default=100,
                      help="Port range per worker")
    parser.add_argument("--batch-size", type=int, default=50,
                      help="Number of duels per batch for progress tracking")
    parser.add_argument("--timeout", type=int, default=180,
                      help="Timeout per duel in seconds")
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug logging")
    parser.add_argument("--save-replays", action="store_true",
                      help="Save replay files")
    
    args = parser.parse_args()
    
    # Create configuration
    config = OptimizedEvaluationConfig(
        known_decks_dir=args.known_dir,
        novel_decks_dir=args.novel_dir,
        output_file=args.output,
        k_known_decks=args.k,
        n_duels_per_pair=args.n,
        parallel_workers=args.parallel_workers,
        base_port=args.base_port,
        port_range=args.port_range,
        batch_size=args.batch_size,
        duel_timeout=args.timeout,
        save_replays=args.save_replays,
        debug=args.debug
    )
    
    # Run evaluation
    evaluator = OptimizedParallelDeckEvaluator(config)
    results = evaluator.run_optimized_evaluation()
    
    print("\n🎉 Optimized parallel deck evaluation completed!")
    print(f"Results saved to: {config.output_file}")
    print(f"Total duels simulated: {len(results)}")

if __name__ == "__main__":
    main()
