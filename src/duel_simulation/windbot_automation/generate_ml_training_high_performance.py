#!/usr/bin/env python3
"""
HIGH PERFORMANCE ML Training Data Generation

Optimized for large-scale evaluation with:
- Massive parallelization (use all CPU cores)
- Hierarchical evaluation (quick screening + detailed evaluation)
- Server reuse and batch processing
- Progress tracking and checkpointing
"""

import os
import sys
import time
import json
import random
import logging
import argparse
import pandas as pd
import multiprocessing as mp
import cProfile
import pstats
import io
from pathlib import Path
from typing import List, Dict, Any, Set

# Add src to path
sys.path.insert(0, "/home/ecast229/Projects/yugioh-deck-generator/src")

from duel_simulation.windbot_automation.optimized_parallel_evaluation import (
    OptimizedParallelDeckEvaluator, 
    OptimizedEvaluationConfig
)

def load_checkpoint(checkpoint_file: Path) -> Dict[str, Any]:
    """Load checkpoint data from file."""
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load checkpoint: {e}")
    return {
        'completed_decks': [],
        'total_duels_completed': 0,
        'start_time': time.time(),
        'last_checkpoint_time': time.time()
    }

def save_checkpoint(checkpoint_file: Path, checkpoint_data: Dict[str, Any]):
    """Save checkpoint data to file."""
    try:
        checkpoint_data['last_checkpoint_time'] = time.time()
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
    except Exception as e:
        logging.error(f"Failed to save checkpoint: {e}")

def get_remaining_decks(all_decks: List[Path], completed_decks: List[str]) -> List[Path]:
    """Get list of decks that haven't been completed yet."""
    completed_set = set(completed_decks)
    remaining = [deck for deck in all_decks if deck.stem not in completed_set]
    return remaining

def append_results_to_csv(csv_file: Path, results: List[Dict[str, Any]]):
    """Append results to CSV file."""
    if not results:
        return
    
    df = pd.DataFrame(results)
    
    # If file doesn't exist, create it with headers
    if not csv_file.exists():
        df.to_csv(csv_file, index=False)
    else:
        # Append without headers
        df.to_csv(csv_file, mode='a', header=False, index=False)

def run_long_running_evaluation(
    novel_decks_dir: str,
    meta_decks: List[str],
    duels_per_deck: int,
    max_workers: int,
    output_dir: str,
    checkpoint_interval: int = 100,  # Save checkpoint every N decks
    batch_size: int = 50,  # Process decks in batches
    logger = None
) -> Dict[str, Any]:
    """
    Run long-running evaluation with checkpointing and resume capability.
    
    Args:
        novel_decks_dir: Directory containing novel decks
        meta_decks: List of meta deck paths
        duels_per_deck: Number of duels per deck
        max_workers: Number of parallel workers
        output_dir: Output directory
        checkpoint_interval: Save checkpoint every N decks
        batch_size: Process decks in batches
        logger: Logger instance
    
    Returns:
        Summary statistics
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # File paths
    checkpoint_file = output_path / "checkpoint.json"
    results_csv = output_path / "evaluation_results.csv"
    progress_log = output_path / "progress.log"
    
    # Load checkpoint
    checkpoint_data = load_checkpoint(checkpoint_file)
    
    # Get all novel decks
    all_novel_decks = list(Path(novel_decks_dir).glob("*.ydk"))
    logger.info(f"Found {len(all_novel_decks)} total novel decks")
    
    # Get remaining decks
    remaining_decks = get_remaining_decks(all_novel_decks, checkpoint_data['completed_decks'])
    logger.info(f"Remaining decks to process: {len(remaining_decks)}")
    
    if not remaining_decks:
        logger.info("✅ All decks already completed!")
        return checkpoint_data
    
    # Show resume information
    if checkpoint_data['completed_decks']:
        logger.info(f"🔄 Resuming from checkpoint:")
        logger.info(f"  ✅ Already completed: {len(checkpoint_data['completed_decks'])} decks")
        logger.info(f"  ⏱️  Previous runtime: {(checkpoint_data.get('last_checkpoint_time', checkpoint_data['start_time']) - checkpoint_data['start_time'])/3600:.1f}h")
        logger.info(f"  🎯 Last completed deck: {checkpoint_data['completed_decks'][-1]}")
    else:
        logger.info("🆕 Starting fresh evaluation (no previous checkpoint found)")
    
    # Process decks in batches
    total_processed = len(checkpoint_data['completed_decks'])
    total_decks = len(all_novel_decks)
    
    logger.info(f"🚀 Starting long-running evaluation:")
    logger.info(f"  Total decks: {total_decks}")
    logger.info(f"  Already completed: {total_processed}")
    logger.info(f"  Remaining: {len(remaining_decks)}")
    logger.info(f"  Duels per deck: {duels_per_deck}")
    logger.info(f"  Meta decks: {len(meta_decks)}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Checkpoint interval: {checkpoint_interval}")
    
    # Process in batches
    for i in range(0, len(remaining_decks), batch_size):
        batch_decks = remaining_decks[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(remaining_decks) + batch_size - 1) // batch_size
        
        logger.info(f"🔄 Processing batch {batch_num}/{total_batches} ({len(batch_decks)} decks)")
        
        # Process this batch
        batch_results = run_parallel_evaluation(
            novel_deck_paths=[str(deck) for deck in batch_decks],
            meta_deck_paths=meta_decks,
            rounds_per_matchup=duels_per_deck,
            max_workers=max_workers,
            output_dir=str(output_path),
            stage_name=f"batch_{batch_num:04d}"
        )
        
        # Convert to simple format and append to CSV
        simple_results = []
        for result in batch_results:
            # Clean deck name
            deck_name = result['deck_name']
            if 'sample_' in deck_name:
                parts = deck_name.split('_')
                sample_index = -1
                for j, part in enumerate(parts):
                    if part == 'sample' and j < len(parts) - 1:
                        sample_index = j + 2
                        break
                if sample_index >= 0 and sample_index < len(parts):
                    deck_name = '_'.join(parts[sample_index:])
            
            simple_results.append({
                'deck_name': deck_name,
                'win_rate': result['win_rate'],
                'total_duels': result['total_duels'],
                'wins': result['wins'],
                'losses': result['losses'],
                'draws': result['draws'],
                'errors': result['errors'],
                'avg_duration': result['avg_duration']
            })
        
        # Append results to CSV
        append_results_to_csv(results_csv, simple_results)
        
        # Update checkpoint
        for deck in batch_decks:
            checkpoint_data['completed_decks'].append(deck.stem)
        
        checkpoint_data['total_duels_completed'] += sum(r['total_duels'] for r in batch_results)
        
        # Note: With random sampling, each novel deck plays exactly duels_per_deck duels
        # against randomly selected meta decks (not all meta decks)
        
        # Save checkpoint after every batch (for resumability)
        save_checkpoint(checkpoint_file, checkpoint_data)
        
        # Log checkpoint save periodically
        if len(checkpoint_data['completed_decks']) % checkpoint_interval == 0:
            logger.info(f"📁 Checkpoint saved at {len(checkpoint_data['completed_decks'])} decks")
        
        # Progress update
        completed_now = len(checkpoint_data['completed_decks'])
        progress_pct = (completed_now / total_decks) * 100
        elapsed_time = time.time() - checkpoint_data['start_time']
        
        if completed_now > 0:
            avg_time_per_deck = elapsed_time / completed_now
            eta_seconds = avg_time_per_deck * (total_decks - completed_now)
            eta_hours = eta_seconds / 3600
            
            logger.info(f"📊 Progress: {completed_now}/{total_decks} ({progress_pct:.1f}%)")
            logger.info(f"⏱️  Elapsed: {elapsed_time/3600:.1f}h, ETA: {eta_hours:.1f}h")
            logger.info(f"📈 Avg: {avg_time_per_deck:.1f}s/deck, Total duels: {checkpoint_data['total_duels_completed']}")
            logger.info(f"💾 Checkpoint saved - can resume from batch {batch_num + 1} if interrupted")
    
    # Final checkpoint save
    save_checkpoint(checkpoint_file, checkpoint_data)
    
    logger.info(f"🎉 Long-running evaluation completed!")
    logger.info(f"📊 Final stats:")
    logger.info(f"  Total decks: {len(checkpoint_data['completed_decks'])}")
    logger.info(f"  Total duels: {checkpoint_data['total_duels_completed']}")
    logger.info(f"  Total time: {(time.time() - checkpoint_data['start_time'])/3600:.1f} hours")
    logger.info(f"  Results saved to: {results_csv}")
    
    return checkpoint_data

def run_parallel_evaluation(novel_deck_paths: List[str], meta_deck_paths: List[str], 
                           rounds_per_matchup: int, max_workers: int, 
                           output_dir: str, stage_name: str) -> List[Dict]:
    """
    Wrapper function to run parallel evaluation using OptimizedParallelDeckEvaluator.
    
    Args:
        novel_deck_paths: List of paths to novel deck files
        meta_deck_paths: List of paths to meta deck files  
        rounds_per_matchup: Number of duels per deck pair
        max_workers: Number of parallel workers
        output_dir: Output directory
        stage_name: Name of the evaluation stage (for logging)
    
    Returns:
        List of evaluation results
    """
    # Create temporary directories for this evaluation
    temp_novel_dir = Path(output_dir) / f"temp_{stage_name}_novel"
    temp_known_dir = Path(output_dir) / f"temp_{stage_name}_known"
    
    # Clean temp directories first to avoid leftover files
    import shutil
    if temp_novel_dir.exists():
        shutil.rmtree(temp_novel_dir)
    if temp_known_dir.exists():
        shutil.rmtree(temp_known_dir)
    
    temp_novel_dir.mkdir(parents=True, exist_ok=True)
    temp_known_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy deck files to temporary directories
    for i, novel_path in enumerate(novel_deck_paths):
        novel_file = Path(novel_path)
        temp_file = temp_novel_dir / f"{stage_name}_novel_{i:04d}_{novel_file.name}"
        temp_file.write_text(novel_file.read_text())
    
    for i, meta_path in enumerate(meta_deck_paths):
        meta_file = Path(meta_path)
        temp_file = temp_known_dir / f"{stage_name}_meta_{i:04d}_{meta_file.name}"
        temp_file.write_text(meta_file.read_text())
    
    # For random sampling mode: create multiple copies of meta decks to enable random selection
    # Each novel deck will play against randomly sampled meta decks
    expanded_meta_dir = Path(output_dir) / f"temp_{stage_name}_expanded_meta"
    if expanded_meta_dir.exists():
        import shutil
        shutil.rmtree(expanded_meta_dir)
    expanded_meta_dir.mkdir(parents=True, exist_ok=True)
    
    # Create enough meta deck copies for random sampling (rounds_per_matchup copies of each)
    meta_deck_copies = []
    for round_num in range(rounds_per_matchup):
        for i, meta_path in enumerate(meta_deck_paths):
            meta_file = Path(meta_path)
            copy_file = expanded_meta_dir / f"{stage_name}_meta_r{round_num:02d}_{i:04d}_{meta_file.name}"
            copy_file.write_text(meta_file.read_text())
            meta_deck_copies.append(str(copy_file))
    
    # Create configuration for 1 duel per pair (since we're doing random sampling)
    config = OptimizedEvaluationConfig(
        known_decks_dir=str(expanded_meta_dir),
        novel_decks_dir=str(temp_novel_dir),
        output_file=f"{output_dir}/{stage_name}_results.csv",
        k_known_decks=rounds_per_matchup,  # Only select this many random meta decks per novel deck
        n_duels_per_pair=1,  # 1 duel per randomly selected meta deck
        parallel_workers=max_workers,
        base_port=8000,
        port_range=100,
        duel_timeout=60,
        batch_size=20,
        save_replays=False,
        debug=False
    )
    
    # Run evaluation
    evaluator = OptimizedParallelDeckEvaluator(config)
    results = evaluator.run_optimized_evaluation()
    
    # Convert results to expected format
    # First, aggregate individual duel outcomes by novel deck
    deck_stats = {}
    for outcome in results:
        novel_deck = outcome.novel_deck
        if novel_deck not in deck_stats:
            deck_stats[novel_deck] = {
                'wins': 0,
                'losses': 0,
                'draws': 0,
                'errors': 0,
                'total_duels': 0,
                'total_duration': 0
            }
        
        deck_stats[novel_deck]['total_duels'] += 1
        deck_stats[novel_deck]['total_duration'] += outcome.duration
        
        if outcome.error:
            deck_stats[novel_deck]['errors'] += 1
        elif outcome.result == 'win':
            deck_stats[novel_deck]['wins'] += 1
        elif outcome.result == 'loss':
            deck_stats[novel_deck]['losses'] += 1
        else:  # draw or other
            deck_stats[novel_deck]['draws'] += 1
    
    # Create formatted results
    formatted_results = []
    for novel_deck, stats in deck_stats.items():
        # Extract original deck names from temporary file names
        novel_name = Path(novel_deck).stem
        if novel_name.startswith(f"{stage_name}_novel_"):
            novel_name = "_".join(novel_name.split("_")[3:])  # Remove temp prefix
        
        formatted_results.append({
            'deck_path': novel_deck,
            'deck_name': novel_name,
            'win_rate': stats['wins'] / max(stats['total_duels'], 1),
            'total_duels': stats['total_duels'],
            'wins': stats['wins'],
            'losses': stats['losses'],
            'draws': stats['draws'],
            'errors': stats['errors'],
            'avg_duration': stats['total_duration'] / max(stats['total_duels'], 1)
        })
    
    # Cleanup temporary directories
    import shutil
    shutil.rmtree(temp_novel_dir, ignore_errors=True)
    shutil.rmtree(temp_known_dir, ignore_errors=True)
    shutil.rmtree(expanded_meta_dir, ignore_errors=True)
    
    return formatted_results

def setup_logging(output_dir: str):
    """Set up logging for high performance run."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    log_file = output_path / "ml_training_hp.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ],
        force=True
    )
    return logging.getLogger(__name__)

def create_sample_novel_decks(novel_dir: Path, sample_dir: Path, sample_size: int, logger):
    """Create a sample of novel decks for evaluation."""
    novel_decks = list(novel_dir.glob("*.ydk"))
    if len(novel_decks) < sample_size:
        logger.warning(f"Only {len(novel_decks)} novel decks available, using all")
        sample_size = len(novel_decks)
    
    # Random sample
    sampled_decks = random.sample(novel_decks, sample_size)
    
    # Create sample directory (clean it first)
    if sample_dir.exists():
        import shutil
        shutil.rmtree(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy sampled decks
    for i, deck in enumerate(sampled_decks):
        sample_path = sample_dir / f"sample_{i:04d}_{deck.name}"
        sample_path.write_text(deck.read_text())
    
    logger.info(f"✅ Created sample of {sample_size} novel decks in {sample_dir}")
    return sample_size

def run_fast_meta_analysis(known_dir: str, output_dir: str, meta_size: int, logger):
    """Run FAST meta analysis using random sampling."""
    logger.info("🔍 Running FAST meta analysis (random sampling)...")
    
    # Get all known decks
    known_decks = list(Path(known_dir).glob("*.ydk"))
    logger.info(f"Found {len(known_decks)} known decks")
    
    # Use random sampling for speed
    if len(known_decks) <= meta_size:
        meta_deck_paths = [str(p) for p in known_decks]
        logger.info(f"Using all {len(meta_deck_paths)} known decks as meta")
    else:
        sampled_decks = random.sample(known_decks, meta_size)
        meta_deck_paths = [str(p) for p in sampled_decks]
        logger.info(f"Randomly sampled {len(meta_deck_paths)} decks as meta")
    
    # Save meta deck info
    meta_info = {
        'meta_decks': [Path(p).stem for p in meta_deck_paths],
        'meta_deck_paths': meta_deck_paths,
        'total_known_decks': len(known_decks),
        'meta_size': len(meta_deck_paths),
        'selection_method': 'random_sampling'
    }
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "meta_analysis.json", 'w') as f:
        json.dump(meta_info, f, indent=2)
    
    logger.info(f"✅ FAST meta analysis complete. Selected {len(meta_deck_paths)} decks")
    return meta_deck_paths

def run_hierarchical_evaluation(sample_dir: str, meta_decks: list, output_dir: str, 
                               max_workers: int, quick_duels: int, detailed_duels: int, 
                               quick_threshold: float, logger):
    """
    Run hierarchical evaluation:
    1. Quick screening with few duels
    2. Detailed evaluation for promising decks
    """
    
    logger.info(f"🎯 Starting HIERARCHICAL evaluation with {max_workers} workers")
    logger.info(f"Stage 1: Quick screening ({quick_duels} duels per deck)")
    logger.info(f"Stage 2: Detailed evaluation ({detailed_duels} duels for decks with ≥{quick_threshold:.0%} win rate)")
    
    # Get sample decks
    sample_decks = list(Path(sample_dir).glob("*.ydk"))
    sample_deck_paths = [str(p) for p in sample_decks]
    
    logger.info(f"Evaluating {len(sample_deck_paths)} decks against {len(meta_decks)} meta decks")
    
    # Stage 1: Quick screening
    logger.info("🚀 Stage 1: Quick screening...")
    quick_start = time.time()
    
    quick_results = run_parallel_evaluation(
        novel_deck_paths=sample_deck_paths,
        meta_deck_paths=meta_decks,
        rounds_per_matchup=quick_duels,
        max_workers=max_workers,
        output_dir=output_dir,
        stage_name="quick_screening"
    )
    
    quick_time = time.time() - quick_start
    logger.info(f"✅ Quick screening completed in {quick_time/60:.1f} minutes")
    
    # Filter promising decks - map back to original deck paths
    promising_decks = []
    deck_name_to_path = {Path(p).stem: p for p in sample_deck_paths}
    
    for result in quick_results:
        if result.get('win_rate', 0) >= quick_threshold:
            # Map the deck name back to the original path
            deck_name = result['deck_name']
            if deck_name in deck_name_to_path:
                promising_decks.append(deck_name_to_path[deck_name])
            else:
                # Fallback: try to find by matching the deck name
                for original_path in sample_deck_paths:
                    if Path(original_path).stem == deck_name:
                        promising_decks.append(original_path)
                        break
    
    logger.info(f"🎯 Found {len(promising_decks)}/{len(sample_deck_paths)} promising decks (≥{quick_threshold:.0%} win rate)")
    
    if not promising_decks:
        logger.warning("No promising decks found. Returning quick screening results only.")
        # Set evaluation_stage for all quick results
        for result in quick_results:
            result['evaluation_stage'] = 'quick_only'
        all_results = quick_results
    else:
        # Stage 2: Detailed evaluation of promising decks
        logger.info(f"🔬 Stage 2: Detailed evaluation of {len(promising_decks)} promising decks...")
        detailed_start = time.time()
        
        detailed_results = run_parallel_evaluation(
            novel_deck_paths=promising_decks,
            meta_deck_paths=meta_decks,
            rounds_per_matchup=detailed_duels,
            max_workers=max_workers,
            output_dir=output_dir,
            stage_name="detailed_evaluation"
        )
        
        detailed_time = time.time() - detailed_start
        logger.info(f"✅ Detailed evaluation completed in {detailed_time/60:.1f} minutes")
        
        # Combine results - need to map by deck names since paths are different
        all_results = []
        detailed_deck_names = {r['deck_name'] for r in detailed_results}
        
        # Add detailed results (these override quick results)
        for result in detailed_results:
            result['evaluation_stage'] = 'detailed'
            all_results.append(result)
        
        # Add quick results for non-promising decks
        for result in quick_results:
            if result['deck_name'] not in detailed_deck_names:
                result['evaluation_stage'] = 'quick_only'
                all_results.append(result)
    
    # Save combined results in simple format: deck_name, win_rate
    simple_results = []
    for result in all_results:
        # Get original deck name by removing all prefixes
        deck_name = result['deck_name']
        # The deck_name comes from the temp file name after removing stage prefix
        # It should be in format: stage_novel_XXXX_sample_XXXX_original_name
        # We need to find "sample_XXXX_" and remove everything up to and including it
        
        # Find the "sample_" part and extract everything after "sample_XXXX_"
        if 'sample_' in deck_name:
            # Split by underscores and find where "sample" appears
            parts = deck_name.split('_')
            sample_index = -1
            for i, part in enumerate(parts):
                if part == 'sample' and i < len(parts) - 1:
                    # Found "sample", skip the next part (the number) and take the rest
                    sample_index = i + 2  # Skip "sample" and the number after it
                    break
            
            if sample_index >= 0 and sample_index < len(parts):
                original_deck_name = '_'.join(parts[sample_index:])
            else:
                original_deck_name = deck_name
        else:
            original_deck_name = deck_name

        simple_results.append({
            'deck_name': original_deck_name,
            'win_rate': result['win_rate']
        })
    
    # Save simple CSV
    simple_df = pd.DataFrame(simple_results)
    csv_path = Path(output_dir) / "evaluation_results.csv"
    simple_df.to_csv(csv_path, index=False)
    
    # Calculate timing info
    if 'detailed_time' in locals():
        total_time = quick_time + detailed_time
        total_duels = len(sample_deck_paths) * len(meta_decks) * quick_duels + len(promising_decks) * len(meta_decks) * detailed_duels
    else:
        total_time = quick_time
        total_duels = len(sample_deck_paths) * len(meta_decks) * quick_duels
    
    logger.info(f"🎉 Hierarchical evaluation complete!")
    logger.info(f"Total time: {total_time/60:.1f} minutes")
    logger.info(f"Total duels: {total_duels:,}")
    logger.info(f"Average: {total_time/total_duels:.1f} seconds per duel")
    logger.info(f"Results saved to: {csv_path}")
    
    return all_results

def generate_analysis(output_dir: str, results: list, logger):
    """Generate analysis from results."""
    logger.info("📊 Generating analysis...")
    
    if not results:
        logger.warning("No results to analyze")
        return
    
    # Basic statistics
    valid_results = [r for r in results if r.get('total_duels', 0) > 0]
    
    if not valid_results:
        logger.warning("No valid results to analyze")
        return
    
    # Separate by evaluation stage
    quick_only = [r for r in valid_results if r.get('evaluation_stage') == 'quick_only']
    detailed = [r for r in valid_results if r.get('evaluation_stage') == 'detailed']
    
    analysis = {
        'summary': {
            'total_novel_decks': len(results),
            'successful_evaluations': len(valid_results),
            'quick_only_evaluations': len(quick_only),
            'detailed_evaluations': len(detailed),
            'overall_avg_win_rate': sum(r['win_rate'] for r in valid_results) / len(valid_results),
            'total_duels': sum(r['total_duels'] for r in valid_results),
            'promising_deck_threshold': len(detailed) / len(valid_results) if valid_results else 0
        },
        'quick_screening': {
            'avg_win_rate': sum(r['win_rate'] for r in quick_only) / len(quick_only) if quick_only else 0,
            'count': len(quick_only)
        },
        'detailed_evaluation': {
            'avg_win_rate': sum(r['win_rate'] for r in detailed) / len(detailed) if detailed else 0,
            'count': len(detailed)
        }
    }
    
    # Save analysis
    analysis_path = Path(output_dir) / "analysis.json"
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # Log summary
    logger.info(f"📈 Analysis Summary:")
    logger.info(f"  Total decks evaluated: {analysis['summary']['total_novel_decks']}")
    logger.info(f"  Quick screening only: {analysis['quick_screening']['count']} (avg {analysis['quick_screening']['avg_win_rate']:.1%} win rate)")
    logger.info(f"  Detailed evaluation: {analysis['detailed_evaluation']['count']} (avg {analysis['detailed_evaluation']['avg_win_rate']:.1%} win rate)")
    logger.info(f"  Overall avg win rate: {analysis['summary']['overall_avg_win_rate']:.1%}")
    logger.info(f"  Total duels: {analysis['summary']['total_duels']:,}")
    
    return analysis

def run_with_profiling(main_func, args, logger):
    """Run the main function with profiling enabled."""
    logger.info("🔬 Profiling enabled - detailed performance analysis will be generated")
    
    # Create profiler
    profiler = cProfile.Profile()
    
    # Run with profiling
    profiler.enable()
    result = main_func(args, logger)
    profiler.disable()
    
    # Save profiling results
    profile_dir = Path(args.output_dir) / "profiling"
    profile_dir.mkdir(exist_ok=True)
    
    # Save raw profile data
    profile_file = profile_dir / "profile.prof"
    profiler.dump_stats(str(profile_file))
    
    # Generate human-readable report
    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s)
    stats.sort_stats('cumulative')
    stats.print_stats(args.profile_top)
    
    # Save text report
    report_file = profile_dir / "profile_report.txt"
    with open(report_file, 'w') as f:
        f.write(s.getvalue())
    
    # Log summary
    logger.info(f"📊 Profiling complete:")
    logger.info(f"  Raw data: {profile_file}")
    logger.info(f"  Report: {report_file}")
    
    # Show top functions in log
    logger.info("🔥 Top time-consuming functions:")
    lines = s.getvalue().split('\n')[5:15]  # Skip header, show top 10
    for line in lines:
        if line.strip():
            logger.info(f"  {line}")
    
    return result

def main_with_args(args, logger):
    """Main function separated for profiling."""
    parser = argparse.ArgumentParser(description="High Performance ML Training Data Generation")
    parser.add_argument("--sample-size", type=int, default=100,
                      help="Number of novel decks to sample")
    parser.add_argument("--max-workers", type=int, default=0,
                      help="Maximum parallel workers (0 = use all CPU cores)")
    parser.add_argument("--meta-size", type=int, default=10,
                      help="Number of meta decks to select")
    parser.add_argument("--quick-duels", type=int, default=3,
                      help="Duels per deck in quick screening")
    parser.add_argument("--detailed-duels", type=int, default=10,
                      help="Duels per deck in detailed evaluation")
    parser.add_argument("--quick-threshold", type=float, default=0.4,
                      help="Win rate threshold for detailed evaluation (0.0-1.0)")
    parser.add_argument("--output-dir", default="outputs/ml_training_hp",
                      help="Output directory")
    parser.add_argument("--quick-test", action="store_true",
                      help="Skip confirmation prompts")
    parser.add_argument("--profile", action="store_true",
                      help="Enable detailed profiling")
    parser.add_argument("--profile-top", type=int, default=20,
                      help="Number of top functions to show in profile")
    
    # New arguments for long-running mode
    parser.add_argument("--long-running", action="store_true",
                      help="Run long-running evaluation on all novel decks")
    parser.add_argument("--duels-per-deck", type=int, default=7,
                      help="Number of duels per deck in long-running mode")
    parser.add_argument("--checkpoint-interval", type=int, default=100,
                      help="Save checkpoint every N decks")
    parser.add_argument("--batch-size", type=int, default=50,
                      help="Process decks in batches of N")
    
    args = parser.parse_args()
    
    # Use all CPU cores if not specified
    if args.max_workers == 0:
        args.max_workers = mp.cpu_count()
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    
    if args.quick_test:
        logger.info("🧪 Quick test mode enabled")
    
    # Long-running mode
    if args.long_running:
        return run_long_running_mode(args, logger)
    
    # Regular hierarchical mode
    logger.info("🚀 HIGH PERFORMANCE ML Training Data Generation")
    logger.info("=" * 60)
    logger.info(f"Sample size: {args.sample_size} novel decks")
    logger.info(f"Meta decks: {args.meta_size}")
    logger.info(f"Quick screening: {args.quick_duels} duels per deck")
    logger.info(f"Detailed evaluation: {args.detailed_duels} duels per deck (≥{args.quick_threshold:.0%} win rate)")
    logger.info(f"Max workers: {args.max_workers}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Calculate estimates
    total_quick_duels = args.sample_size * args.meta_size * args.quick_duels
    estimated_promising = int(args.sample_size * args.quick_threshold)
    total_detailed_duels = estimated_promising * args.meta_size * args.detailed_duels
    total_estimated_duels = total_quick_duels + total_detailed_duels
    
    # Estimated time (assuming 1.2s per duel, but with parallelization)
    estimated_time_serial = total_estimated_duels * 1.2 / 60  # minutes
    estimated_time_parallel = estimated_time_serial / args.max_workers  # with parallelization
    
    logger.info(f"Estimated total duels: {total_estimated_duels:,}")
    logger.info(f"  Quick screening: {total_quick_duels:,}")
    logger.info(f"  Detailed evaluation: {total_detailed_duels:,} (estimated)")
    logger.info(f"Estimated time (serial): {estimated_time_serial:.1f} minutes")
    logger.info(f"Estimated time (parallel): {estimated_time_parallel:.1f} minutes")
    
    if not args.quick_test and total_estimated_duels > 1000:
        response = input("⚠️  Continue with this configuration? (y/N): ")
        if response.lower() != 'y':
            logger.info("Cancelled by user")
            return 0
    
    # Run with or without profiling
    if args.profile:
        return run_with_profiling(main_with_args, args, logger)
    else:
        return main_with_args(args, logger)

def run_long_running_mode(args, logger):
    """Run long-running evaluation mode."""
    logger.info("🚀 LONG-RUNNING ML Training Data Generation")
    logger.info("=" * 60)
    
    # Count novel decks
    novel_dir = Path("/home/ecast229/Projects/yugioh-deck-generator/decks/novel")
    all_novel_decks = list(novel_dir.glob("*.ydk"))
    
    logger.info(f"Total novel decks: {len(all_novel_decks)}")
    logger.info(f"Meta decks: {args.meta_size}")
    logger.info(f"Duels per deck: {args.duels_per_deck}")
    logger.info(f"Max workers: {args.max_workers}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Checkpoint interval: {args.checkpoint_interval}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Calculate estimates (with random sampling: duels_per_deck duels per novel deck)
    total_estimated_duels = len(all_novel_decks) * args.duels_per_deck
    estimated_time_serial = total_estimated_duels * 1.2 / 3600  # hours
    estimated_time_parallel = estimated_time_serial / args.max_workers  # with parallelization
    
    logger.info(f"Estimated total duels: {total_estimated_duels:,}")
    logger.info(f"Estimated time (serial): {estimated_time_serial:.1f} hours")
    logger.info(f"Estimated time (parallel): {estimated_time_parallel:.1f} hours")
    
    if not args.quick_test:
        # Make the warning more appropriate based on estimated time
        if estimated_time_parallel >= 2.0:
            time_warning = f"This will run for approximately {estimated_time_parallel:.1f} hours"
        elif estimated_time_parallel >= 1.0:
            time_warning = f"This will run for approximately {estimated_time_parallel:.1f} hour"
        else:
            time_warning = f"This will run for approximately {estimated_time_parallel*60:.0f} minutes"
        
        response = input(f"⚠️  {time_warning}. Continue? (y/N): ")
        if response.lower() != 'y':
            logger.info("Cancelled by user")
            return 0
    
    try:
        # Fast meta analysis
        logger.info("🔍 Step 1/2: Fast meta analysis...")
        meta_decks = run_fast_meta_analysis(
            "/home/ecast229/Projects/yugioh-deck-generator/decks/known",
            args.output_dir,
            args.meta_size,
            logger
        )
        
        # Run long-running evaluation
        logger.info("⚡ Step 2/2: Running long-running evaluation...")
        results = run_long_running_evaluation(
            novel_decks_dir=str(novel_dir),
            meta_decks=meta_decks,
            duels_per_deck=args.duels_per_deck,
            max_workers=args.max_workers,
            output_dir=args.output_dir,
            checkpoint_interval=args.checkpoint_interval,
            batch_size=args.batch_size,
            logger=logger
        )
        
        logger.info("🎉 Long-running evaluation completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("🛑 Process interrupted by user. Progress has been saved.")
        logger.info("💡 Run the same command again to resume from where you left off.")
        return 1
    except Exception as e:
        logger.error(f"❌ Error in long-running evaluation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

def main():
    """Main function for high performance ML training data generation."""
    parser = argparse.ArgumentParser(description="High Performance ML Training Data Generation")
    parser.add_argument("--sample-size", type=int, default=100,
                      help="Number of novel decks to sample")
    parser.add_argument("--max-workers", type=int, default=0,
                      help="Maximum parallel workers (0 = use all CPU cores)")
    parser.add_argument("--meta-size", type=int, default=10,
                      help="Number of meta decks to select")
    parser.add_argument("--quick-duels", type=int, default=3,
                      help="Duels per deck in quick screening")
    parser.add_argument("--detailed-duels", type=int, default=10,
                      help="Duels per deck in detailed evaluation")
    parser.add_argument("--quick-threshold", type=float, default=0.4,
                      help="Win rate threshold for detailed evaluation (0.0-1.0)")
    parser.add_argument("--output-dir", default="outputs/ml_training_hp",
                      help="Output directory")
    parser.add_argument("--quick-test", action="store_true",
                      help="Skip confirmation prompts")
    parser.add_argument("--profile", action="store_true",
                      help="Enable detailed profiling")
    parser.add_argument("--profile-top", type=int, default=20,
                      help="Number of top functions to show in profile")
    
    # New arguments for long-running mode
    parser.add_argument("--long-running", action="store_true",
                      help="Run long-running evaluation on all novel decks")
    parser.add_argument("--duels-per-deck", type=int, default=7,
                      help="Number of duels per deck in long-running mode")
    parser.add_argument("--checkpoint-interval", type=int, default=100,
                      help="Save checkpoint every N decks")
    parser.add_argument("--batch-size", type=int, default=50,
                      help="Process decks in batches of N")
    
    args = parser.parse_args()
    
    # Use all CPU cores if not specified
    if args.max_workers == 0:
        args.max_workers = mp.cpu_count()
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    
    if args.quick_test:
        logger.info("🧪 Quick test mode enabled")
    
    # Long-running mode
    if args.long_running:
        return run_long_running_mode(args, logger)
    
    # Regular hierarchical mode
    logger.info("🚀 HIGH PERFORMANCE ML Training Data Generation")
    logger.info("=" * 60)
    logger.info(f"Sample size: {args.sample_size} novel decks")
    logger.info(f"Meta decks: {args.meta_size}")
    logger.info(f"Quick screening: {args.quick_duels} duels per deck")
    logger.info(f"Detailed evaluation: {args.detailed_duels} duels per deck (≥{args.quick_threshold:.0%} win rate)")
    logger.info(f"Max workers: {args.max_workers}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Calculate estimates
    total_quick_duels = args.sample_size * args.meta_size * args.quick_duels
    estimated_promising = int(args.sample_size * args.quick_threshold)
    total_detailed_duels = estimated_promising * args.meta_size * args.detailed_duels
    total_estimated_duels = total_quick_duels + total_detailed_duels
    
    # Estimated time (assuming 1.2s per duel, but with parallelization)
    estimated_time_serial = total_estimated_duels * 1.2 / 60  # minutes
    estimated_time_parallel = estimated_time_serial / args.max_workers  # with parallelization
    
    logger.info(f"Estimated total duels: {total_estimated_duels:,}")
    logger.info(f"  Quick screening: {total_quick_duels:,}")
    logger.info(f"  Detailed evaluation: {total_detailed_duels:,} (estimated)")
    logger.info(f"Estimated time (serial): {estimated_time_serial:.1f} minutes")
    logger.info(f"Estimated time (parallel): {estimated_time_parallel:.1f} minutes")
    
    if not args.quick_test and total_estimated_duels > 1000:
        response = input("⚠️  Continue with this configuration? (y/N): ")
        if response.lower() != 'y':
            logger.info("Cancelled by user")
            return 0
    
    # Run with or without profiling
    if args.profile:
        return run_with_profiling(main_with_args, args, logger)
    else:
        return main_with_args(args, logger)

if __name__ == "__main__":
    exit(main())
