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
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, "/home/ecast229/Projects/yugioh-deck-generator/src")

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
    from duel_simulation.windbot_automation.optimized_parallel_evaluation import run_parallel_evaluation
    
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
    
    # Filter promising decks
    promising_decks = []
    for result in quick_results:
        if result.get('win_rate', 0) >= quick_threshold:
            promising_decks.append(result['deck_path'])
    
    logger.info(f"🎯 Found {len(promising_decks)}/{len(sample_deck_paths)} promising decks (≥{quick_threshold:.0%} win rate)")
    
    if not promising_decks:
        logger.warning("No promising decks found. Returning quick screening results only.")
        return quick_results
    
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
    
    # Combine results
    all_results = []
    detailed_deck_paths = {r['deck_path'] for r in detailed_results}
    
    # Add detailed results (these override quick results)
    for result in detailed_results:
        result['evaluation_stage'] = 'detailed'
        all_results.append(result)
    
    # Add quick results for non-promising decks
    for result in quick_results:
        if result['deck_path'] not in detailed_deck_paths:
            result['evaluation_stage'] = 'quick_only'
            all_results.append(result)
    
    # Save combined results
    df = pd.DataFrame(all_results)
    csv_path = Path(output_dir) / "evaluation_results.csv"
    df.to_csv(csv_path, index=False)
    
    total_time = quick_time + detailed_time
    total_duels = len(sample_deck_paths) * len(meta_decks) * quick_duels + len(promising_decks) * len(meta_decks) * detailed_duels
    
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
    
    args = parser.parse_args()
    
    # Use all CPU cores if not specified
    if args.max_workers == 0:
        args.max_workers = mp.cpu_count()
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    
    if args.quick_test:
        logger.info("🧪 Quick test mode enabled")
    
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
    
    # Estimated time (assuming 17.5s per duel, but with parallelization)
    estimated_time_serial = total_estimated_duels * 17.5 / 60  # minutes
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
    
def main_with_args(args, logger):
    """Main function separated for profiling."""
    start_time = time.time()
    
    try:
        # Step 1: Create sample of novel decks
        logger.info("📂 Step 1/4: Sampling novel decks...")
        novel_dir = Path("/home/ecast229/Projects/yugioh-deck-generator/decks/novel")
        sample_dir = Path(f"{args.output_dir}/sampled_novel_decks")
        
        actual_sample_size = create_sample_novel_decks(novel_dir, sample_dir, args.sample_size, logger)
        
        # Step 2: Fast meta analysis
        logger.info("🔍 Step 2/4: Fast meta analysis...")
        meta_decks = run_fast_meta_analysis(
            "/home/ecast229/Projects/yugioh-deck-generator/decks/known",
            args.output_dir,
            args.meta_size,
            logger
        )
        
        # Step 3: Run hierarchical evaluation
        logger.info("⚡ Step 3/4: Running hierarchical evaluation...")
        results = run_hierarchical_evaluation(
            str(sample_dir),
            meta_decks,
            args.output_dir,
            args.max_workers,
            args.quick_duels,
            args.detailed_duels,
            args.quick_threshold,
            logger
        )
        
        # Step 4: Generate analysis
        logger.info("📊 Step 4/4: Generating analysis...")
        analysis = generate_analysis(args.output_dir, results, logger)
        
        total_time = time.time() - start_time
        logger.info(f"🎉 HIGH PERFORMANCE ML training data generation completed in {total_time/60:.1f} minutes!")
        
        print(f"\n🎉 COMPLETED!")
        print(f"Runtime: {total_time/60:.1f} minutes")
        print(f"Results: {args.output_dir}/evaluation_results.csv")
        print(f"Log: {args.output_dir}/ml_training_hp.log")
        print(f"Analysis: {args.output_dir}/analysis.json")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error in HIGH PERFORMANCE ML training data generation: {e}")
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
    
    args = parser.parse_args()
    
    # Use all CPU cores if not specified
    if args.max_workers == 0:
        args.max_workers = mp.cpu_count()
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    
    if args.quick_test:
        logger.info("🧪 Quick test mode enabled")
    
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
    
    # Estimated time (assuming 17.5s per duel, but with parallelization)
    estimated_time_serial = total_estimated_duels * 17.5 / 60  # minutes
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
