import mlflow
from mlflow.tracking import MlflowClient

# Set the tracking URI to connect to the MLflow server
mlflow.set_tracking_uri("http://localhost:5000")

# Initialize MLflow client with the correct tracking URI
client = MlflowClient("http://localhost:5000")
experiment = client.get_experiment_by_name("/yugioh_card_clustering")

# First, let's search for all runs to see what metrics are available
runs = client.search_runs(
    experiment.experiment_id,
    filter_string="",  # Get all runs, no filter
    max_results=2000  # Increase to get more runs
)

print(f"Found {len(runs)} runs")

# Debug: Print available metrics from recent runs and look for noise_percentage
if runs:
    print("Available metrics in recent runs:")
    noise_runs = 0
    enhanced_runs = 0
    for i, run in enumerate(runs[:20]):  # Check first 20 runs
        metrics = list(run.data.metrics.keys())
        has_noise = 'noise_percentage' in metrics
        has_enhanced = any('enhanced_' in m for m in metrics)
        
        if has_noise:
            noise_runs += 1
        if has_enhanced:
            enhanced_runs += 1
            
        if i < 5:  # Show details for first 5 runs
            print(f"Run {i+1} metrics: {metrics[:10]}...")
            if has_noise:
                print(f"  -> HAS noise_percentage: {run.data.metrics['noise_percentage']}")
            if has_enhanced:
                enhanced_metrics = [m for m in metrics if m.startswith('enhanced_')][:5]
                print(f"  -> HAS enhanced metrics: {enhanced_metrics}")
    
    print(f"\nSummary: {noise_runs} runs with noise_percentage, {enhanced_runs} runs with enhanced metrics")

def run_score(run):
    silhouette = run.data.metrics.get("silhouette_score", -1)
    adjusted_rand = run.data.metrics.get("adjusted_rand_index", 0)
    
    noise = run.data.metrics.get("enhanced_noise_percentage", run.data.metrics.get("noise_percentage", 0))
    cluster_std = run.data.metrics.get("enhanced_cluster_size_std", 0)
    entropy = run.data.metrics.get("enhanced_archetype_entropy_mean", 0)
    n_clusters = run.data.metrics.get("enhanced_num_clusters", run.data.metrics.get("n_clusters_found", 0))

    is_kmeans = noise == 0 and cluster_std < 10  # likely K-Means
    
    # Silhouette component is still weighted heavily
    score = 0.7 * silhouette

    # Noise adjustment
    if 5 <= noise <= 15:
        score += 0.2  # reward good outlier detection
    elif 15 < noise <= 25:
        score += 0.05  # small bonus, still decent
    elif noise > 25:
        score -= 0.1 * (noise - 25) / 100  # gradual penalty

    # Penalize zero-noise clustering when cluster std is low (too uniform)
    if is_kmeans:
        score -= 0.1

    # Diversity bonus
    score += min(0.1, cluster_std / 100)

    # Archetype coherence (lower entropy → better clusters)
    if entropy > 0:
        score += max(0, 0.1 * (3.0 - entropy))  # ideal is < 2.5 entropy

    return score


# Select best run based on composite score
if runs:
    best_run = max(runs, key=run_score)
    
    print("\n" + "="*50)
    print("BEST RUN ANALYSIS")
    print("="*50)
    print("Best run ID:", best_run.info.run_id)
    print("Best composite score:", f"{run_score(best_run):.4f}")
    print("\nParameters:")
    for key, value in best_run.data.params.items():
        print(f"  {key}: {value}")
    print("\nMetrics:")
    for key, value in best_run.data.metrics.items():
        print(f"  {key}: {value:.4f}")
        
    # Show top 5 runs for comparison
    print("\n" + "="*50)
    print("TOP 5 RUNS COMPARISON")
    print("="*50)
    top_runs = sorted(runs, key=run_score, reverse=True)[:5]
    
    for i, run in enumerate(top_runs):
        print(f"\nRank {i+1}: Run {run.info.run_id[:8]}...")
        print(f"  Score: {run_score(run):.4f}")
        print(f"  Algorithm: {run.data.params.get('experiment_config', run.data.tags.get('algorithm', 'unknown'))}")
        print(f"  Feature Type: {run.data.params.get('feature_type', 'unknown')}")
        print(f"  Silhouette: {run.data.metrics.get('silhouette_score', 0):.4f}")
        print(f"  Noise %: {run.data.metrics.get('enhanced_noise_percentage', run.data.metrics.get('noise_percentage', 0)):.2f}")
        print(f"  Clusters: {run.data.metrics.get('enhanced_num_clusters', run.data.metrics.get('n_clusters_found', 'N/A'))}")
        print(f"  Cluster Size Std: {run.data.metrics.get('enhanced_cluster_size_std', 0):.2f}")
        print(f"  Archetype Entropy: {run.data.metrics.get('enhanced_archetype_entropy_mean', 0):.3f}")

else:
    print("No runs found!")
