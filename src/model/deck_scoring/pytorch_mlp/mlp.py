import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Sampler
import numpy as np
import mlflow
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from typing import List, Optional, Tuple
import math

from .Dataset import YugiohDeckDataset
from src.preprocessing.deck_preprocessor.deck_preprocessor import DeckPreprocessor
from src.utils.mlflow.mlflow_utils import setup_experiment, log_params, log_metrics, log_pytorch_model

# =============================================================================
# CONFIGURATION - Simple tweaks to recover Spearman
# =============================================================================
# Keep changes minimal and easy to toggle.

# Model Architecture and Width Presets
MLP_WIDTHS = {
    "128-128-64": [128, 128, 64],    # default (back to original)
    "192-128-64": [192, 128, 64],    # wider first layer
}

# Simple settings that often boost rank stability
NORM_TYPE = "batch"                   # back to BatchNorm for a crisper signal
MODEL_ARCHITECTURE = "128-128-128"     # step back from wider net

# RankNet parameters (slightly sharper + a bit more weight)
RANKNET_WEIGHT = 0.30                  # give ranking a bit more say (try 0.25 next)
RANKNET_TAU = 1.25                     # sharper than 2.0; try 0.5 next if needed
RANKNET_WEIGHT_GRID = None
RANKNET_TAU_GRID = None

# Experimental options
USE_QUANTILE_SAMPLER = False           # stick to standard shuffling for stability
HARD_PAIRS_ONLY = False                # OFF for now; can prune useful pairs otherwise
HARD_GAP_RANGE = (0.03, 0.25)
WEIGHT_RANKNET_BY_BATTLE_WEIGHTS = False # True
USE_ADAPTIVE_WEIGHTS = True           # OFF: can overweight extremes

# Training parameters
NUM_EPOCHS = 450 #350
PATIENCE = 35
LEARNING_RATE = 1e-2
DROPOUT_RATE = 0.20
WEIGHT_DECAY = 1e-4
LOG_EVERY = 5

# Legacy architectures for backward compatibility
LEGACY_ARCHITECTURES = {
    "small": [64, 64],
    "medium": [128, 128, 128],
    "large": [256, 256, 256],
    "deep": [128, 128, 128, 64],
    "wide": [512, 256],
}


def create_model(num_features: int, architecture: str = MODEL_ARCHITECTURE, dropout_rate: float = DROPOUT_RATE):
    """Create a DeckScoringMLP with specified architecture"""
    if architecture in MLP_WIDTHS:
        hidden_layers = MLP_WIDTHS[architecture]
    elif architecture in LEGACY_ARCHITECTURES:
        hidden_layers = LEGACY_ARCHITECTURES[architecture]
    else:
        hidden_layers = MLP_WIDTHS["128-128-64"]
    return DeckScoringMLP(num_features, hidden_layers, dropout_rate, norm_type=NORM_TYPE)


def ranknet_weight_schedule(epoch: int, max_epochs: int) -> float:
    """RankNet weight scheduling: 0.2 → cosine down → 0.1 (kept for grid mode)"""
    progress = epoch / max_epochs
    if progress <= 0.4:
        return 0.2
    elif progress <= 0.8:
        cosine_progress = (progress - 0.4) / 0.4
        return 0.2 - 0.1 * (1 + math.cos(math.pi * cosine_progress)) / 2
    else:
        return 0.1


class QuantileMixSampler(Sampler):
    """Samples batches mixing low/mid/high quantiles of target values"""
    
    def __init__(self, targets: np.ndarray, batch_size: int = 32, min_distinct_targets: int = 5):
        self.targets = np.array(targets)
        self.batch_size = batch_size
        self.min_distinct_targets = min_distinct_targets
        self.indices = np.arange(len(targets))
        self._compute_quantile_buckets()
    
    def _compute_quantile_buckets(self):
        self.q33 = np.percentile(self.targets, 33.33)
        self.q66 = np.percentile(self.targets, 66.67)
        
        self.low_bucket = self.indices[self.targets <= self.q33]
        self.mid_bucket = self.indices[(self.targets > self.q33) & (self.targets <= self.q66)]
        self.high_bucket = self.indices[self.targets > self.q66]
    
    def __iter__(self):
        np.random.shuffle(self.low_bucket)
        np.random.shuffle(self.mid_bucket)
        np.random.shuffle(self.high_bucket)
        
        n_batches = len(self.targets) // self.batch_size
        for _ in range(n_batches):
            batch = self._sample_mixed_batch()
            if len(np.unique(self.targets[batch])) >= self.min_distinct_targets:
                yield batch
            else:
                yield self._sample_mixed_batch()
    
    def _sample_mixed_batch(self) -> List[int]:
        target_per_bucket = self.batch_size // 3
        remainder = self.batch_size % 3
        
        low_sample = np.random.choice(self.low_bucket, min(target_per_bucket, len(self.low_bucket)), replace=False)
        mid_sample = np.random.choice(self.mid_bucket, min(target_per_bucket, len(self.mid_bucket)), replace=False)
        high_sample = np.random.choice(self.high_bucket, min(target_per_bucket + remainder, len(self.high_bucket)), replace=False)
        
        batch = np.concatenate([low_sample, mid_sample, high_sample])
        
        if len(batch) < self.batch_size:
            all_remaining = np.setdiff1d(self.indices, batch)
            fill_needed = self.batch_size - len(batch)
            if len(all_remaining) > 0:
                fill_sample = np.random.choice(all_remaining, min(fill_needed, len(all_remaining)), replace=False)
                batch = np.concatenate([batch, fill_sample])
        
        return batch[:self.batch_size].tolist()
    
    def __len__(self):
        return len(self.targets) // self.batch_size


def calibrate_predictions(val_preds: np.ndarray, val_targets: np.ndarray, test_preds: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> Tuple[np.ndarray, IsotonicRegression]:
    """Post-hoc isotonic calibration on validation predictions (optionally weighted)"""
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(val_preds, val_targets, sample_weight=sample_weight)
    calibrated_preds = calibrator.predict(test_preds)
    return calibrated_preds, calibrator


class DeckScoringMLP(nn.Module):
    def __init__(self, num_features: int, hidden_layers: List[int] = None, dropout_rate: float = 0.2, norm_type: str = "batch"):
        super().__init__()
        
        if hidden_layers is None:
            hidden_layers = [128, 128, 128]
        
        layers = []
        input_size = num_features
        
        for i, hidden_size in enumerate(hidden_layers):
            layers.append(nn.Linear(input_size, hidden_size))
            if norm_type == "layer":
                layers.append(nn.LayerNorm(hidden_size))
            else:
                layers.append(nn.BatchNorm1d(hidden_size, momentum=0.15))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout_rate if i < len(hidden_layers) - 1 else dropout_rate * 0.5))
            input_size = hidden_size
        
        layers.append(nn.Linear(input_size, 1))  # Raw scores, no sigmoid
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def ranknet_loss(scores: torch.Tensor,
                 y: torch.Tensor,
                 tau: float = 1.0,
                 hard_pairs_only: bool = False,
                 hard_gap_range: Tuple[float, float] = (0.02, 0.3),
                 sample_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """RankNet loss using only i<j pairs, optional hard-pair filtering and pairwise weights."""
    s = scores.squeeze()
    t = y.squeeze()

    D = (s.unsqueeze(1) - s.unsqueeze(0)) / tau            # [B,B]
    P = (t.unsqueeze(1) > t.unsqueeze(0)).float()           # [B,B]

    B = t.shape[0]
    triu = torch.triu(torch.ones(B, B, device=t.device, dtype=torch.bool), diagonal=1)

    if hard_pairs_only:
        target_diffs = torch.abs(t.unsqueeze(1) - t.unsqueeze(0))
        hard_mask = (target_diffs >= hard_gap_range[0]) & (target_diffs <= hard_gap_range[1]) & triu
        if hard_mask.any():
            if sample_weights is not None:
                W = ((sample_weights.unsqueeze(1) + sample_weights.unsqueeze(0)) / 2.0)[hard_mask]
                return F.binary_cross_entropy_with_logits(D[hard_mask], P[hard_mask], weight=W)
            return F.binary_cross_entropy_with_logits(D[hard_mask], P[hard_mask])

    mask = triu
    if sample_weights is not None:
        W = ((sample_weights.unsqueeze(1) + sample_weights.unsqueeze(0)) / 2.0)[mask]
        return F.binary_cross_entropy_with_logits(D[mask], P[mask], weight=W)
    return F.binary_cross_entropy_with_logits(D[mask], P[mask])


def compute_adaptive_weights(battle_weights: torch.Tensor,
                             predictions: Optional[torch.Tensor] = None,
                             epoch_fraction: float = 0.0) -> torch.Tensor:
    base_weights = battle_weights
    if predictions is not None:
        pred_probs = torch.sigmoid(predictions.squeeze())
        confidence = 1 - 4 * torch.abs(pred_probs - 0.5)
        adaptive_factor = min(1.0, epoch_fraction * 2)
        confidence_multiplier = 1 + adaptive_factor * confidence
        return base_weights * confidence_multiplier
    return base_weights


def train_model(
    model: DeckScoringMLP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    experiment_name: str = "deck_scoring_model",
    num_epochs: int = NUM_EPOCHS,
    lr: float = LEARNING_RATE,
    patience: int = PATIENCE,
    ranknet_weight: Optional[float] = None,  # None = use schedule
    ranknet_tau: float = RANKNET_TAU,
    device: str = "cpu",
    log_to_mlflow: bool = True
):
    if log_to_mlflow:
        experiment_id = setup_experiment(experiment_name)
        with mlflow.start_run(experiment_id=experiment_id):
            log_params({
                "num_epochs": num_epochs,
                "learning_rate": lr,
                "ranknet_weight": ranknet_weight if ranknet_weight is not None else "scheduled",
                "ranknet_tau": ranknet_tau,
                "patience": patience,
                "norm_type": NORM_TYPE,
                "architecture": MODEL_ARCHITECTURE,
            })
            return _train_model_core(model, train_loader, val_loader, num_epochs, lr, patience,
                                     ranknet_weight, ranknet_tau, device, log_to_mlflow=True)
    else:
        return _train_model_core(model, train_loader, val_loader, num_epochs, lr, patience,
                                 ranknet_weight, ranknet_tau, device, log_to_mlflow=False)


def _train_model_core(model,
                      train_loader,
                      val_loader,
                      num_epochs,
                      lr,
                      patience,
                      ranknet_weight,
                      ranknet_tau,
                      device,
                      log_to_mlflow=False):
    """Core training logic (simple tweaks: BN, MAE point loss, Spearman on logits)."""
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_val_loss = float('inf')
    best_metrics = {}
    epochs_no_improve = 0
    use_schedule = (ranknet_weight is None)

    for epoch in range(num_epochs):
        current_ranknet_weight = ranknet_weight_schedule(epoch, num_epochs) if use_schedule else ranknet_weight
        epoch_frac = epoch / max(1, num_epochs - 1)

        # ------------------------ Train ------------------------
        model.train()
        train_loss_running = 0.0
        for batch_x, batch_y, batch_w in train_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_w = batch_w.float().to(device)
            
            optimizer.zero_grad()
            raw_scores = model(batch_x)
            pred_scores = torch.sigmoid(raw_scores)

            # Pointwise MAE (simple + aligns with metric)
            w_eff = batch_w
            if USE_ADAPTIVE_WEIGHTS and w_eff is not None:
                with torch.no_grad():
                    w_eff = compute_adaptive_weights(w_eff, raw_scores.detach(), epoch_frac)
            point_loss = (torch.abs(pred_scores.squeeze() - batch_y) * w_eff).mean()

            # RankNet
            pair_weights = batch_w if WEIGHT_RANKNET_BY_BATTLE_WEIGHTS else None
            rank_loss = ranknet_loss(raw_scores, batch_y,
                                     tau=ranknet_tau,
                                     hard_pairs_only=HARD_PAIRS_ONLY,
                                     hard_gap_range=HARD_GAP_RANGE,
                                     sample_weights=pair_weights)

            total_loss = (1 - current_ranknet_weight) * point_loss + current_ranknet_weight * rank_loss
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_running += total_loss.item()

        # ---------------------- Validate ----------------------
        model.eval()
        val_point_losses = []
        val_preds, val_targets, val_weights, val_logits_list = [], [], [], []
        with torch.no_grad():
            for batch_x, batch_y, batch_w in val_loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_w = batch_w.float().to(device)
                raw_scores = model(batch_x)
                pred_scores = torch.sigmoid(raw_scores)
                # MAE validation loss for scheduler
                val_point_losses.append((torch.abs(pred_scores.squeeze() - batch_y) * batch_w).mean().item())
                val_preds.extend(pred_scores.cpu().numpy().flatten())
                val_targets.extend(batch_y.cpu().numpy().flatten())
                val_weights.extend(batch_w.cpu().numpy().flatten())
                val_logits_list.append(raw_scores.cpu().numpy().flatten())

        val_loss = float(np.mean(val_point_losses))
        val_preds_np = np.array(val_preds)
        val_targets_np = np.array(val_targets)
        val_weights_np = np.array(val_weights)
        val_logits_np = np.concatenate(val_logits_list)

        val_rmse = float(np.sqrt(np.mean((val_preds_np - val_targets_np) ** 2)))
        val_mae = float(np.mean(np.abs(val_preds_np - val_targets_np)))
        # Spearman on logits (monotonic; avoids sigmoid squashing ties)
        val_spearman = float(spearmanr(val_logits_np, val_targets_np)[0]) if len(val_preds_np) > 1 else 0.0

        if log_to_mlflow and (epoch % LOG_EVERY == 0 or epoch == num_epochs - 1):
            log_metrics({
                "train_loss": train_loss_running / max(1, len(train_loader)),
                "val_loss": val_loss,
                "val_rmse": val_rmse,
                "val_mae": val_mae,
                "val_spearman": val_spearman,
                "ranknet_weight_effective": current_ranknet_weight,
            }, step=epoch)
            print(f"Epoch {epoch}: Val Loss={val_loss:.4f}, RMSE={val_rmse:.4f}, MAE={val_mae:.4f}, Spearman={val_spearman:.4f}, RankNet={current_ranknet_weight:.3f}")

        scheduler.step(val_loss)

        # Early stopping (track best epoch's predictions for calibration)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = {
                "rmse": val_rmse,
                "mae": val_mae,
                "spearman": val_spearman,
                "val_preds": val_preds_np,
                "val_targets": val_targets_np,
                "val_weights": val_weights_np,
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if log_to_mlflow:
                    print(f"Early stopping at epoch {epoch}")
                break

    # ----------------- Post-hoc calibration (best epoch) -----------------
    val_preds_final = best_metrics.get("val_preds")
    val_targets_final = best_metrics.get("val_targets")
    if val_preds_final is not None and len(val_preds_final) > 0:
        val_weights_final = best_metrics.get("val_weights")
        calibrated_preds, calibrator = calibrate_predictions(
            val_preds_final, val_targets_final, val_preds_final, sample_weight=val_weights_final
        )
        val_mae_cal = float(np.mean(np.abs(calibrated_preds - val_targets_final)))
        val_rmse_cal = float(np.sqrt(np.mean((calibrated_preds - val_targets_final) ** 2)))
        best_metrics.update({
            "mae_raw": best_metrics["mae"],
            "rmse_raw": best_metrics["rmse"],
            "mae_cal": val_mae_cal,
            "rmse_cal": val_rmse_cal,
            "calibrator": calibrator,
        })
    else:
        best_metrics.setdefault("mae_raw", best_metrics.get("mae", float('nan')))
        best_metrics.setdefault("rmse_raw", best_metrics.get("rmse", float('nan')))

    if log_to_mlflow:
        input_example = torch.randn(1, train_loader.dataset[0][0].shape[0]).to(device)
        log_pytorch_model(
            model=model,
            artifact_path="model",
            model_name="DeckScoringMLP",
            input_example=input_example
        )

    return best_metrics


def detect_leakage(train_dataset, val_dataset, preprocessor):
    print("🔍 LEAKAGE DETECTION:")
    suspicious_features = []
    for feature in preprocessor.feature_names_:
        if any(keyword in feature.lower() for keyword in ['win_rate', 'target', 'score', 'rank', 'percentile']):
            suspicious_features.append(feature)
    if suspicious_features:
        print(f"⚠️  Suspicious features found: {suspicious_features}")
    else:
        print("✅ No obviously suspicious feature names detected")

    X_train, y_train = train_dataset.X, train_dataset.y
    high_corr_features = []
    for i, feature_name in enumerate(preprocessor.feature_names_):
        if i < X_train.shape[1]:
            corr = np.corrcoef(X_train[:, i], y_train)[0, 1]
            if abs(corr) > 0.9:
                high_corr_features.append((feature_name, corr))
    if high_corr_features:
        print("⚠️  Features with >0.9 correlation to target:")
        for feature, corr in high_corr_features:
            print(f"   {feature}: {corr:.4f}")
    else:
        print("✅ No features with >0.9 correlation to target")

    y_train_mean, y_val_mean = np.mean(y_train), np.mean(val_dataset.y)
    y_train_std, y_val_std = np.std(y_train), np.std(val_dataset.y)
    print("📊 Target distributions:")
    print(f"   Train: mean={y_train_mean:.4f}, std={y_train_std:.4f}")
    print(f"   Val:   mean={y_val_mean:.4f}, std={y_val_std:.4f}")
    if abs(y_train_mean - y_val_mean) < 0.01 and abs(y_train_std - y_val_std) < 0.01:
        print("⚠️  Train/val target distributions are suspiciously similar")
    else:
        print("✅ Train/val target distributions look appropriately different")


def run_grid_search():
    from .Dataset import create_cv_splits_no_leakage

    weight_grid = RANKNET_WEIGHT_GRID if RANKNET_WEIGHT_GRID is not None else [RANKNET_WEIGHT] if (RANKNET_WEIGHT is not None) else [None]
    tau_grid = RANKNET_TAU_GRID if RANKNET_TAU_GRID is not None else [RANKNET_TAU]

    results = []
    for weight in weight_grid:
        for tau in tau_grid:
            print(f"🔬 GRID SEARCH: weight={weight}, tau={tau}")
            splits = create_cv_splits_no_leakage(use_pca=False, n_splits=3)
            fold_metrics = []
            for fold, (train_dataset, val_dataset, preprocessor) in enumerate(splits):
                if USE_QUANTILE_SAMPLER:
                    train_sampler = QuantileMixSampler(train_dataset.y, batch_size=32)
                    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
                else:
                    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=32)

                model = create_model(num_features=train_dataset.X.shape[1],
                                     architecture=MODEL_ARCHITECTURE,
                                     dropout_rate=DROPOUT_RATE)

                metrics = train_model(model=model,
                                      train_loader=train_loader,
                                      val_loader=val_loader,
                                      experiment_name="deck_scoring_model",
                                      num_epochs=NUM_EPOCHS,
                                      patience=PATIENCE,
                                      ranknet_weight=weight,
                                      ranknet_tau=tau,
                                      log_to_mlflow=False)
                fold_metrics.append(metrics)

            avg_metrics = {
                "ranknet_weight": weight if weight is not None else "scheduled",
                "ranknet_tau": tau,
                "val_spearman": float(np.mean([m['spearman'] for m in fold_metrics])),
                "val_mae_raw": float(np.mean([m.get('mae_raw', m['mae']) for m in fold_metrics])),
                "val_rmse_raw": float(np.mean([m.get('rmse_raw', m['rmse']) for m in fold_metrics])),
                "val_mae_cal": float(np.mean([m.get('mae_cal', m.get('mae_raw', m['mae'])) for m in fold_metrics])),
                "val_rmse_cal": float(np.mean([m.get('rmse_cal', m.get('rmse_raw', m['rmse'])) for m in fold_metrics])),
            }
            results.append(avg_metrics)

    results.sort(key=lambda x: (-x['val_spearman'], x['val_mae_cal']))

    print(f"📊 GRID SEARCH RESULTS:")
    print(f"{'Weight':<12} {'Tau':<6} {'Spearman':<9} {'MAE_Raw':<8} {'RMSE_Raw':<9} {'MAE_Cal':<8} {'RMSE_Cal':<9}")
    print("-" * 75)
    for result in results:
        weight_str = str(result['ranknet_weight'])[:10] if result['ranknet_weight'] != "scheduled" else "scheduled"
        print(f"{weight_str:<12} {result['ranknet_tau']:<6.1f} {result['val_spearman']:<9.4f} "
              f"{result['val_mae_raw']:<8.4f} {result['val_rmse_raw']:<9.4f} "
              f"{result['val_mae_cal']:<8.4f} {result['val_rmse_cal']:<9.4f}")

    return results


if __name__ == "__main__":
    if (RANKNET_WEIGHT_GRID is not None) or (RANKNET_TAU_GRID is not None):
        run_grid_search()
    else:
        from .Dataset import create_cv_splits_no_leakage
        print("Running PyTorch MLP Cross-Validation...")
        splits = create_cv_splits_no_leakage(use_pca=False, n_splits=3)

        oof_raw_preds, oof_targets, oof_weights, oof_cal_preds = [], [], [], []
        all_metrics = []
        for fold, (train_dataset, val_dataset, preprocessor) in enumerate(splits):
            print(f"=== FOLD {fold + 1} ===")

            if fold == 0:
                detect_leakage(train_dataset, val_dataset, preprocessor)
                print()

            if USE_QUANTILE_SAMPLER:
                train_sampler = QuantileMixSampler(train_dataset.y, batch_size=32)
                train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
            else:
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32)

            model = create_model(num_features=train_dataset.X.shape[1],
                                 architecture=MODEL_ARCHITECTURE,
                                 dropout_rate=DROPOUT_RATE)

            metrics = train_model(model=model,
                                  train_loader=train_loader,
                                  val_loader=val_loader,
                                  experiment_name="deck_scoring_model",
                                  num_epochs=NUM_EPOCHS,
                                  patience=PATIENCE,
                                  ranknet_weight=RANKNET_WEIGHT,
                                  ranknet_tau=RANKNET_TAU,
                                  log_to_mlflow=True)

            all_metrics.append(metrics)

            # OOF collectors
            oof_raw_preds.append(metrics["val_preds"])  # uncalibrated
            oof_targets.append(metrics["val_targets"])  # targets
            oof_weights.append(metrics.get("val_weights", np.ones_like(metrics["val_targets"])))

            # Per-fold calibration on its own val
            val_preds_f = metrics["val_preds"]
            val_targets_f = metrics["val_targets"]
            val_weights_f = metrics.get("val_weights", np.ones_like(val_targets_f))
            cal_f, _ = calibrate_predictions(val_preds_f, val_targets_f, val_preds_f, sample_weight=val_weights_f)
            oof_cal_preds.append(cal_f)

            print(f"Fold {fold + 1} Results: RMSE={metrics.get('rmse_raw', metrics['rmse']):.4f}, MAE={metrics.get('mae_raw', metrics['mae']):.4f}, Spearman={metrics['spearman']:.4f}")

        avg_rmse = float(np.mean([m.get('rmse_raw', m['rmse']) for m in all_metrics]))
        avg_mae = float(np.mean([m.get('mae_raw', m['mae']) for m in all_metrics]))
        avg_spearman = float(np.mean([m['spearman'] for m in all_metrics]))
        avg_rmse_cal = float(np.mean([m.get('rmse_cal', m.get('rmse_raw', m['rmse'])) for m in all_metrics]))
        avg_mae_cal = float(np.mean([m.get('mae_cal', m.get('mae_raw', m['mae'])) for m in all_metrics]))

        print(f"=== CROSS-VALIDATION SUMMARY ===")
        print(f"Average Spearman: {avg_spearman:.4f}")
        print(f"Average RMSE Raw: {avg_rmse:.4f}, Cal: {avg_rmse_cal:.4f}")
        print(f"Average MAE Raw: {avg_mae:.4f}, Cal: {avg_mae_cal:.4f}")

        if avg_spearman > 0.95:
            print("🚨 WARNING: Spearman correlation > 0.95 suggests possible data leakage!")
        elif avg_spearman > 0.90:
            print("⚠️  CAUTION: Spearman correlation > 0.90 is quite high, check for leakage")

        # OOF evaluation (informational)
        oof_raw = np.concatenate(oof_raw_preds)
        oof_cal = np.concatenate(oof_cal_preds)
        oof_y   = np.concatenate(oof_targets)
        oof_w   = np.concatenate(oof_weights)

        def _mae(a,b): return float(np.mean(np.abs(a-b)))
        def _rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))
        from scipy.stats import spearmanr as _spr

        print("=== OOF EVALUATION (across folds) ===")
        print(f"Raw: MAE={_mae(oof_raw,oof_y):.4f} RMSE={_rmse(oof_raw,oof_y):.4f} Spearman={_spr(oof_raw,oof_y)[0]:.4f}")
        print(f"Cal: MAE={_mae(oof_cal,oof_y):.4f} RMSE={_rmse(oof_cal,oof_y):.4f} Spearman={_spr(oof_cal,oof_y)[0]:.4f}")

        experiment_id = setup_experiment("deck_scoring_model")
        with mlflow.start_run(experiment_id=experiment_id):
            log_params({
                "model_type": "PyTorch_MLP",
                "num_folds": len(all_metrics),
                "num_epochs": NUM_EPOCHS,
                "patience": PATIENCE,
                "ranknet_weight": RANKNET_WEIGHT if RANKNET_WEIGHT is not None else "scheduled",
                "ranknet_tau": RANKNET_TAU,
                "learning_rate": LEARNING_RATE,
                "architecture": MODEL_ARCHITECTURE,
                "hidden_layers": str(MLP_WIDTHS.get(MODEL_ARCHITECTURE, MLP_WIDTHS["128-128-64"])) ,
                "dropout_rate": DROPOUT_RATE,
                "norm_type": NORM_TYPE,
                "weight_decay": WEIGHT_DECAY,
            })
            log_metrics({
                "cv_rmse_mean": avg_rmse,
                "cv_mae_mean": avg_mae,
                "cv_spearman_mean": avg_spearman,
                "cv_rmse_cal_mean": avg_rmse_cal,
                "cv_mae_cal_mean": avg_mae_cal,
                "cv_rmse_std": float(np.std([m.get('rmse_raw', m['rmse']) for m in all_metrics])),
                "cv_mae_std": float(np.std([m.get('mae_raw', m['mae']) for m in all_metrics])),
                "cv_spearman_std": float(np.std([m['spearman'] for m in all_metrics])),
            })
            print("✅ Mean results logged to MLflow experiment: deck_scoring_model")
