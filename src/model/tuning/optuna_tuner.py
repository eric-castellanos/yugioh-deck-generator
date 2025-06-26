from typing import Tuple, Dict, Any
import os

import matplotlib.pyplot as plt
import optuna
import xgboost as xgb
from xgboost.callback import EarlyStopping
import numpy as np
from sklearn.metrics import root_mean_squared_error
import mlflow

from src.utils.mlflow.mlflow_utils import log_params, log_metrics, log_ml_model, log_artifact

def tune_xgboost_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50
) -> Tuple[xgb.XGBRegressor, Dict[str, Any]]:
    """
    Tune XGBoost hyperparameters using Optuna and return the best model and parameters.
    """

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "n_estimators": trial.suggest_int("n_estimators", 500, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "random_state": 42
        }

        callbacks=[EarlyStopping(rounds=20)]

        model = xgb.XGBRegressor(**params, callbacks=callbacks)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        preds = model.predict(X_val)
        rmse = root_mean_squared_error(y_val, preds)
        return rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_params["objective"] = "reg:squarederror"
    best_params["tree_method"] = "hist"
    best_params["random_state"] = 42

    # Train final model with best params
    best_model = xgb.XGBRegressor(**best_params)
    best_model.fit(X_train, y_train)

    # Evaluate on validation set
    preds = best_model.predict(X_val)
    rmse = root_mean_squared_error(y_val, preds)

    # Log best trial
    log_params(best_params)
    log_metrics({"val_rmse": rmse})
    log_ml_model(best_model, artifact_path="xgboost_tuned", framework="xgboost")

    plot_path = plot_optuna_loss_curve(study)
    log_artifact(plot_path, artifact_path="visualizations")

    return best_model, best_params

def plot_optuna_loss_curve(study: optuna.Study, output_path: str = "artifacts/optuna_loss_curve.png") -> str:
    """
    Plots the Optuna trial loss (RMSE) as it evolves over time and saves the figure.
    
    Args:
        study (optuna.Study): The completed Optuna study object.
        output_path (str): Path to save the loss curve plot.

    Returns:
        str: The path to the saved plot image.
    """
    trials = study.trials_dataframe()
    trials = trials[trials["state"] == "COMPLETE"]
    trials = trials.sort_values("number")

    plt.figure(figsize=(10, 6))
    plt.plot(trials["number"], trials["value"], marker="o", label="Trial RMSE")
    plt.xlabel("Trial Number")
    plt.ylabel("Validation RMSE")
    plt.title("Optuna Hyperparameter Tuning Loss Curve")
    plt.grid(True)
    plt.legend()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    return output_path