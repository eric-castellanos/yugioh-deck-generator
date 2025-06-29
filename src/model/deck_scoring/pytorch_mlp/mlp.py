import torch
from torch.nn import Linear, Sequential, ReLU, MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
import mlflow

from .Dataset import YugiohDeckDataset  # Make sure this file imports DeckPreprocessor
from src.preprocessing.deck_preprocessor.deck_preprocessor import DeckPreprocessor
from src.utils.mlflow.mlflow_utils import setup_experiment, log_params, log_metrics, log_tags, log_pytorch_model


class DeckScoringMLP(torch.nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.layers = Sequential(
            Linear(num_features, 100),
            ReLU(),
            Linear(100, 50),
            ReLU(),
            Linear(50, 25),
            ReLU(),
            Linear(25,1) # output layer
        )

    def forward(self, X):
        return self.layers(X)
    
def set_layer_config(model : DeckScoringMLP):

    layer_config = []

    for layer in model.layers:
        if isinstance(layer, Linear):
            layer_config.append({
                "type": "Linear",
                "in_features": layer.in_features,
                "out_features": layer.out_features
            })
        else:
            layer_config.append({"type": layer.__class__.__name__})

    return layer_config

def train_model(
    model: DeckScoringMLP,
    train_loader,
    val_loader,
    dataset,
    experiment_name="deck_scoring_model",
    num_epochs=1000,
    lr=1e-3,
    patience=20,
    min_delta=1e-4,
    device="cpu",
    register_model: bool = True,
    registered_model_name: str = "yugioh_deck_scoring_mlp"
):
    experiment_id = setup_experiment(experiment_name)
    layer_config = set_layer_config(model)

    with mlflow.start_run(experiment_id=experiment_id):
        log_tags({
            "model_type": "regression",
            "framework": "pytorch",
            "purpose": "deck_scoring"
        })

        log_params({
            "preprocessing_method": dataset.method.upper(),
            "embedding_components": dataset.preprocessor.embedding_n_components,
            "dense_features_count": len(dataset.preprocessor.dense_features),
            "total_features": dataset.X.shape[1],
            "num_epochs": num_epochs,
            "learning_rate": lr,
            "early_stopping_patience": patience,
            "early_stopping_min_delta": min_delta,
            "layer_structure" : layer_config
        })

        model.to(device)
        optimizer = Adam(model.parameters(), lr=lr)
        loss_fn = MSELoss()

        best_loss = float('inf')
        epochs_no_improve = 0
        best_state_dict = None

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().unsqueeze(1).to(device)

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = loss_fn(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.float().to(device)
                    batch_y = batch_y.float().unsqueeze(1).to(device)

                    outputs = model(batch_x)
                    loss = loss_fn(outputs, batch_y)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss
            }, step=epoch)

            if epoch % 100 == 0:
                print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            if val_loss + min_delta < best_loss:
                best_loss = val_loss
                best_state_dict = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

        if best_state_dict:
            model.load_state_dict(best_state_dict)

        # Log final model with mlflow utils
        input_example = torch.randn(1, train_loader.dataset[0][0].shape[0]).to(device)
        log_pytorch_model(
            model=model,
            artifact_path="model",
            model_name="DeckScoringMLP",
            input_example=input_example,
            register=register_model,
            registered_model_name=registered_model_name
        )

if __name__ == "__main__":
    dataset = YugiohDeckDataset(use_pca=True)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model = DeckScoringMLP(num_features=dataset.X.shape[1])
    train_model(model, train_loader, val_loader, dataset=dataset)

