import torch
from torch.nn import Linear, Sequential, ReLU, MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from .Dataset import YugiohDeckDataset  # Make sure this file imports DeckPreprocessor
from src.preprocessing.deck_preprocessor.deck_preprocessor import DeckPreprocessor

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

def train_model(model: DeckScoringMLP, dataloader: DataLoader, num_epochs: int = 500, lr: float = 1e-3):
    loss_fn = MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.float()
            batch_y = batch_y.float().unsqueeze(1)  # Make sure shape is (B, 1)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 100 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"[Epoch {epoch}] Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    dataset = YugiohDeckDataset(use_pca=True)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = DeckScoringMLP(num_features=dataset.X.shape[1])
    train_model(model, dataloader, num_epochs=500, lr=1e-3)

