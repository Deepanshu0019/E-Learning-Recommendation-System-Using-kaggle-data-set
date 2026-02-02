import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

# Define the custom dataset class for your user-item interactions
class InteractionsDataset(Dataset):
    def __init__(self, user_ids, item_ids, ratings):
        self.user_ids = torch.tensor(user_ids, dtype=torch.long)
        self.item_ids = torch.tensor(item_ids, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.ratings[idx]

# Implement a simple Neural Collaborative Filtering model
class NeuralCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=50):
        super(NeuralCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, user_id, item_id):
        user_embedded = self.user_embedding(user_id)
        item_embedded = self.item_embedding(item_id)
        x = torch.cat([user_embedded, item_embedded], dim=1)
        return self.fc_layers(x)

# Example of a training function
def train_model(model, dataloader, epochs=5, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        for user_ids, item_ids, ratings in dataloader:
            optimizer.zero_grad()
            predictions = model(user_ids, item_ids).squeeze()
            loss = criterion(predictions, ratings)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")