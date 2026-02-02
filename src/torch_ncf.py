from typing import Dict
import numpy as np
import pandas as pd


class TorchNCFRecommender:
	def __init__(self, model, num_items: int, device):
		self.model = model
		self.num_items = num_items
		self.device = device

	def predict_user(self, user_idx: int) -> np.ndarray:
		import torch
		self.model.eval()
		with torch.no_grad():
			user_tensor = torch.full((self.num_items,), user_idx, dtype=torch.long, device=self.device)
			item_tensor = torch.arange(self.num_items, dtype=torch.long, device=self.device)
			pred = self.model(user_tensor, item_tensor)
			return pred.detach().cpu().numpy()


def train_torch_ncf(
	interactions: pd.DataFrame,
	user_index: Dict[str, int],
	item_index: Dict[str, int],
	factors: int = 32,
	epochs: int = 5,
	lr: float = 1e-3,
):
	import torch
	import torch.nn as nn
	import torch.optim as optim

	class NCF(nn.Module):
		def __init__(self, num_users: int, num_items: int, k: int):
			super().__init__()
			self.user_emb = nn.Embedding(num_users, k)
			self.item_emb = nn.Embedding(num_items, k)
			self.fc1 = nn.Linear(k * 2, k)
			self.fc2 = nn.Linear(k, 1)
			self.relu = nn.ReLU()
			self.sigmoid = nn.Sigmoid()

		def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor):
			u = self.user_emb(user_indices)
			i = self.item_emb(item_indices)
			x = torch.cat([u, i], dim=-1)
			h = self.relu(self.fc1(x))
			y = self.fc2(h).squeeze(-1)
			return self.sigmoid(y)  # predict rating in 0..1

	device = torch.device('cpu')
	num_users = len(user_index)
	num_items = len(item_index)
	model = NCF(num_users, num_items, factors).to(device)
	optimizer = optim.Adam(model.parameters(), lr=lr)
	criterion = nn.MSELoss()

	# Prepare tensors
	u_idx = torch.tensor(interactions['user_id'].map(user_index).to_numpy(), dtype=torch.long, device=device)
	i_idx = torch.tensor(interactions['course_id'].map(item_index).to_numpy(), dtype=torch.long, device=device)
	r = torch.tensor((interactions['rating'].astype(float) / 5.0).to_numpy(), dtype=torch.float32, device=device)

	model.train()
	batch_size = 256
	n = u_idx.shape[0]
	for _ in range(epochs):
		perm = torch.randperm(n)
		u_idx = u_idx[perm]
		i_idx = i_idx[perm]
		r = r[perm]
		for start in range(0, n, batch_size):
			end = min(start + batch_size, n)
			optimizer.zero_grad()
			pred = model(u_idx[start:end], i_idx[start:end])
			loss = criterion(pred, r[start:end])
			loss.backward()
			optimizer.step()

	return TorchNCFRecommender(model, num_items, device)


