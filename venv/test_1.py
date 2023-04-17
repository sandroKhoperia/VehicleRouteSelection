import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
import preprocess
import torch.nn as nn
from torch.optim import Adam
import random

from torch_geometric.data import Dataset, DataLoader
from torch_geometric.utils import subgraph


class CustomDataset(Dataset):
    def __init__(self, data, paths):
        self.paths = paths
        self.data = data
        super().__init__()

    def len(self):
        return len(self.paths)

    def get(self, idx):
        node_list = self.paths[idx]

        nodes = torch.tensor(node_list)
        edge_index = subgraph(nodes, self.data.edge_index)[0]

        return {'x': self.data.x[nodes], 'edge_index': edge_index, 'path': nodes}


class SequencePredictionModel(torch.nn.Module):
    def __init__(self, num_features, hidden_size):
        super(SequencePredictionModel, self).__init__()
        self.num_features = num_features
        self.conv1 = SAGEConv((-1, -1), 2 * hidden_size)
        self.conv2 = GCNConv(2 * hidden_size, 2 * hidden_size)
        self.conv3 = GCNConv(2 * hidden_size, hidden_size)
        self.fc1 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.out = torch.nn.Linear(hidden_size, 4 * self.num_features)

    def forward(self, x, edge_index):
        # Apply graph convolutional layers and linear layer
        x = F.relu(self.conv1(x.squeeze(), edge_index.squeeze()))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.out(x)  # (b, len * dim) -> (b, len, dim)
        logits = logits.view(self.num_features, 4, self.num_features)

        return logits


def main():
    num_features = 30

    all_paths, best_paths, data = preprocess.preprocess(num_features, 1, 0.15)

    paths, _ = zip(*best_paths)
    paths = list(paths)
    pathsss = [p[2:6] for p in paths]
    # Create the custom dataset
    dataset = CustomDataset(data, pathsss)

    # Create the dataloader
    batch_size = 4
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = SequencePredictionModel(num_features=num_features, hidden_size=256)
    # edge_index = data['car','visit','car'].edge_index.long()
    optimizer = Adam(model.parameters(), lr=0.01)
    epochs = 500
    first_batch = next(iter(loader))
    for epoch in range(1, epochs + 1):
        for batch in loader:
            edg = batch['edge_index']
            nodes = batch['x']
            output = model(batch['x'], batch['edge_index'])
            # path = [p[2:6] for p in paths[:num_features]]
            path = torch.tensor(batch['path'], dtype=torch.long)
            loss = None
            for i in range(4):
                logits = output[:, i, :]
                y = path[:, i]
                if not loss:
                    loss = F.cross_entropy(logits, y)
                else:
                    loss += F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()

        print(f"Epoch:{epoch}  Training Loss:{loss}")

    first_batch = next(iter(loader))
    pred = model(first_batch['x'], first_batch['edge_index'])  # outputs a tensor of shape (100, 4)
    print(x)
    print("Actual:", first_batch['path'])
    print("Predicted:", torch.argmax(pred, dim=-1))


if __name__ == '__main__':
    main()