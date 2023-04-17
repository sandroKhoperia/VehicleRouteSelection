import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
import preprocess
import torch.nn as nn
from torch.optim import Adam
import random

class SequencePredictionModel(torch.nn.Module):
    def __init__(self, num_features, hidden_size):
        super(SequencePredictionModel, self).__init__()
        self.num_features = num_features
        self.conv1 = SAGEConv((-1,-1), 2*hidden_size)
        self.conv2 = GCNConv(2*hidden_size, 2*hidden_size)
        self.conv3 = GCNConv(2*hidden_size, hidden_size)
        self.fc1 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.out = torch.nn.Linear(hidden_size, 4*self.num_features)

    def forward(self, x, edge_index):
        # Apply graph convolutional layers and linear layer
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.out(x)  # (b, len * dim) -> (b, len, dim)
        logits = logits.view(self.num_features, 4, self.num_features)

        return logits


def main():
    all_paths, best_paths, data = preprocess.preprocess(30, 1, 0.15)
    num_features = 30

    paths, _ = zip(*best_paths)
    paths = list(paths)
    random.shuffle(paths)
    #pathsss = [p[2:6] for p in paths]
    # Create the custom dataset
    #dataset = CustomDataset(pathsss, data)

    # Create the dataloader
    #batch_size = 32
    #loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = SequencePredictionModel(num_features=num_features, hidden_size=256)
    edge_index = data.edge_index.long()
    optimizer = Adam(model.parameters(), lr=0.01)
    epochs = 500
    for epoch in range(1, epochs+1):
        output = model(data.x, edge_index)
        path = [p[2:6] for p in paths[:num_features]]
        path = torch.tensor(path, dtype=torch.long)
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

    pred = model(data.x, edge_index)  # outputs a tensor of shape (100, 4)
    print("Actual:", path)
    print("Predicted:", torch.argmax(pred, dim=-1))
# array([3.  , 4.75, 4.5 , 3.75, 5.5 , 4.25, 4.5 , 6.5 , 6.  , 1.75])
if __name__ =='__main__':
    main()