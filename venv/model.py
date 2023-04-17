import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, MessagePassing, to_hetero, SAGEConv, GAE
import preprocess
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData,DataLoader, Dataset
from torch_geometric.loader import HGTLoader
import torch.optim as optim


class Autoencoder(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super(Autoencoder, self).__init__()
        self.conv1 = GCNConv(input_channels, 2*hidden_channels)
        self.conv2 = GCNConv(2*hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, 6)


    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.fc(x)
        return x



# class GNNDataset(torch_geometric.data.Dataset):
#     def __init__(self):


class GraphWalkDataset(Dataset):
    def __init__(self, walks, data):
        self.walks = walks
        self.data = data
    def __len__(self):
        return len(self.walks)

    def __getitem__(self, idx):
        src, dst, walk = self.walks[idx]

        walk = torch.tensor(walk)

        return src, dst, walk
def main():

    all_paths, best_paths, data = preprocess.preprocess(10, 3, 0.15)
    shortest_paths = []

    for best_path in best_paths:
        path = best_path[0]
        src = path[0]
        dst = path[6]
        shortest_paths.append([src, dst, path])

    dataset = GraphWalkDataset(shortest_paths)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    batch_size = 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define the model and optimizer
    in_channels = {'car': data['car'].x.shape[1], 'truck': data['car'].x.shape[1]}
    hidden_channels = 128
    out_channels = 128
    #model = Autoencoder(hidden_channels, out_channels)
    #metadata = data.metadata()
    #node_types = data.node_types
    # Convert your model to be compatible with HeteroData
    #model = to_hetero(model, data.metadata())
    data.node_types = ['car', 'truck']
    print(data.metadata())
    print(data)
    model = Autoencoder(hidden_channels)
    model = GAE(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    num_epochs = 2
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for i, data in enumerate(train_loader):
            src, dst, walk = data
            optimizer.zero_grad()
            output = model(data.x_dict, data.edge_index, source=src, destination=dst)
            loss = model.recon_loss(output, walk)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                src, dst, walk = data
                output = model(walk, source=src, destination=dst)
                loss = criterion(output, walk)
                test_loss += loss.item()
            test_loss /= len(test_loader)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

if __name__ == '__main__':
    main()