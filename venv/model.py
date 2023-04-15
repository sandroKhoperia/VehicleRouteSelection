import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, MessagePassing
import preprocess
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import GCNConv
import torch.optim as optim


class PathPredictionModel(torch.nn.Module):
    def __init__(self, num_hidden):
        super(PathPredictionModel, self).__init__()
        self.conv_truck = GCNConv(1, num_hidden)
        self.conv_car = GCNConv(3, num_hidden)
        self.lin = torch.nn.Linear(num_hidden, 1)

    def forward(self, data, src, dst):
        x_dict = {'truck': None, 'car': data['car'].x}
        edge_index_dict = data.edge_index_dict
        edge_weight_dict = data.edge_attr_dict

        # message passing for truck nodes
        x_dict['truck'] = torch.ones((data['truck'].num_nodes, 1)).to(data['truck'].x.device)
        x_dict['truck'] = self.conv_truck(x_dict['truck'], edge_index_dict['truck'])

        # message passing for car nodes
        x_dict['car'] = self.conv_car(x_dict['car'], edge_index_dict['car'], edge_weight_dict['car'])

        # concatenate truck and car node features
        x = torch.cat([x_dict['truck'], x_dict['car']], dim=0)

        # predict path
        src_emb = x[src].squeeze()
        dst_emb = x[dst].squeeze()
        path_score = self.lin(src_emb * dst_emb)
        return path_score

    def predict_path(self, data, src, dst):
        # perform graph search to find path with highest probability
        x_dict = {'truck': None, 'car': data['car'].x}
        edge_index_dict = data.edge_index_dict
        edge_weight_dict = data.edge_attr_dict

        # message passing for truck nodes
        x_dict['truck'] = torch.ones((data['truck'].num_nodes, 1)).to(data['truck'].x.device)
        x_dict['truck'] = self.conv_truck(x_dict['truck'], edge_index_dict['truck'])

        # message passing for car nodes
        x_dict['car'] = self.conv_car(x_dict['car'], edge_index_dict['car'], edge_weight_dict['car'])

        # concatenate truck and car node features
        x = torch.cat([x_dict['truck'], x_dict['car']], dim=0)

        # perform graph search to find path with highest probability
        path_prob, path = self._graph_search(x, edge_index_dict, src, dst)
        return path

    def _graph_search(self, x, edge_index_dict, src, dst):
        visited = set()
        frontier = [(src, [src], 1.0)]
        while frontier:
            node, path, path_prob = frontier.pop(0)
            if node == dst:
                return path_prob, path
            visited.add(node)
            neighbors = edge_index_dict['car'][1, edge_index_dict['car'][0] == node].tolist()
            for neighbor in neighbors:
                if neighbor not in visited:
                    edge_index = torch.tensor([[node, neighbor], [neighbor, node]], dtype=torch.long).to(x.device)
                    neighbor_emb = x[neighbor].squeeze()
                    edge_weight = self.lin(x[node] * neighbor_emb).item()
                    new_path_prob = path_prob * edge_weight
                    new_path = path + [neighbor]
                    frontier.append((neighbor, new_path, new_path_prob))
        return 0.0, []  # no path found


def main():

    all_paths, best_paths, graph = preprocess.preprocess(10, 3, 0.15)
    shortest_paths = []

    for best_path in best_paths:
        path = best_path[0]
        src = path[0]
        dst = path[6]
        shortest_paths.append([src, dst, path])

    # define model and optimizer
    model = PathPredictionModel(num_hidden=16)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # define loss function
    mse_loss = torch.nn.MSELoss()

    # train model
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()

        # forward pass
        path_score = model(graph, src, dst)
        actual_path_prob, actual_path = model._graph_search(x, edge_index_dict, src, dst)
        actual_path = torch.tensor(actual_path, dtype=torch.long).to(path_score.device)

        # compute loss
        loss = mse_loss(path_score, actual_path_prob)

        # backward pass and optimization
        loss.backward()
        optimizer.step()

        # print progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, loss: {loss.item()}")

    # test model
    model.eval()
    path_score = model(data, src, dst)
    actual_path_prob, actual_path = model._graph_search(x, edge_index_dict, src, dst)
    print(f"Predicted path score: {path_score.item()}")
    print(f"Actual path: {actual_path}")

if __name__ == '__main__':
    main()