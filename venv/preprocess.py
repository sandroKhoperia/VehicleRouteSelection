import pandas as pd
import numpy as np
from tqdm import tqdm
from new_hav import compute_truck_and_cars
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
import graphviz
from collections import deque
from itertools import permutations
from embedding import generate_node_embeddings, generate_path_embeddings
from torch_geometric.utils import from_networkx
from torch_geometric.data import HeteroData
import torch

def get_paths(graph, source, dest, length=3, path=[]):
    path = path + [source]
    if len(path) == length + 1:
        if path[-1] == dest:
            return [path]
        else:
            return []
    paths = []
    for neighbor in graph.neighbors(source):
        if neighbor not in path and graph.nodes[neighbor]['type'] == 'car':
            new_paths = get_paths(graph, neighbor, dest, length, path)
            paths.extend(new_paths)
    return paths


def create_graph(trucks, cars, distance_matrix, mapping):
    # Create empty graph
    G = nx.DiGraph()
    data = HeteroData()
    data['car'].x = torch.tensor(cars[['type','dest_node_id','dest_profit', 'car_weight']].values)
    data['truck'].x = torch.tensor(trucks[['type']].values, dtype=torch.float)

    # Add car nodes with attributes
    for _, row in cars.iterrows():
        G.add_node(int(row['id']), type='car', dest_node=mapping[int(row['dest_node_id'])], dest_profit=row['dest_profit'],
                   weight=int(row['car_weight']))
        #G.add_edge(int(row['id']),int(row['dest_node']), weight=distance_matrix[int(row['id'])][int(row['dest_node'])])

    # Add truck nodes with attributes
    for _, row in trucks.iterrows():
        G.add_node(int(row['id']), type='truck', node_id=mapping[int(row['node_id'])], visit_cost=row['visit_cost'])
        G.add_edge(int(row['id']), int(row['node_id']), weight=row['visit_cost'])

    id_to_idx = {car_id: idx for idx, car_id in enumerate(cars['id'])}

    edges = []
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix)):
            if distance_matrix[i][j] != 0:
                if G.nodes[j]['type'] == 'car':
                    G.add_edge(i, j, weight=distance_matrix[i][j])
                edges.append((id_to_idx[i], id_to_idx[j]))
    data['truck'].edge_index =torch.tensor([trucks['id'].tolist(), trucks['node_id'].tolist()], dtype=torch.int)
    data['car'].edge_index = torch.tensor(edges, dtype=torch.int).t()

    #print(data['car'].edge_index_dict)
    print(len(G.nodes()))
    print(len(G.edges()))
    return G, data


def ensure_unique_nodes(map, nodes):
    return frozenset(nodes) in map

def get_all_paths(G, src, dst, mapping):
    paths = get_paths(G, src, dst)
    total_paths = []
    unique_node_map = set()
    first_path_dests = [mapping[G.nodes[paths[0][1]]['dest_node']], mapping[G.nodes[paths[0][2]]['dest_node']],
                mapping[G.nodes[paths[0][3]]['dest_node']]]
    best_path = paths[0] + first_path_dests
    best_path_profit = -10000
    best_path_cars = [paths[0][1:]]
    best_path_dest = first_path_dests
    for path in paths:
        with_no_src = path[1:]
        dest = [mapping[G.nodes[with_no_src[0]]['dest_node']], mapping[G.nodes[with_no_src[1]]['dest_node']],
                mapping[G.nodes[with_no_src[2]]['dest_node']]]

        if ensure_unique_nodes(unique_node_map, with_no_src):
           continue

        unique_node_map.add(frozenset(with_no_src))
        all_permutations = list(permutations(with_no_src + dest))
        weight_map = {
            with_no_src[0]: dest[0],
            with_no_src[1]: dest[1],
            with_no_src[2]: dest[2],
            dest[0]: with_no_src[0],
            dest[1]: with_no_src[1],
            dest[2]: with_no_src[2]
        }
        best_permutation = [path[0]] + list(all_permutations[0])
        best_profit = -5000
        for perm in all_permutations:
            if (perm.index(dest[0]) > perm.index(with_no_src[0])) and \
                    (perm.index(dest[1]) > perm.index(with_no_src[1])) and (
                    perm.index(dest[2]) > perm.index(with_no_src[2]) and list(perm)[4] == dst):
                total_cost = G.nodes[path[0]]['visit_cost']
                total_profit = G.nodes[with_no_src[0]]['dest_profit'] + G.nodes[with_no_src[1]]['dest_profit']
                total_profit += G.nodes[with_no_src[2]]['dest_profit']

                total_weight = G.nodes[dest[0]]['weight']
                total_weight += G.nodes[dest[1]]['weight'] + G.nodes[dest[1]]['weight']
                violation = False  # if total_weight < 13000 else True
                curr_weight = 0;

                for i in range(1, len(perm)):
                    if perm[i - 1] in with_no_src:
                        curr_weight += G.nodes[perm[i - 1]]['weight']
                    else:
                        curr_weight -= G.nodes[weight_map[perm[i - 1]]]['weight']
                    if curr_weight > 13000:
                        violation = True
                    total_cost = total_cost + G.get_edge_data(perm[i - 1], perm[i])['weight'] if perm[i - 1] != perm[
                        i] else 0
                score = total_profit - total_cost if not violation else -1000

                total_paths.append([[path[0]] + list(perm), score])
                if best_profit < score:
                    best_profit = score
                    best_permutation = [path[0]] + list(perm)
        if best_path_profit < best_profit:
            best_path = best_permutation
            best_path_profit = best_profit
            best_path_dest = dest
            best_path_cars =with_no_src
    return total_paths, [best_path, best_path_profit], [best_path_cars, best_path_dest]


def get_every_path(G, mapping):
    trucks = []
    cars = []
    for node in G.nodes:
        if G.nodes[node]['type'] == 'truck':
            trucks.append(node)
        else:
            cars.append(node)

    all_paths = []
    best_paths = []

    for truck in trucks:
        for car in tqdm(cars, desc=f'Processing paths between truck:{truck} and all cars'):
            paths, [best_path, best_profit], [best_path_src, best_path_dst] = get_all_paths(G, truck, car, mapping)
            all_paths.append(paths)
            best_paths.append([best_path, best_profit])


    return all_paths, best_paths



def preprocess(n_cars, n_trucks, gas_price):
    trucks, cars, dist_matrix, mapping = compute_truck_and_cars(n_cars,n_trucks, gas_price)
    G, data = create_graph(trucks, cars, dist_matrix, mapping)
    #embeddings = generate_node_embeddings(G)

    #best_path,best_profit = find_best_path(G,34,5)
    #print(best_path, best_profit)
    #paths, [best_path, best_profit], [best_path_src, best_path_dst] = get_all_paths(G, 34, 5, mapping)
    #all_path_embeddings = []

    #for path in tqdm(paths, desc="Generating Path Embeddings"):
     #   path_embeddings = generate_path_embeddings(path[0][1:], embeddings)
      #  all_path_embeddings.append(path_embeddings)

    all_paths, best_paths = get_every_path(G, mapping)
    print(all_paths[0][:5])
    print(best_paths[:6])
    print(data)
    return all_paths, best_paths, data



def main():
    preprocess(10,3, 0.15)

if __name__ == '__main__':
    main()