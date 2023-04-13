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





def get_paths(graph, source, dest, length=3, path=[]):
    path = path + [source]
    if len(path) == length + 1:
        return [path]
    paths = []
    for neighbor in graph.neighbors(source):
        if neighbor not in path and neighbor != dest and graph.nodes[neighbor]['type']=='car':
            new_paths = get_paths(graph, neighbor, dest, length, path)
            paths.extend(new_paths)
        elif neighbor == dest and len(path) == length and graph.nodes[neighbor]['type']=='car':
            paths.append(path + [dest])
    return paths



def create_graph(trucks, cars, distance_matrix, mapping):
    # Create empty graph
    G = nx.Graph()

    # Add car nodes with attributes
    for _, row in cars.iterrows():
        G.add_node(int(row['id']), type='car', dest_node=mapping[int(row['dest_node_id'])], dest_profit=row['dest_profit'],
                   weight=int(row['car_weight']))
        #G.add_edge(int(row['id']),int(row['dest_node']), weight=distance_matrix[int(row['id'])][int(row['dest_node'])])

    # Add truck nodes with attributes
    for _, row in trucks.iterrows():
        G.add_node(int(row['id']), type='truck', node_id=mapping[int(row['node_id'])], visit_cost=row['visit_cost'])
        G.add_edge(int(row['id']), int(row['node_id']), weight=row['visit_cost'])


    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix)):
            if distance_matrix[i][j] != 0:
                if G.nodes[j]['type'] == 'car':
                    G.add_edge(i, j, weight=distance_matrix[i][j])

    print(G.nodes[0])

    return G




def get_all_paths(G, src, dst, mapping):
    paths = get_paths(G, src, dst)
    total_paths = []
    for path in tqdm(paths, desc='Computing path permutations'):
        with_no_src = path[1:]
        dest = [mapping[G.nodes[with_no_src[0]]['dest_node']], mapping[G.nodes[with_no_src[1]]['dest_node']],
                mapping[G.nodes[with_no_src[2]]['dest_node']]]
        all_permutations = permutations(with_no_src + dest)
        weight_map = {
            with_no_src[0]: dest[0],
            with_no_src[1]: dest[1],
            with_no_src[2]: dest[2],
            dest[0]: with_no_src[0],
            dest[1]: with_no_src[1],
            dest[2]: with_no_src[2]
        }
        for perm in all_permutations:
            if (perm.index(dest[0]) > perm.index(with_no_src[0])) and \
                    (perm.index(dest[1]) > perm.index(with_no_src[1])) and (
                    perm.index(dest[2]) > perm.index(with_no_src[2])):
                total_cost = G.nodes[path[0]]['visit_cost']
                total_profit = G.nodes[with_no_src[0]]['dest_profit'] + G.nodes[with_no_src[1]]['dest_profit']
                total_profit += G.nodes[with_no_src[2]]['dest_profit']

                total_weight = G.nodes[dest[0]]['weight']
                total_weight += G.nodes[dest[1]]['weight'] + G.nodes[dest[1]]['weight']
                violation = False  # if total_weight < 13000 else True
                curr_weight = 0;
                # cars - 0,1,2 (5000,6000,7000)  dst - 4,5,6
                # perm - 0,1,4,2,5,6

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

    return total_paths


def preprocess(n, gas_price):
    trucks, cars, dist_matrix, mapping = compute_truck_and_cars(30,5, gas_price)
    G = create_graph(trucks, cars, dist_matrix, mapping)
    embeddings = generate_node_embeddings(G)

    #best_path,best_profit = find_best_path(G,34,5)
    #print(best_path, best_profit)
    paths = get_all_paths(G, 34, 5, mapping)
    all_path_embeddings = []

    for path in tqdm(paths, desc="Generating Path Embeddings"):
        path_embeddings = generate_path_embeddings(path[0][1:], embeddings)
        all_path_embeddings.append(path_embeddings)

    print(all_path_embeddings[0])







def main():
    preprocess(30, 0.15)

if __name__ == '__main__':
    main()