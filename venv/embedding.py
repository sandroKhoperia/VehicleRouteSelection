import networkx as nx
from node2vec import Node2Vec
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd
import numpy as np


def generate_node_embeddings(G, dimensions=128, walk_length=6, num_walks=400, p=0.5, q=0.5, workers=1):
    node_features = pd.DataFrame(data={
        'dest_node': [],
        'dest_profit': [],
        'weight': []
    })
    le = LabelEncoder()
    sc = MinMaxScaler()
    for node in G.nodes():
        if G.nodes[node]['type'] == 'car':
            new_data = {'dest_node': G.nodes[node]['dest_node'],
                        'dest_profit': G.nodes[node]['dest_profit'],
                        'weight': G.nodes[node]['weight']}
            node_features = pd.concat([node_features, pd.DataFrame([new_data])], ignore_index=True)

    print(len(node_features))
    # encode the 'dest_node' column
    node_features['dest_node'] = le.fit_transform(node_features['dest_node'])

    # scale the other columns
    node_features[['dest_profit', 'weight']] = sc.fit_transform(node_features[['dest_profit', 'weight']])

    node2vec_model = Node2Vec(G, dimensions=dimensions, walk_length=walk_length,
                              num_walks=num_walks, p=p, q=q, workers=workers)



    model = node2vec_model.fit()

    node_embeddings = {}
    for node in G.nodes():
        if G.nodes[node]['type'] == 'car':
            node_embeddings[node] = np.concatenate((model.wv[str(node)],
                                                    np.array([node_features.iloc[node]['dest_node'],
                                                              node_features.iloc[node]['dest_profit'],
                                                              node_features.iloc[node]['weight']])))
        else:
            node_embeddings[node] = model.wv[str(node)]

    return node_embeddings

def generate_path_embeddings(path, node_embeddings):
    embeddings = []
    for node in path:
        embeddings.append(node_embeddings[node])
    path_embedding = np.mean(embeddings, axis=0)
    return path_embedding
