import numpy as np
import networkx as nx
from trainData import Dataset
from param import parameter_parser
from data_process import prepare_data

def construct_hypergraph(adjacency, is_binary=True):
    # data_dir = '../Data/'
    # adjacency = np.load(data_dir + net_name)
    n = adjacency.shape[0]

    alpha = 1
    I = np.eye(n)
    Z = alpha * np.linalg.inv(alpha * np.dot(adjacency.T, adjacency) + I).dot(adjacency.T).dot(adjacency)
    # print(Z)
    # facebook:0.015, Bio-CE-GT:
    row, col = np.where(Z > 0.25)  # 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3
    H = np.zeros((n, n))

    # binary incidence matrix
    if is_binary:
        H[row, col] = 1

    # probalitical incidence matrix
    else:
        H[row, col] = Z[row, col]

    W_temp = np.dot(H.T, Z)
    W_norm = np.linalg.norm(W_temp, axis=1)
    W = np.diag(W_norm / np.sum(H, axis=0))
    # # W = np.eye(n)
    DV = np.diag(np.sum(np.dot(H, W), axis=1))
    #
    hypergraph_adj = np.dot(H, W).dot(H.T) - DV
    #graph = nx.from_numpy_matrix(hypergraph_adj)
    # graph = nx.from_numpy_matrix(adjacency)
    return hypergraph_adj, H + adjacency

opt = parameter_parser()
dataset = prepare_data(opt)

print('begin--')
adj=dataset['md_p']
adj1=adj.numpy()
print(adj1)
hypergraph_adj, H_adjacency=construct_hypergraph(adj,True)
print(hypergraph_adj.shape())
print(H_adjacency.shape())
print(hypergraph_adj)
print(H_adjacency)



