from . import K_means_h
from . import KNN_h
import math
import numpy as np



def constructHW_knn(X,K_neigs,is_probH):

    """incidence matrix"""
    H = KNN_h.construct_H_with_KNN(X, K_neigs, is_probH)#超图的邻接矩阵

    G = KNN_h._generate_G_from_H(H)#这个函数通过处理超图的邻接矩阵和超边权重，生成了一个图的拉普拉斯矩阵，从超图邻接矩阵 H 生成图 G 的邻接矩阵的过程，这种处理在图卷积网络中用于平衡不同节点和边的贡献，提高模型的性能。

    return G

def constructHW_kmean(X,clusters):

    """incidence matrix"""
    H = K_means_h.construct_H_with_Kmeans(X,clusters)#基于K

    G = K_means_h._generate_G_from_H(H)

    return G
