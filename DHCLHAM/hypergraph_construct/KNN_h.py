import numpy as np
import torch
import math


# 用于计算输入数据点之间的欧几里得距离矩阵。
def Eu_dis(x):
    x = np.mat(x)

    aa = np.sum(np.multiply(x, x), 1)
    ab = x * x.T
    dist_mat = aa + aa.T - 2 * ab
    dist_mat[dist_mat < 0] = 0

    dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)
    return dist_mat


def log_cosh_dis(x):  # 双曲余弦
    # 将输入转换为 NumPy 数组
    x = np.array(x)
    n = x.shape[0]  # 获取样本数量
    dist_mat = np.zeros((n, n))  # 初始化距离矩阵

    for i in range(n):
        for j in range(n):
            diff = x[i] - x[j]  # 计算两个向量的差值
            cosh_values = np.cosh(diff)  # 计算 cosh 值
            log_cosh_values = np.log(cosh_values)  # 计算 log(cosh) 值
            dist_mat[i, j] = np.sum(log_cosh_values)  # 求和得到距离

    return dist_mat


def cosine_distance(x):  # 夹角余弦
    # 将输入转换为 NumPy 数组
    x = np.array(x)
    n = x.shape[0]  # 获取样本数量
    dist_mat = np.zeros((n, n))  # 初始化距离矩阵

    # 计算向量的模长
    norms = np.linalg.norm(x, axis=1)

    for i in range(n):
        for j in range(n):
            dot_product = np.dot(x[i], x[j])  # 计算点积
            norm_product = norms[i] * norms[j]  # 计算模长的乘积
            if norm_product == 0:
                cosine_similarity = 0
            else:
                cosine_similarity = dot_product / norm_product  # 计算余弦相似度
            dist_mat[i, j] = 1 - cosine_similarity  # 计算余弦夹角距离

    return dist_mat


def feature_concat(*F_list, normal_col=False):
    features = None
    for f in F_list:
        if f is not None and f != []:

            if len(f.shape) > 2:
                f = f.reshape(-1, f.shape[-1])

            if normal_col:
                f_max = np.max(np.abs(f), axis=0)
                f = f / f_max

            if features is None:
                features = f
            else:
                features = np.hstack((features, f))
    if normal_col:
        features_max = np.max(np.abs(features), axis=0)
        features = features / features_max
    return features


def hyperedge_concat(*H_list):
    H = None
    for h in H_list:
        # if h is not None and h != []:
        if h is not None and len(h) != 0:
            # for the first H appended to fused hypergraph incidence matrix
            if H is None:
                H = h
            else:
                if type(h) != list:
                    H = np.hstack((H, h))
                else:
                    tmp = []
                    for a, b in zip(H, h):
                        tmp.append(np.hstack((a, b)))
                    H = tmp
    return H


def generate_G_from_H(H, variable_weight=False):
    if type(H) != list:
        return _generate_G_from_H(H, variable_weight)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G


# 它根据给定的超图邻接矩阵 H 生成一个图 G，这个图可以用于图卷积网络。
def _generate_G_from_H(H, variable_weight=False):
    H = np.array(H)

    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = np.ones(n_edge)

    DV = np.sum(H * W, axis=1)
    DE = np.sum(H, axis=0)

    invDE = np.mat(np.diag(np.power(DE, -1)))
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))

    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        G = torch.Tensor(G)
        return G


# 它根据给定的距离矩阵和 k-最近邻算法构建一个超图的邻接矩阵
def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=False, m_prob=1):
    n_obj = dis_mat.shape[0]  # 获取距离矩阵 dis_mat 的行数，即数据点的数量。
    n_edge = n_obj  # 设置边的数量等于数据点的数量，这里可能意味着每个数据点都与图中的其他点相连。
    H = np.zeros((n_obj, n_edge))  # 初始化一个形状为 (n_obj, n_edge) 的零矩阵 H，用于存储邻接信息。
    for center_idx in range(n_obj):  # 遍历每个数据点，center_idx 是当前数据点的索引。
        dis_mat[center_idx, center_idx] = 0
        dis_vec = dis_mat[center_idx]
        nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
        avg_dis = np.average(dis_vec)
        if not np.any(nearest_idx[:k_neig] == center_idx):
            nearest_idx[k_neig - 1] = center_idx

        for node_idx in nearest_idx[:k_neig]:
            if is_probH:
                H[node_idx, center_idx] = np.exp(
                    -dis_vec[0, node_idx] ** 2 / (m_prob * avg_dis) ** 2)  ## affinity matrix的计算公式
            else:
                H[node_idx, center_idx] = 1.0
    return H


def construct_H_with_KNN(X, K_neigs, split_diff_scale=False, is_probH=False, m_prob=1):
    if len(X.shape) != 2:  # 检查输入 X 是否为二维矩阵。
        X = X.reshape(-1, X.shape[-1])  # 如果 X 不是二维的，将其重塑为二维矩阵。

    if type(K_neigs) == int:  # 检查 K_neigs 是否为整数。
        K_neigs = [K_neigs]  # 如果 K_neigs 是整数，将其转换为只包含一个元素的列表。

    dis_mat = Eu_dis(X)  # 计算输入特征矩阵 X 的欧几里得距离矩阵。
    H = []  # 初始化一个空列表 H，用于存储构建的超图结构。
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)

        if not split_diff_scale:
            H = hyperedge_concat(H, H_tmp)
        else:
            H.append(H_tmp)

    return H

