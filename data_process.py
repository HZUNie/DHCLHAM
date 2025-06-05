import csv
import os
import torch as t
import numpy as np
from math import e
import pandas as pd
from scipy import io
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import norm
import nonfusion



def read_csv(path):
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        md_data = []
        md_data += [[float(i) for i in row] for row in reader]
        return t.FloatTensor(md_data)


def read_txt(path):
    with open(path, 'r', newline='') as txt_file:
        reader = txt_file.readlines()
        md_data = []
        md_data += [[float(i) for i in row.split()] for row in reader]
        return t.FloatTensor(md_data)


def read_mat(path, name):
    matrix = io.loadmat(path)
    matrix = t.FloatTensor(matrix[name])
    return matrix


def read_md_data(path, validation):
    result = [{} for _ in range(validation)]
    for filename in os.listdir(path):
        data_type = filename[filename.index('_')+1:filename.index('.')-1]
        num = int(filename[filename.index('.')-1])
        result[num-1][data_type] = read_csv(os.path.join(path, filename))
    return result


def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.size(0)):
        for j in range(matrix.size(1)):
            if matrix[i][j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return t.LongTensor(edge_index)


def Gauss_M(adj_matrix, N):
    GM = np.zeros((N, N))
    rm = N * 1. / sum(sum(adj_matrix * adj_matrix))
    for i in range(N):
        for j in range(N):
            GM[i][j] = e ** (-rm * (np.dot(adj_matrix[i, :] - adj_matrix[j, :], adj_matrix[i, :] - adj_matrix[j, :])))
    return GM


def Gauss_D(adj_matrix, M):
    GD = np.zeros((M, M))#这行代码初始化一个 M 行 M 列的零矩阵 GD，用于存储最终的高斯距离矩阵。
    T = adj_matrix.transpose()#这行代码计算 adj_matrix 的转置矩阵 T。
    rd = M * 1. / sum(sum(T * T))#这行代码计算一个缩放因子 rd。它首先计算矩阵 T 的元素平方和，然后除以 M，得到每个节点的平均平方距离。这个缩放因子用于调整高斯核的宽度。
    for i in range(M):
        for j in range(M):
            GD[i][j] = e ** (-rd * (np.dot(T[i] - T[j], T[i] - T[j])))
    return GD


def prepare_data(opt):
    dataset = {}#初始化一个空字典，用于存储数据集相关的信息

    # dd_data = pd.read_csv(opt.data_path + '/df.csv',index_col=0)#使用pandas库读取CSV文件，并将第一列作为索引。
    # dd_mat = np.array(dd_data)#将读取的数据转换为NumPy数组。
    mm_data = pd.read_csv(opt.data_path + '/mf.csv',index_col=0)#使用pandas库读取CSV文件，并将第一列作为索引。
    mm_mat = np.array(mm_data)#将读取的数据转换为NumPy数组。

    # mm_data = pd.read_csv(opt.data_path + '/mf.csv',index_col=0)
    # mm_mat = np.array(mm_data)
    dd_data = pd.read_csv(opt.data_path + '/df.csv',index_col=0)
    dd_mat = np.array(dd_data)

    #mi_dis_data = pd.read_csv(opt.data_path + '/adj.csv',index_col=0)
    m_d_data = pd.read_csv(opt.data_path + '/adj.csv',index_col=0)

    dataset['md_p'] = t.FloatTensor(np.array(m_d_data))#将读取的mi_dis_data转换为PyTorch的浮点张量，并存储在dataset字典中。这个是邻接矩阵
    dataset['md_true'] = dataset['md_p']# 创建一个与md_p相同的副本，存储为md_true。

    all_zero_index = []#初始化空列表，用于存储所有值为0的索引位置。
    all_one_index = []#初始化空列表，用于存储所有值为1的索引位置。
    for i in range(dataset['md_p'].size(0)):#的循环遍历md_p张量的每个元素，将值为0和1的元素的索引位置分别添加到all_zero_index和all_one_index列表中。
        for j in range(dataset['md_p'].size(1)):
            if dataset['md_p'][i][j] < 1:
                all_zero_index.append([i, j])
            if dataset['md_p'][i][j] >= 1:
                all_one_index.append([i, j])


    np.random.seed(0)#设置随机种子为0，然后打乱all_zero_index和all_one_index列表的顺序。
    np.random.shuffle(all_zero_index)
    np.random.shuffle(all_one_index)


    zero_tensor = t.LongTensor(all_zero_index)#将all_zero_index列表转换为PyTorch的长整型张量。
    zero_index = zero_tensor.split(int(zero_tensor.size(0) / 10), dim=0)#将zero_tensor张量分割成10份，每份包含大约相等的元素数量。
    one_tensor = t.LongTensor(all_one_index)
    one_index = one_tensor.split(int(one_tensor.size(0) / 10), dim=0)

    
    cross_zero_index = t.cat([zero_index[i] for i in range(9)])#zero_index列表的前9个元素合并成一个张量，用于交叉验证。
    cross_one_index = t.cat([one_index[j] for j in range(9)])#将one_index列表的前9个元素合并成一个张量。
    new_zero_index = cross_zero_index.split(int(cross_zero_index.size(0) / opt.validation), dim=0)#将cross_zero_index张量进一步分割，以适应opt.validation指定的验证组数
    new_one_index = cross_one_index.split(int(cross_one_index.size(0) / opt.validation), dim=0)#同上，对cross_one_index进行分割。
    dataset['md'] = []#初始化一个空列表，用于存储每个验证组的数据。
    for i in range(opt.validation):#开始一个循环，循环次数为opt.validation指定的验证组数。
        a = [i for i in range(opt.validation)]#创建一个包含0到opt.validation-1的整数列表。
        if opt.validation != 1:# 如果验证组数不为1，执行以下操作。
            del a[i]# 从列表a中删除当前索引i的元素，以确保在交叉验证中，测试集不包括当前的验证组。
        dataset['md'].append({'test': [new_one_index[i], new_zero_index[i]],#将测试集的索引（1和0的索引）添加到dataset['md']列表的当前元素中。
                              'train': [t.cat([new_one_index[j] for j in a]), t.cat([new_zero_index[j] for j in a])]})#将训练集的索引（1和0的索引）添加到dataset['md']列表的当前元素中。训练集索引是通过合并列表a中的索引对应的new_one_index和new_zero_index张量得到的。

    
    dataset['independent'] = []#初始化一个空列表，用于存储独立测试集的数据。
    in_zero_index_test = zero_index[-2]#选取zero_index列表的倒数第二个元素作为独立测试集的0索引。
    in_one_index_test = one_index[-2]#选取one_index列表的倒数第二个元素作为独立测试集的1索引。
    dataset['independent'].append({'test': [in_one_index_test, in_zero_index_test],
                                   'train': [cross_one_index,cross_zero_index]})


    MGSM = Gauss_D(dataset['md_p'].numpy(), dataset['md_p'].size(1))#调用Gauss_D函数计算基于md_p数据的高斯相似度矩阵，并将结果存储在变量DGSM中
    DGSM = Gauss_M(dataset['md_p'].numpy(), dataset['md_p'].size(0))#调用Gauss_D函数计算基于md_p数据的高斯相似度矩阵，并将结果存储在变量DGSM中


    '''nonfusion'''
    k1 = int(m_d_data.shape[1]/8)
    k2 = int(m_d_data.shape[0]/8)
    '''mic'''
    m1 = nonfusion.new_normalization(mm_mat)
    m2 = nonfusion.new_normalization(MGSM)
    Sm_1 = nonfusion.KNN_kernel(mm_mat, k1)
    Sm_2 = nonfusion.KNN_kernel(MGSM, k1)

    Pm = nonfusion.updating(Sm_1, Sm_2, m1, m2)
    Pm_final = (Pm + Pm.T) / 2
    IM=Pm_final


    '''drug'''
    d1 = nonfusion.new_normalization(dd_mat)
    d2 = nonfusion.new_normalization(DGSM)
    Sd_1 = nonfusion.KNN_kernel(dd_mat, k2)
    Sd_2 = nonfusion.KNN_kernel(DGSM, k2)

    Pd = nonfusion.updating(Sd_1, Sd_2, d1, d2)
    Pd_final = (Pd + Pd.T) / 2
    ID=Pd_final

    '''nonfusion'''
    # nd = mi_dis_data.shape[1]
    # nm = mi_dis_data.shape[0]
    #
    # ID = np.zeros([nd, nd])# 初始化一个形状为[nd, nd]的全零NumPy数组，得到最后的DD综合相似度
    #
    # for h1 in range(nd):
    #     for h2 in range(nd):
    #         if dd_mat[h1, h2] == 0:
    #             ID[h1, h2] = DGSM[h1, h2]
    #         else:
    #             ID[h1, h2] = (dd_mat[h1, h2] + DGSM[h1, h2]) / 2
    #
    #
    # IM = np.zeros([nm, nm])#初始化一个形状为[nm, nm]的全零NumPy数组，得到最后的MM综合相似度
    #
    # for q1 in range(nm):
    #     for q2 in range(nm):
    #         if mm_mat[q1, q2] == 0:
    #             IM[q1, q2] = MGSM[q1, q2]
    #         else:
    #             IM[q1, q2] = (mm_mat[q1, q2] + MGSM[q1, q2]) / 2

    dataset['ID'] = t.from_numpy(ID)#将计算得到的ID相似度矩阵转换为PyTorch张量，并存储在dataset字典中。
    dataset['IM'] = t.from_numpy(IM)#将计算得到的IM相似度矩阵转换为PyTorch张量，并存储在dataset字典中。

    return dataset


def RWR(SM):
    alpha = 0.1
    nd = len(SM)  # 获取矩阵的维度
    E = np.identity(nd)  # 单位矩阵
    M = np.zeros((nd, nd))

    # 构建转移矩阵 M
    for i in range(nd):
        for j in range(nd):
            M[i][j] = SM[i][j] / (np.sum(SM[i, :]))

    # 随机游走过程
    s = np.zeros((nd, nd))  # 初始化一个二维数组来存储结果
    for i in range(nd):
        e_i = E[i, :]  # 初始分布
        p_i1 = np.copy(e_i)
        for j in range(10):
            p_i = np.copy(p_i1)
            p_i1 = alpha * (np.dot(M, p_i)) + (1 - alpha) * e_i
        s[i, :] = p_i1  # 将最终分布存储在二维数组的第 i 行
    return s



#knsn
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import norm

def KSNS_opt(X, sim=None):
    if sim is None:
        distance = pdist(X)
        distance = squareform(distance)
        sim = np.max(distance) - distance

    neighbor_num = int(0.3 * X.shape[0])
    feature_matrix = X
    nearst_neighbor_matrix = KSNS_neighbors(sim, neighbor_num)
    S, objv = jisuanW_opt(feature_matrix, nearst_neighbor_matrix)
    return S

def KSNS_neighbors(sim, neighbor_num):
    N = sim.shape[0]
    D = sim - np.diag(np.inf * np.ones(N))
    si = np.argsort(D, axis=1)[:, :neighbor_num]
    nearst_neighbor_matrix = np.zeros((N, N))
    for i in range(N):
        nearst_neighbor_matrix[i, si[i, :]] = 1
    return nearst_neighbor_matrix

def jisuanW_opt(feature_matrix, nearst_neighbor_matrix):
    lata1 = 4
    lata2 = 1
    X = feature_matrix.T
    N = X.shape[1]
    np.random.seed(1)
    W = np.random.rand(N, N) - np.diag(np.diag(np.random.rand(N, N)))
    W = W - np.diag(np.diag(W))
    W = W / np.sum(W, axis=0, keepdims=True)
    C = nearst_neighbor_matrix.T
    G = jisuan_Kel(X)
    G = np.nan_to_num(G, nan=0.0)
    G = G / np.max(G)

    WC1 = W.T @ G @ W - 2 * W @ G + G
    WC = np.sum(np.diag(WC1)) / 2
    wucha = WC + norm(W * (1 - C), 'fro')**2 * lata1 / 2 + norm(W, 'fro')**2 * lata2 / 2
    objv = [wucha]
    jingdu = 0.0001
    error = jingdu * (1 + lata1 + lata2)
    we = 1
    gen = 0
    while gen < 100 and we > error:
        FZ = G + lata1 * C * W
        FM = G @ W + lata1 * W + lata2 * W
        FM[FM == 0] = np.finfo(float).eps
        W = FZ / FM * W
        WC1 = W.T @ G @ W - 2 * W @ G + G
        WC = np.sum(np.diag(WC1)) / 2
        objv1 = WC + norm(W * (1 - C), 'fro')**2 * lata1 / 2 + norm(W, 'fro')**2 * lata2 / 2
        we = abs(objv1 - objv[-1])
        objv.append(objv1)
        gen += 1
    W = matrix_normalize(W)
    return W, objv

def matrix_normalize(W):
    K = 10
    W = np.nan_to_num(W, nan=0.0)
    W[np.arange(W.shape[0]), np.arange(W.shape[1])] = 0
    for round in range(K):
        SW = np.sum(W, axis=1)
        ind = np.where(SW > 0)[0]
        SW[ind] = 1.0 / np.sqrt(SW[ind])
        D1 = np.diag(SW)
        W = D1 @ W @ D1
    return W

def jisuan_Kel(X):
    X = X.T
    sA = np.sum(X**2, axis=1)
    sB = np.sum(X**2, axis=1)
    K = np.exp(-2 * X @ X.T + sA[:, None] - sB[None, :]) / np.mean(sA)
    return K







