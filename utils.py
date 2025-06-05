from torch import nn
from param import parameter_parser
import random
from itertools import permutations

import numpy as np

from torch import Tensor
import pandas as pd

args = parameter_parser()


class Myloss(nn.Module):
    def __init__(self):
        super(Myloss, self).__init__()

    def forward(self, one_index, zero_index, input, target):
        loss = nn.MSELoss(reduction='none')  # 创建一个均方误差损失（MSELoss）对象，设置 reduction='none' 以保留每个样本的损失值。
        loss_sum = loss(input, target)  # 计算模型输出和目标值之间的均方误差损失，并返回每个样本的损失值。

        return (1 - args.alpha) * loss_sum[one_index].sum() + args.alpha * loss_sum[zero_index].sum()  # 计算最终的损失值


# 这段代码定义了一个自定义损失函数类和一个 L2 正则化函数，用于在训练神经网络时计算损失和正则化项。
def get_L2reg(parameters):
    reg = 0  # 初始化正则化项的总和为 0
    for param in parameters:
        reg += 0.5 * (param ** 2).sum()
    return reg


def hyperedge_preprocess(hyperedge_index, num_nodes, num_edges):
    hyperedge_adj = pd.DataFrame(np.zeros((num_nodes, num_edges)), index=range(0, num_nodes),
                                 columns=range(0, num_edges))

    nodes = hyperedge_index.iloc[0, :].tolist()
    edges = hyperedge_index.iloc[1, :].tolist()

    for node, edge in zip(nodes, edges):
        hyperedge_adj[edge][node] = 1

    hyperedge_adj = nn.tensor(hyperedge_adj.values, dtype=nn.float32).to('cuda')

    return hyperedge_adj


class Metric_fun(object):
    def __init__(self):
        super(Metric_fun).__init__()

    def cv_mat_model_evaluate(self, association_mat, predict_mat):
        real_score = np.mat(association_mat.detach().cpu().numpy().flatten())
        predict_score = np.mat(predict_mat.detach().cpu().numpy().flatten())

        return self.get_metrics(real_score, predict_score)

    def get_metrics(self, real_score, predict_score):
        sorted_predict_score = np.array(sorted(list(set(np.array(predict_score).flatten()))))

        sorted_predict_score_num = len(sorted_predict_score)
        thresholds = sorted_predict_score[
            (np.array([sorted_predict_score_num]) * np.arange(1, 1000) / np.array([1000])).astype(int)]
        thresholds = np.mat(thresholds)
        thresholds_num = thresholds.shape[1]

        predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
        negative_index = np.where(predict_score_matrix < thresholds.T)
        positive_index = np.where(predict_score_matrix >= thresholds.T)
        predict_score_matrix[negative_index] = 0
        predict_score_matrix[positive_index] = 1

        TP = predict_score_matrix * real_score.T
        FP = predict_score_matrix.sum(axis=1) - TP
        FN = real_score.sum() - TP
        TN = len(real_score.T) - TP - FP - FN

        fpr = FP / (FP + TN)
        tpr = TP / (TP + FN)
        ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
        ROC_dot_matrix.T[0] = [0, 0]
        ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
        x_ROC = ROC_dot_matrix[0].T
        y_ROC = ROC_dot_matrix[1].T

        auc = 0.5 * (x_ROC[1:] - x_ROC[:-1]).T * (y_ROC[:-1] + y_ROC[1:])

        recall_list = tpr
        precision_list = TP / (TP + FP)
        PR_dot_matrix = np.mat(sorted(np.column_stack((recall_list, -precision_list)).tolist())).T
        PR_dot_matrix[1, :] = -PR_dot_matrix[1, :]
        PR_dot_matrix.T[0] = [0, 1]
        PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
        x_PR = PR_dot_matrix[0].T
        y_PR = PR_dot_matrix[1].T
        aupr = 0.5 * (x_PR[1:] - x_PR[:-1]).T * (y_PR[:-1] + y_PR[1:])

        f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
        accuracy_list = (TP + TN) / len(real_score.T)
        specificity_list = TN / (TN + FP)

        max_index = np.argmax(f1_score_list)
        f1_score = f1_score_list[max_index, 0]
        accuracy = accuracy_list[max_index, 0]
        specificity = specificity_list[max_index, 0]
        recall = recall_list[max_index, 0]
        precision = precision_list[max_index, 0]

        return [auc[0, 0], aupr[0, 0], f1_score, accuracy, recall, specificity, precision, x_ROC, y_ROC, x_PR, y_PR]