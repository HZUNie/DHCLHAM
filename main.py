import torch
from data_process import prepare_data
import numpy as np
import pandas as pd
from torch import optim
from param import parameter_parser
from module import DHCLHAM
from utils import get_L2reg, Myloss, Metric_fun
from dataset import Dataset
import matplotlib.pyplot as plt
from hypergraph_construct.ConstructHW import constructHW_knn, constructHW_kmean
import xlsxwriter

import warnings

warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(model, train_data, optim, opt):
    model.train()  # 将模型设置为训练模式。
    regression_crit = Myloss()  # 创建自定义的回归损失函数实例。
    Metric = Metric_fun()  # 创建 Metric_fun 的实例用于计算指标

    one_index = train_data[2][0].to(device).t().tolist()  # 获取训练数据中索引为1样本的索引，并将它们转换为列表。
    zero_index = train_data[2][1].to(device).t().tolist()  # 获取训练数据中索引为0的样本的索引，并将它们转换为列表。

    mi_sim_integrate_tensor = train_data[0].to(device)  # 获取训练数据中的DD相似度的整合张量。#
    d_sim_integrate_tensor = train_data[1].to(device)  # 获取训练数据中的MM相似度的整合张量。#

    concat_m = np.hstack(
        [train_data[4].numpy(), mi_sim_integrate_tensor.detach().cpu().numpy()])  # 将相似度信息和原始数据进行拼接[A,MM]
    concat_mi_tensor = torch.FloatTensor(concat_m)  #
    concat_mi_tensor = concat_mi_tensor.to(device)

    # microbe
    G_mi_Kn = constructHW_knn(concat_mi_tensor.detach().cpu().numpy(), K_neigs=[15],
                              is_probH=False)  # 基于Knn构成的超图在转换为图结构
    G_mi_Km = constructHW_kmean(concat_mi_tensor.detach().cpu().numpy(), clusters=[10])  # 基于K-构成的超图在转换为图结构
    G_mi_Kn = G_mi_Kn.to(device)
    G_mi_Km = G_mi_Km.to(device)

    # drug
    concat_d = np.hstack(
        [train_data[4].numpy().T, d_sim_integrate_tensor.detach().cpu().numpy()])  # 将相似度信息和原始数据进行拼接[A,DD]
    concat_d_tensor = torch.FloatTensor(concat_d)
    concat_d_tensor = concat_d_tensor.to(device)

    G_d_Kn = constructHW_knn(concat_d_tensor.detach().cpu().numpy(), K_neigs=[15], is_probH=False)  # 基于Knn构成的超图在转换为图结构
    G_d_Km = constructHW_kmean(concat_d_tensor.detach().cpu().numpy(), clusters=[10])  # 基于K-构成的超图在转换为图结构
    G_d_Kn = G_d_Kn.to(device)
    G_d_Km = G_d_Km.to(device)

    for epoch in range(1, opt.epoch + 1):
        score, mi_cl_loss, dis_cl_loss = model(concat_mi_tensor, concat_d_tensor,
                                               G_mi_Kn, G_mi_Km, G_d_Kn, G_d_Km)

        recover_loss = regression_crit(one_index, zero_index, train_data[4].to(device), score)
        reg_loss = get_L2reg(model.parameters())

        tol_loss = recover_loss + mi_cl_loss + dis_cl_loss + 0.00001 * reg_loss

        optim.zero_grad()
        tol_loss.backward()
        optim.step()
        true_value_one, true_value_zero, pre_value_one, pre_value_zero = test(model, train_data, concat_mi_tensor,
                                                                              concat_d_tensor, G_mi_Kn, G_mi_Km,
                                                                              G_d_Kn, G_d_Km)

        evaluate(true_value_one, true_value_zero, pre_value_one, pre_value_zero, epoch, opt.epoch)


    return true_value_one, true_value_zero, pre_value_one, pre_value_zero


def test(model, data, concat_mi_tensor, concat_dis_tensor, G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km):
    model.eval()
    score, _, _ = model(concat_mi_tensor, concat_dis_tensor,
                        G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km)
    test_one_index = data[3][0].t().tolist()
    test_zero_index = data[3][1].t().tolist()
    true_one = data[5][test_one_index]
    true_zero = data[5][test_zero_index]

    pre_one = score[test_one_index]
    pre_zero = score[test_zero_index]

    return true_one, true_zero, pre_one, pre_zero


def evaluate(true_one, true_zero, pre_one, pre_zero, epoch=None, total_epochs=None):
    Metric = Metric_fun()
    metrics_tensor = np.zeros((1, 7))

    if epoch is not None and total_epochs is not None:
        for seed in range(10):
            test_po_num = true_one.shape[0]
            test_index = np.array(np.where(true_zero == 0))
            np.random.seed(seed)
            np.random.shuffle(test_index.T)
            test_ne_index = tuple(test_index[:, :test_po_num])

            eval_true_zero = true_zero[test_ne_index]
            eval_true_data = torch.cat([true_one, eval_true_zero])

            eval_pre_zero = pre_zero[test_ne_index]
            eval_pre_data = torch.cat([pre_one, eval_pre_zero])

            metrics = Metric.cv_mat_model_evaluate(eval_true_data, eval_pre_data)[:7]
            metrics_tensor = metrics_tensor + np.array(metrics)

        metrics_tensor_avg = metrics_tensor / 10
        auc, aupr, f1_score, accuracy, _, _, _ = metrics_tensor_avg[0]
        print(f"Epoch {epoch}/{total_epochs}:")
        print(f"  AUC: {auc:.4f}, AUPR: {aupr:.4f}, F1-score: {f1_score:.4f}, Accuracy: {accuracy:.4f}")
        return metrics_tensor_avg
    else:
        metrics_list = []
        fpr_list = []
        tpr_list = []
        recall_list = []
        precision_list = []
        np.random.seed(42)

        for seed in range(10):
            test_po_num = true_one.shape[0]
            test_index = np.array(np.where(true_zero == 0))
            np.random.seed(seed)
            np.random.shuffle(test_index.T)
            test_ne_index = tuple(test_index[:, :test_po_num])

            eval_true_zero = true_zero[test_ne_index]
            eval_true_data = torch.cat([true_one, eval_true_zero])

            eval_pre_zero = pre_zero[test_ne_index]
            eval_pre_data = torch.cat([pre_one, eval_pre_zero])

            result = Metric.cv_mat_model_evaluate(eval_true_data, eval_pre_data)
            metrics = result[:7]
            metrics_list.append(metrics)

            fpr = np.array(result[7]).flatten()
            tpr = np.array(result[8]).flatten()
            fpr_list.append(fpr)
            tpr_list.append(tpr)

            recall = np.array(result[9]).flatten()
            precision = np.array(result[10]).flatten()
            recall_list.append(recall)
            precision_list.append(precision)

        metrics_tensor_avg = np.mean(metrics_list, axis=0)
        fpr_avg = np.mean(fpr_list, axis=0)
        tpr_avg = np.mean(tpr_list, axis=0)
        recall_avg = np.mean(recall_list, axis=0)
        precision_avg = np.mean(precision_list, axis=0)

        return metrics_tensor_avg, fpr_avg, tpr_avg, recall_avg, precision_avg

def main(opt):

    dataset = prepare_data(opt)
    train_data = Dataset(opt, dataset)

    metrics_cross = np.zeros((1, 7))
    auc_list = []  # 存储每一折的AUC
    aupr_list = []  # 存储每一折的AUPR
    fold_roc = []
    fold_pr = []

    for i in range(opt.validation):
        print(f"\n=== 开始第 {i + 1} 折交叉验证 ===")
        hidden_list = [256, 256]
        num_proj_hidden = 64

        model = DHCLHAM(args.m_num, args.d_num, hidden_list, num_proj_hidden, args)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        true_score_one, true_score_zero, pre_score_one, pre_score_zero = train_epoch(model, train_data[i], optimizer,
                                                                                     opt)
        metrics_value, fpr, tpr, recall, precision = evaluate(true_score_one, true_score_zero, pre_score_one,
                                                              pre_score_zero)
        metrics_cross = metrics_cross + metrics_value
        auc = metrics_value[0]  # metrics_value是一个(1, 7)数组，取第0行的第0列
        aupr = metrics_value[1]  # 取第0行的第1列
        auc_list.append(auc)
        aupr_list.append(aupr)
        fold_roc.append((fpr, tpr))
        fold_pr.append((recall, precision))

    metrics_cross_avg = metrics_cross / opt.validation
    print('metrics_avg:', metrics_cross_avg)
    auc_avg = np.mean(auc_list)
    aupr_avg = np.mean(aupr_list)
    print(f'五折平均AUC: {auc_avg:.4f}, 五折平均AUPR: {aupr_avg:.4f}')


if __name__ == '__main__':
    args = parameter_parser()
    main(args)