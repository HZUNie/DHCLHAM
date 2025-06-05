import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
import math
from typing import Optional, Callable
from utils import hyperedge_preprocess

# 假设ScaledDotProductAttention_hyper类未变，保留其定义
class ScaledDotProductAttention_hyper(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = attn_dropout

    def forward(self, q, k, v, orginal_attention, feature, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(1, 2))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        if feature != True:
            orginal_attention = orginal_attention.permute(0, 2, 1)
        orginal_attention = F.softmax(orginal_attention, dim=-1)
        attn = F.dropout(F.softmax(attn, dim=-1), self.dropout, training=self.training)
        attn = torch.add(orginal_attention, attn)
        output = torch.matmul(attn, v)
        return output, attn

# 假设hyperedge_preprocess函数存在于.utils中


class ProposedConv(nn.Module):
    def __init__(self, transfer, alpha, in_dim: int, hid_dim: int, out_dim: int, dropout: float = 0.2,
                 act: Callable = nn.PReLU(), bias: bool = True, cached: bool = False,
                 row_norm: bool = True, **kwargs):
        super().__init__(**kwargs)

        self.in_features = in_dim
        self.hid_dim = hid_dim
        self.out_features = out_dim
        self.dropout = dropout
        self.act = act
        self.alpha = alpha
        self.transfer = transfer
        self.cached = cached
        self.row_norm = row_norm
        self.concat = True  # 保留concat标志以控制激活函数

        # 定义权重参数
        if self.transfer:
            self.weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        else:
            self.register_parameter('weight', None)
        self.weight2 = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.weight3 = Parameter(torch.Tensor(self.out_features, self.out_features))

        # 定义全局上下文嵌入
        self.word_context = nn.Embedding(1, self.out_features)

        # 定义超图注意力机制
        self.attention1 = ScaledDotProductAttention_hyper(temperature=self.out_features ** 0.5, attn_dropout=self.dropout)
        self.attention2 = ScaledDotProductAttention_hyper(temperature=self.out_features ** 0.5, attn_dropout=self.dropout)

        # 定义偏置
        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)
        self.weight2.data.uniform_(-stdv, stdv)
        self.weight3.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        nn.init.uniform_(self.word_context.weight.data, -stdv, stdv)

    def forward(self, x: Tensor, hyperedge_index: Tensor,
                num_nodes: Optional[int] = None, num_edges: Optional[int] = None):
        # 处理节点和超边数量
        if num_nodes is None:
            num_nodes = x.shape[0]
        if num_edges is None and hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1

        # 生成超图关联矩阵
        hyperedge_adj = hyperedge_preprocess(hyperedge_index, num_nodes, num_edges)

        # 添加批量维度
        x = x.unsqueeze(0)
        adj = hyperedge_adj.unsqueeze(0).permute(0, 2, 1)

        # 特征变换
        x_4att = x.matmul(self.weight2)

        if self.transfer:
            x = x.matmul(self.weight)
            if self.bias is not None:
                x = x + self.bias

        # 计算查询向量（超边全局上下文）
        N1 = adj.shape[1]  # 超边数
        q1 = self.word_context.weight[0:].view(1, 1, -1).repeat(x.shape[0], N1, 1).view(x.shape[0], N1, self.out_features)

        # 超图注意力：节点到超边
        edge, att1 = self.attention1(q1, x_4att, x, adj, feature=True, mask=adj)
        att2he = att1.permute(0, 2, 1)
        edge_4att = edge.matmul(self.weight3)

        # 超图注意力：超边到节点
        node, att2hn = self.attention2(x_4att, edge_4att, edge, adj, feature=False, mask=adj.transpose(1, 2))

        # 计算注意力输出
        att2_he_out = torch.mm(att2he.squeeze(0), att2he.squeeze(0).t())
        att2_hn_out = torch.mm(att2hn.squeeze(0), att2hn.squeeze(0).t())
        attention_out = torch.add(att2_he_out, att2_hn_out)
        h_prime_g = torch.matmul(attention_out, x_4att.squeeze(0))

        # 应用激活函数
        if self.concat:
            h_prime_g = F.elu(h_prime_g)
            node = F.relu(node)
            edge = F.relu(edge)

        # 移除批量维度
        node = node.squeeze()

        return h_prime_g

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
