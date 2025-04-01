import random
import os
import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from layers import VariLengthInputLayer, EncodeLayer, FeedForwardLayer
from proposedConv import ProposedConv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_torch(seed=1234)


class TransformerEncoder(nn.Module):
    def __init__(self, input_data_dims, hyperpm):#它接受两个参数：input_data_dims 是输入数据的维度信息，hyperpm 是一个包含超参数的配置对象
        super(TransformerEncoder, self).__init__()
        self.hyperpm = hyperpm
        self.input_data_dims = input_data_dims
        # 下来的几行代码初始化了 Transformer 模型的关键参数，如 d_q（查询维度）、d_k（键维度）、d_v（值维度）、n_head（注意力头数）、dropout（dropout 比率）、n_layer（编码器层数）、modal_num（模态数量）、d_out（输出维度）。
        self.d_q = hyperpm.n_hidden#d_q（查询维度）
        self.d_k = hyperpm.n_hidden#d_k（键维度）
        self.d_v = hyperpm.n_hidden#d_v（值维度）
        self.n_head = hyperpm.n_head#n_head（注意力头数）
        self.dropout = hyperpm.dropout#dropout（dropout 比率）
        self.n_layer = hyperpm.nlayer#n_layer（编码器层数）
        self.modal_num = hyperpm.nmodal#modal_num（模态数量）
        self.d_out = self.d_v * self.n_head * self.modal_num#d_out（输出维度）

        self.InputLayer = VariLengthInputLayer(self.input_data_dims, self.d_k, self.d_v, self.n_head, self.dropout)#这行代码实例化了一个 VariLengthInputLayer 类的对象 InputLayer，用于处理变长输入。

        self.Encoder = []
        self.FeedForward = []

        for i in range(self.n_layer):#循环创建了指定数量的编码器层和前馈网络层，并将它们添加到相应的列表中。
            encoder = EncodeLayer(self.d_k * self.n_head, self.d_k, self.d_v, self.n_head, self.dropout)
            self.add_module('encode_%d' % i, encoder)
            self.Encoder.append(encoder)

            feedforward = FeedForwardLayer(self.d_v * self.n_head, self.d_v * self.n_head, dropout=self.dropout)
            self.add_module('feed_%d' % i, feedforward)
            self.FeedForward.append(feedforward)


    def forward(self, x):#这是模型的前向传播函数，它接受一个输入张量 x。
        bs = x.size(0)#这行代码获取输入张量 x 的批量大小。
        attn_map = []#这行代码初始化了一个空列表 attn_map，用于存储注意力图。
        x, _attn = self.InputLayer(x)#这行代码通过输入层处理输入张量 x，并获取注意力权重。

        attn = _attn.mean(dim=1)
        attn_map.append(attn.detach().cpu().numpy())

        for i in range(self.n_layer):
            x, _attn = self.Encoder[i](q=x, k=x, v=x, modal_num=self.modal_num)
            attn = _attn.mean(dim=1)
            x = self.FeedForward[i](x)
            attn_map.append(attn.detach().cpu().numpy())

        x = x.view(bs, -1)

        # output = self.Outputlayer(x)
        return x

'''
这个 HGNN_conv 类实现了一个基本的异构图卷积层，它可以处理图结构数据的特征提取。
构造函数中初始化了权重和偏置参数，并提供了参数重置的方法。前向传播函数定义了数据通过网络的流程，包括矩阵乘法和图结构的应用。
__repr__ 方法提供了类的字符串表示，方便调试和日志记录。
'''
class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()
        self.in_features = in_ft
        self.out_features = out_ft
        self.weight = Parameter(torch.Tensor(in_ft, out_ft))#创建一个可训练的参数 weight，其大小为 [in_ft, out_ft]，表示卷积核的权重。
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):#定义了参数初始化的方法
        stdv = 1. / math.sqrt(self.weight.size(1))#计算权重的标准化标准差
        self.weight.data.uniform_(-stdv, stdv)#使用均匀分布初始化权重
        if self.bias is not None:#如果存在偏置项，同样使用均匀分布初始化
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, G):#定义了模型的前向传播函数，接受输入特征 x 和图结构 G。
        x = x.matmul(self.weight)
        if self.bias is not None:#如果存在偏置项
            x = x + self.bias#将偏置项加到特征上
        x = G.matmul(x)#将图结构 G 与特征相乘，这可能表示应用邻接矩阵或其他图操作。
        return x#返回经过卷积操作的特征。

    def __repr__(self):#定义了类的字符串表示方法，用于输出类的可读性描述。
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


#这个 HGCN 类实现了一个简单的异构图卷积网络组件，它通过一个 HGNN_conv 层来学习节点的嵌入表示，并通过 LeakyReLU 激活函数进行非线性变换。
# 这个组件可以作为更复杂图神经网络模型的一部分，用于提取图中节点的特征表示。
# HGCN 类的前向传播函数定义了数据通过网络的流程，并返回处理后的特征表示。
class HGCN(nn.Module):
    def __init__(self, in_dim, hidden_list, dropout = 0.5):
        super(HGCN, self).__init__()
        self.dropout = dropout

        self.hgnn1 = HGNN_conv(in_dim, hidden_list[0])

    def forward(self,x, G):#定义了模型的前向传播函数，接受输入特征 x 和图结构 G。

        x_embed = self.hgnn1(x, G)#将输入特征 x 和图结构 G 通过 HGNN_conv 层进行卷积操作，得到嵌入表示 x_embed。
        x_embed_1 = F.leaky_relu(x_embed, 0.25)#应用 LeakyReLU 激活函数到嵌入表示 x_embed，其负斜率为 0.25。


        return x_embed_1#返回经过 LeakyReLU 激活函数处理后的嵌入表示。

class CL_HGCN(nn.Module):
    def __init__(self, in_size, hid_list, num_proj_hidden, alpha = 0.5,act = nn.PReLU()):
    #def __init__(self, in_size, hid_list, num_proj_hidden, alpha=0.5, act=nn.PReLU()):
        super(CL_HGCN, self).__init__()
        self.hgcn1 = HGCN(in_size, hid_list)#超图卷积
        self.hgcn2 = HGCN(in_size, hid_list)#超图卷积

        #self.AttCn1 = ProposedConv(self.transfer, alpha, in_size, 256, 256, cached=False, act=act)
        #self.AttCn2 = ProposedConv(self.transfer, alpha, in_size, 256, 256, cached=False, act=act)

        self.fc1 = torch.nn.Linear(hid_list[-1], num_proj_hidden)#这行代码定义了一个全连接层 fc1，它将 HGCN 的输出维度映射到 num_proj_hidden 维度256->64
        self.fc2 = torch.nn.Linear(num_proj_hidden, hid_list[-1])#64->256

        self.tau = 0.5
        self.alpha = alpha

    def forward(self, x1, adj1, x2, adj2):
        #forward(self, x1, adj1, x2, adj2, Knn_adj, Km_adj):
        #knn_hyperedge_index = torch.nonzero(Knn_adj, as_tuple=False).T
        z1 = self.hgcn1(x1, adj1)

        #A_z1=self.AttCn1(x1, knn_hyperedge_index,Knn_adj.shape[0],Km_adj.shape[1])

        h1 = self.projection(z1)
        #A_h1=self.projection(A_z1)

       # km_hyperedge_index = torch.nonzero(Km_adj, as_tuple=False).T
        z2 = self.hgcn2(x2, adj2)
       # A_z2 = self.AttCn1(x2, km_hyperedge_index,Knn_adj.shape[0],Km_adj.shape[1])
        h2 = self.projection(z2)
        #A_h2 = self.projection(A_z2)

        loss = self.alpha*self.sim(h1, h2) + (1-self.alpha)*self.sim(h2,h1)#计算相似度损失

        return z1, z2, loss

    def projection(self, z):
        z = F.elu(self.fc1(z))
        return self.fc2(z)

#norm_sim 方法用于计算特征向量的规范化相似度
    def norm_sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())#用于执行矩阵乘法

#则基于这些相似度计算一个损失值
    def sim(self, z1, z2):
        f = lambda x: torch.exp(x / self.tau)#定义一个匿名函数 f，它接受一个参数 x 并返回 torch.exp(x / self.tau) 的结果
        refl_sim = f(self.norm_sim(z1, z1))
        between_sim = f(self.norm_sim(z1, z2))
        loss = -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        loss = loss.sum(dim=-1).mean()
        return loss


class HGCN_Attention_mechanism(nn.Module):
    def __init__(self):
        super(HGCN_Attention_mechanism,self).__init__()
        self.hiddim = 64

        self.fc_x1 = nn.Linear(in_features=2, out_features=self.hiddim)
        self.fc_x2 = nn.Linear(in_features=self.hiddim, out_features=2)
        self.sigmoidx = nn.Sigmoid()


    def forward(self,input_list):#这是模型的前向传播函数，它接受一个输入列表 input_list，其中包含两个张量。

        XM = torch.cat((input_list[0], input_list[1]), 1).t()#这行代码将输入列表中的两个张量沿着第二个维度（特征维度）拼接起来，然后转置。
        XM = XM.view(1, 1 * 2, input_list[0].shape[1], -1)#这行代码将拼接后的张量 XM 重塑为四维张量，以适应后续的二维平均池化操作。

        globalAvgPool_x = nn.AvgPool2d((input_list[0].shape[1], input_list[0].shape[0]), (1, 1))#这行代码定义了一个二维平均池化层 globalAvgPool_x，池化窗口大小与输入张量的空间维度相同。
        x_channel_attenttion = globalAvgPool_x(XM)#这行代码应用二维平均池化到 XM，得到通道注意力的全局平均表示。

        x_channel_attenttion = x_channel_attenttion.view(x_channel_attenttion.size(0), -1)#这行代码将池化后的张量重塑为二维张量，以适应全连接层。
        x_channel_attenttion = self.fc_x1(x_channel_attenttion)#这行代码通过全连接层 fc_x1 处理通道注意力表示
        x_channel_attenttion = torch.relu(x_channel_attenttion)#这行代码应用 ReLU 激活函数到通道注意力表示。
        x_channel_attenttion = self.fc_x2(x_channel_attenttion)#这行代码通过全连接层 fc_x2 进一步处理通道注意力表示。
        x_channel_attenttion = self.sigmoidx(x_channel_attenttion)#这行代码应用 Sigmoid 激活函数到通道注意力表示，得到最终的注意力权重。
        x_channel_attenttion = x_channel_attenttion.view(x_channel_attenttion.size(0), x_channel_attenttion.size(1), 1, 1)#这行代码将注意力权重重塑为四维张量，以适应后续的张量乘法操作。

        XM_channel_attention = x_channel_attenttion * XM#这行代码将注意力权重与原始张量 XM 相乘，应用通道注意力。
        XM_channel_attention = torch.relu(XM_channel_attention)#这行代码应用 ReLU 激活函数到应用了通道注意力的张量。

        return XM_channel_attention[0]#这行代码返回应用了通道注意力的张量的第一个元素。


class DHCLHAM(nn.Module):
    def __init__(self, m_num, d_num, hidd_list, num_proj_hidden,hyperpm):
        super(DHCLHAM, self).__init__()

        self.CL_HGCN_mi = CL_HGCN(m_num + d_num, hidd_list,num_proj_hidden)#对mi进行超图卷积提取
        self.CL_HGCN_d = CL_HGCN(d_num + m_num, hidd_list,num_proj_hidden)#d进行超图卷积提取

        self.CL_AttCn_mi = CL_HGCN(m_num + d_num, hidd_list, num_proj_hidden)  # 对mi进行超图卷积提取
        self.CL_AttCn_d = CL_HGCN(d_num + m_num, hidd_list, num_proj_hidden)  # d进行超图卷积提取



        #创建两个注意力机制的实例，
        self.AM_m = HGCN_Attention_mechanism()
        self.AM_d = HGCN_Attention_mechanism()

        self.Transformer_m = TransformerEncoder([hidd_list[-1],hidd_list[-1]], hyperpm)
        self.Transformer_d = TransformerEncoder([hidd_list[-1],hidd_list[-1]], hyperpm)

      
        self.linear_x_1 = nn.Linear(hyperpm.n_head*hyperpm.n_hidden*hyperpm.nmodal, 256)
        self.linear_x_2 = nn.Linear(256, 128)
        self.linear_x_3 = nn.Linear(128, 64)

        self.linear_y_1 = nn.Linear(hyperpm.n_head*hyperpm.n_hidden*hyperpm.nmodal, 256)
        self.linear_y_2 = nn.Linear(256, 128)
        self.linear_y_3 = nn.Linear(128, 64)


    def forward(self, concat_mi_tensor, concat_d_tensor, G_mi_Kn, G_mi_Km, G_d_Kn, G_d_Km):#G_mi_K是Km视图的拉普拉斯矩阵。

        mi_embedded = concat_mi_tensor#
        d_embedded = concat_d_tensor

        
        mi_feature1, mi_feature2, mi_cl_loss = self.CL_HGCN_mi(mi_embedded, G_mi_Kn, mi_embedded, G_mi_Km)
        mi_feature1, mi_feature2, mi_cl_loss = self.CL_HGCN_mi(mi_embedded, G_mi_Kn, mi_embedded, G_mi_Km)


        mi_feature_att = self.AM_m([mi_feature1,mi_feature2])
        mi_feature_att1 = mi_feature_att[0].t()
        mi_feature_att2 = mi_feature_att[1].t()
        mi_concat_feature = torch.cat([mi_feature_att1, mi_feature_att2], dim=1)
        mi_feature = self.Transformer_m(mi_concat_feature)

       
        d_feature1, d_feature2, d_cl_loss = self.CL_HGCN_d(d_embedded, G_d_Kn, d_embedded, G_d_Km)
        d_feature_att = self.AM_d([d_feature1,d_feature2])
        d_feature_att1 = d_feature_att[0].t()
        d_feature_att2 = d_feature_att[1].t()
        d_concat_feature = torch.cat([d_feature_att1, d_feature_att2], dim=1)
        d_feature = self.Transformer_d(d_concat_feature)

        x1 = torch.relu(self.linear_x_1(mi_feature))
        x2 = torch.relu(self.linear_x_2(x1))
        x = torch.relu(self.linear_x_3(x2))

        y1 = torch.relu(self.linear_y_1(d_feature))
        y2 = torch.relu(self.linear_y_2(y1))
        y = torch.relu(self.linear_y_3(y2))

        
        score = x.mm(y.t())

        return score, mi_cl_loss, d_cl_loss



