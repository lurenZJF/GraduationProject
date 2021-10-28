#!/usr/bin/python
# -*- encoding:utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
from Baseline.GCN.layers import GraphConvolution	  # 简单的GCN层


class GCN(nn.Module):	# nn.Module类的单继承
    def __init__(self, nfeat, nhid, nclass, dropout):
        """
        GCN由两个GraphConvolution层构成,输出为输出层做log_softmax变换的结果
        :param nfeat: 底层节点的参数，feature的个数
        :param nhid: 隐层节点个数
        :param nclass: 最终的分类数
        :param dropout: dropout参数
        """
        super(GCN, self).__init__()
        self.gc1_outdim = nhid
        self.gc2_outdim = nclass
        self.gc1 = GraphConvolution(nfeat, nhid)
        # self.gc1代表GraphConvolution()，gc1输入尺寸nfeat，输出尺寸nhid
        self.gc2 = GraphConvolution(nhid, nclass)
        # self.gc2代表GraphConvolution()，gc2输入尺寸nhid，输出尺寸ncalss
        self.dropout = dropout
        # 这里使用一层MLP
        self.predict = nn.Linear(nclass, nclass)
        # dropout参数

    def forward(self, x, adj):
        """
        :param x: 输入特征
        :param adj: 邻接矩阵
        :return:
        """
        shape = x.shape[0]
        gcn_out = F.relu(self.gc1(x, adj))
        # 第二层GCN输出的图嵌入
        gcn_out= F.relu(self.gc2(gcn_out, adj))
        # training=self.training表示将模型整体的training状态参数传入dropout函数，没有此参数无法进行dropout
        # gcn_out = F.dropout(gcn_out, self.dropout, training=self.training)
        out = self.predict(gcn_out)
        # 输出为输出层做log_softmax变换的结果，dim表示log_softmax将计算的维度
        return F.log_softmax(out, dim=1)