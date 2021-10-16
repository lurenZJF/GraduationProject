import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv


class SemanticAttention(nn.Module):
    # 语义级别的GAT
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=True)  # 这里原先是False
        )

    def forward(self, z):
        w = self.project(z)  # tanh(Wz+b)
        beta = torch.softmax(w, dim=1)  # 经过softmax()得到语义权重
        return (beta * z).sum(1)   # 返回语义层的嵌入

class HANLayer(nn.Module):
    """
    HAN layer.
    Arguments
    ---------
    num_meta_paths : number of homogeneous graphs generated from the metapaths.
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability
    Inputs
    ------
    g : list[DGLGraph]
        List of graphs
    h : tensor
        Input features
    Outputs
    -------
    tensor
        The output feature
    """
    def __init__(self, num_meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(num_meta_paths):
            #ELU激活函数 当x大于0时不变
            self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads,
                                           dropout, dropout, activation=F.elu))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.num_meta_paths = num_meta_paths

    def forward(self, gs, h):
        semantic_embeddings = []
        for i, g in enumerate(gs):
            # 每次循环根据一个图返回一条meat-path上的嵌入
            semantic_embeddings.append(self.gat_layers[i](g, h).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D * K)
        return self.semantic_attention(semantic_embeddings)   # (N, D * K)

class HAN(nn.Module):
    def __init__(self, num_meta_paths, house_nums, in_size, hidden_size, out_size, num_heads, dropout):
        super(HAN, self).__init__()
        self.house_nums = house_nums
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(num_meta_paths, in_size, hidden_size, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(num_meta_paths, hidden_size * num_heads[l-1],
                                        hidden_size, num_heads[l], dropout))
        # 根据最终嵌入结果产生预测
        # 这里使用一层MLP
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)

    def forward(self, g, h):
        for gnn in self.layers:
            h = gnn(g, h)
        predict_result = self.predict(h)
        return predict_result.reshape(self.house_nums*1)
