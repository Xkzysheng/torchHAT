import torch
import torch.nn as nn
import torch.nn.functional as F
from HATConv import HyperbolicGraphAttentionLayer as HATLayer

# 两层网络，输入-隐藏-输出
class HAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, n_nodes):
        """Dense version of HAT (Single head, Without multi-heads)."""
        super(HAT, self).__init__()
        # hidden_layer
        self.attentions = HATLayer(nfeat, nhid, n_nodes=n_nodes, concat=False)
        # out_layer
        self.out_att = HATLayer(nhid, nclass, n_nodes=n_nodes, concat=False)

    def forward(self, x, adj):
        x = self.attentions(x, adj)
        x = self.out_att(x, adj)
        # 分类问题，最后一层过一遍softmax
        return x # 配合crossentropy_loss (隐含softmax实现）
