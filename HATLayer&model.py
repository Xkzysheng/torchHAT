import torch
import torch.nn as nn
import torch.nn.functional as F

# 2-layer model for node classification
# single and multi head are both supported now
disable_dropout = False
class HyperbolicGraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(HyperbolicGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.EPS = 1e-15
        self.clip_value = torch.tensor(0.98)
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.c = nn.Parameter(torch.tensor([1.0]))
    def f2p_exp_projection(self, f):
        # f表示欧氏空间的原始特征矩阵
        c = 1.0
        f = f + self.EPS
        sqrt_c = c
        f_norm_raw = torch.norm(f, p=2, dim=1, keepdim=True)  # 计算各行（每一个节点的特征向量）的l2范数
        clip_norm = self.clip_value/sqrt_c
        f = f * clip_norm / torch.max(clip_norm, f_norm_raw)  # 将范数沿行裁剪到 clip_value / sqrt_c
        f_norm = torch.norm(f, p=2, dim=1, keepdim=True)
        p = torch.tanh(sqrt_c*f_norm)*f/sqrt_c/f_norm
        return p
    def p2h_log_exp_projection(self, p):
        c = self.c
        p = p + self.EPS
        sqrt_c = torch.sqrt(c)
        p_norm_raw = torch.norm(p, p=2, dim=1, keepdim=True)  # 计算各行（每一个节点的特征向量）的l2范数
        clip_norm_1 = self.clip_value
        p = p * clip_norm_1 / torch.max(clip_norm_1, p_norm_raw)  # 将范数沿行裁剪
        Mp = torch.mm(p, self.W)
        p_norm = torch.norm(p, p=2, dim=1, keepdim=True) 
        atan_norm = torch.atanh(torch.clamp(sqrt_c * p_norm, min=-0.9, max=0.9))
        temp1 = Mp * atan_norm / p_norm / sqrt_c # Mp*atanh(sqrt(c)*norm(p)) / norm(p) / sqrt(c)
        temp1_norm_raw = torch.norm(temp1, p=2, dim=1, keepdim=True)
        clip_norm_2 = self.clip_value / sqrt_c
        temp1 = temp1 * clip_norm_2 / torch.max(clip_norm_2, temp1_norm_raw)  # 将范数沿行裁剪
        temp1_norm = torch.norm(temp1, p=2, dim=1, keepdim=True) # norm of temp1
        #  tanh(sqrt(c)*norm(Mp)*atanh(norm(p))/norm(p)) * norm(p) / sqrt(c) / norm(Mp) / atanh(norm(p)) * Mp * atanh(norm(p)) / norm(p)
        # = 1/sqrt(c) * Mp/norm(Mp) * tanh(sqrt(c)*norm(Mp)/norm(p)*atanh(norm(p)))
        h = torch.tanh(sqrt_c*temp1_norm) / sqrt_c / temp1_norm * temp1
        return h
    def mobius_addition(self, u, v):
        # 输入 [nodes, features]
        c = self.c
        norm_u_sq = torch.norm(u, dim=1) ** 2
        norm_v_sq = torch.norm(v, dim=1) ** 2
        dot_uv = torch.sum(u * v, dim=1) 
        coef_1 = 1 + 2*c*dot_uv + c*norm_v_sq 
        coef_2 = 1 - c*norm_u_sq
        denominator = 1 + 2*c*dot_uv + (c**2)*norm_u_sq*norm_v_sq
        u_mobadd_v = (coef_1.unsqueeze(1)*u+coef_2.unsqueeze(1)*v)/denominator.unsqueeze(1)
        return u_mobadd_v
        # norm_u_sq = torch.norm(u, dim=1) ** 2
        # norm_v_sq = torch.norm(v, dim=1) ** 2
        # uv_dot_times = 4 * torch.mean(u * v, dim=1) * c
        # denominator = 1 + uv_dot_times + norm_u_sq * norm_v_sq * c * c
        # coef_1 = (1 + uv_dot_times + c * norm_v_sq) / denominator
        # coef_2 = (1 - c * norm_u_sq) / denominator
        # return coef_1.unsqueeze(1) * u + coef_2.unsqueeze(1) * v
    def every_edge_mobius_distance(self, x_features, y_features):
        # 输入: 所有有边相连的两节点对应的各自特征组成的列表，长度为全图的边数 [x_features, y_features]
        c = self.c
        mat_add = self.mobius_addition(-x_features, y_features)  # 使用之前定义的 my_prod_mob_addition
        sqrt_c = torch.sqrt(c)
        res_norm = torch.norm(mat_add, dim=1)
        # 使用 torch.clamp 来裁剪数值，并计算 atanh
        res = torch.atanh(torch.clamp(sqrt_c * res_norm, min=1e-8, max=self.clip_value))
        return -2 / sqrt_c * res
    def log_projection(self, h):
        c = self.c
        h = h + self.EPS
        sqrt_c = torch.sqrt(c)
        h_norm_raw = torch.norm(h, p=2, dim=1, keepdim=True)  # 计算各行（每一个节点的特征向量）的l2范数
        clip_norm = self.clip_value
        h = h * clip_norm / torch.max(clip_norm, h_norm_raw)  # 将范数沿行裁剪  
        h_norm = torch.norm(h, p=2, dim=1, keepdim=True) 
        h = torch.atanh(torch.clamp(sqrt_c * h_norm, min=-0.9, max=0.9)) * h / h_norm
        return h
    def exp_projection(self, h):
        c = self.c
        h = h + self.EPS
        sqrt_c = torch.sqrt(c)
        h_norm_raw = torch.norm(h, p=2, dim=1, keepdim=True)  # 计算各行（每一个节点的特征向量）的l2范数
        clip_norm = self.clip_value / sqrt_c
        h = h * clip_norm / torch.max(clip_norm, h_norm_raw)  # 将范数沿行裁剪  
        h_norm = torch.norm(h, p=2, dim=1, keepdim=True) 
        h = torch.tanh(sqrt_c * h_norm) * h / sqrt_c / h_norm        
        return h
    def forward(self, h, adj):
        c = self.c
        p = self.f2p_exp_projection(h)
        h = self.p2h_log_exp_projection(p)
        edge_indices = torch.nonzero(adj)
        node_i_features = h[edge_indices[:, 0]]  # i 对应的特征
        node_j_features = h[edge_indices[:, 1]]  # j 对应的特征
        e = self.every_edge_mobius_distance(node_i_features, node_j_features)
        attention = -9e15 * torch.ones_like(adj)
        attention[edge_indices[:, 0], edge_indices[:, 1]] = e.squeeze(0)[:] 
        attention.fill_diagonal_(0) # 自己的注意力系数最大（为0），邻居的注意力系数都比自己小（全为负数）
        attention = F.softmax(attention, dim=1)
        if disable_dropout == False:
            attention = F.dropout(attention, p=0.6, training=self.training)
        log_h = self.log_projection(h)
        att_log_h = torch.sqrt(c)*torch.mm(attention, log_h)
        h_prime = self.exp_projection(F.elu(att_log_h))
        return h_prime


class HAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nheads, dropout):
        super(HAT, self).__init__()
        self.attentions = [HyperbolicGraphAttentionLayer(nfeat, nhid) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = HyperbolicGraphAttentionLayer(nhid * nheads, nclass)
    def forward(self, x, adj):
        if disable_dropout == False:
            x = F.dropout(x, p=0.6, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        if disable_dropout == False:
            x = F.dropout(x, p=0.6, training=self.training)
        x = self.out_att(x, adj)
        return x # 配合crossentropy_loss (隐含softmax实现）
