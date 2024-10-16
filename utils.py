import numpy as np
import scipy.sparse as sp
import torch
import warnings
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingWarmRestarts
import math
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from collections import defaultdict

def encode_onehot(labels):
    # The classes must be sorted before encoding to enable static class encoding.
    # In other words, make sure the first class always maps to index 0.
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def stratified_split622(labels, train_size=0.6, val_size=0.2, test_size=0.2):
    """根据类别均匀划分数据集"""
    num_samples = labels.shape[0]
    num_classes = labels.max() + 1
    indices = np.arange(num_samples)
    train_indices = []
    val_indices = []
    test_indices = []

    for i in range(num_classes):
        class_indices = indices[labels == i]
        np.random.shuffle(class_indices)
        train_end = int(train_size * len(class_indices))
        val_end = int(val_size * len(class_indices)) + train_end
        
        train_indices.extend(class_indices[:train_end])
        val_indices.extend(class_indices[train_end:val_end])
        test_indices.extend(class_indices[val_end:])
    
    return np.array(train_indices), np.array(val_indices), np.array(test_indices)


def stratified_split205001000(labels, num_per_class=20, val_size=500, test_size=1000):
    """根据类别划分数据集，训练集每类取20条，验证集和测试集分别随机取500和1000条"""
    num_samples = labels.shape[0]
    num_classes = labels.max() + 1
    indices = np.arange(num_samples)
    train_indices = []
    # 记录已经被用作训练集的样本，避免后续验证集和测试集中包含它们
    used_indices = set()
    # 每个类别取 num_per_class 条数据
    for i in range(num_classes):
        class_indices = indices[labels == i]
        np.random.shuffle(class_indices)
        train_indices.extend(class_indices[:num_per_class])
        used_indices.update(class_indices[:num_per_class])
    # 剩余数据（非训练集）作为候选
    remaining_indices = np.array(list(set(indices) - used_indices))
    np.random.shuffle(remaining_indices)
    # 验证集从剩余数据中随机选取 val_size 条
    val_indices = remaining_indices[:val_size]
    used_indices.update(val_indices)
    # 测试集从剩余数据（排除已选中的验证集）中随机选取 test_size 条
    remaining_indices = np.array(list(set(remaining_indices) - used_indices))
    np.random.shuffle(remaining_indices)
    test_indices = remaining_indices[:test_size]
    return np.array(train_indices), np.array(val_indices), np.array(test_indices)


def load_data(dataset, ratio='622', path="./"):
    """加载并划分数据集"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])
    if dataset == 'cora':
        # 建立图结构
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    elif dataset == 'citeseer':
        # 建立图结构
        idx = np.array(idx_features_labels[:, 0], dtype=np.dtype(str))
        # 创建字符串到整数的映射
        id_to_int = {id_str: id_int for id_int, id_str in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.dtype(str))
        edges_unordered[:, 0] = np.array([id_to_int.get(id_str, -1) for id_str in edges_unordered[:, 0]])
        edges_unordered[:, 1] = np.array([id_to_int.get(id_str, -1) for id_str in edges_unordered[:, 1]])
        # 将 edges_unordered 的 ID 转换为整数，处理缺失的 ID
        edges = np.array([id_to_int.get(id_str, -1) for id_str in edges_unordered.flatten()], dtype=np.int32).reshape(edges_unordered.shape)
        # 创建布尔掩码，确定哪些行不包含 -1
        valid_edges_mask = ~(edges == -1).any(axis=1)
        # 过滤出有效的边
        valid_edges = edges[valid_edges_mask]
        # 使用有效的边构建稀疏矩阵
        adj = sp.coo_matrix((np.ones(valid_edges.shape[0]), (valid_edges[:, 0], valid_edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    else:
        pass
    # 构建对称邻接矩阵
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    features = normalize_features(features)

    # 获取标签的整数形式
    labels_int = np.where(labels)[1]
    
    # 使用分层采样方法划分数据集
    if ratio == '622':
        idx_train, idx_val, idx_test = stratified_split622(labels_int, train_size=0.6, val_size=0.2, test_size=0.2)
    elif ratio == '205001000':
        idx_train, idx_val, idx_test = stratified_split205001000(labels_int, train_size=0.6, val_size=0.2, test_size=0.2)
    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels_int)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

def load_data_pubmed(ratio='622'):
    print(f'Loading pubmed dataset...')
    pubmed_content_file = './Pubmed-Diabetes.NODE.paper.tab'
    pubmed_cite_file = './Pubmed-Diabetes.DIRECTED.cites.tab'

    feat_data = []
    labels = [] # label sequence of node
    node_map = {} # map node to Node_ID
    with open(pubmed_content_file) as fp:
        fp.readline()
        feat_map = {entry.split(":")[1]:i-1 for i,entry in enumerate(fp.readline().split("\t"))}
        for i, line in enumerate(fp):
            info = line.split("\t")
            node_map[info[0]] = i
            labels.append(int(info[1].split("=")[1])-1)
            tmp_list = np.zeros(len(feat_map)-2)
            for word_info in info[2:-1]:
                word_info = word_info.split("=")
                tmp_list[feat_map[word_info[0]]] = float(word_info[1])
            feat_data.append(tmp_list)
    feat_data = np.asarray(feat_data)
    labels = np.asarray(labels, dtype=np.int64)
    adj_lists = defaultdict(set)
    with open(pubmed_cite_file) as fp:
        fp.readline()
        fp.readline()
        for line in fp:
            info = line.strip().split("\t")
            paper1 = node_map[info[1].split(":")[1]]
            paper2 = node_map[info[-1].split(":")[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    assert len(feat_data) == len(labels) == len(adj_lists)
    features = normalize_features(sp.csr_matrix(feat_data))
    num_nodes = feat_data.shape[0]  # 假设 node_map 中有节点的总数
    adj = np.zeros((num_nodes, num_nodes))
    for node, neighbors in adj_lists.items():
        for neighbor in neighbors:
            adj[node, neighbor] = 1  # 或者其他适合的值表示边的存在
    np.fill_diagonal(adj, 1)
    # 获取标签的整数形式
    labels_int = labels
    # 使用分层采样方法划分数据集
    if ratio == '622':
        idx_train, idx_val, idx_test = stratified_split622(labels_int, train_size=0.6, val_size=0.2, test_size=0.2)
    elif ratio == '205001000':
        idx_train, idx_val, idx_test = stratified_split205001000(labels_int, train_size=0.6, val_size=0.2, test_size=0.2)
    else:
        pass
    adj = torch.FloatTensor(np.array(adj))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels_int)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def macro_f1(output, labels):
    preds = output.max(1)[1].type_as(labels)
    macro_f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
    return macro_f1

class CosineAnnealingWarmRestartsWithStepScale(CosineAnnealingWarmRestarts):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, step_scale=1.0, last_epoch=-1, verbose=False):
        super(CosineAnnealingWarmRestartsWithStepScale, self).__init__(optimizer, T_0, T_mult, eta_min, last_epoch, verbose)
        self.step_scale = step_scale
        self.eta_max = [group['lr'] for group in optimizer.param_groups]  # 初始化 eta_max

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        return [self.eta_min + (eta_max - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
            for eta_max in self.eta_max]

    def step(self, epoch=None):
        if epoch is None and self.last_epoch < 0:
            epoch = 0
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
                # 每次重启时，更新 eta_max
                # 压低最后一次重启的lr上限
                if epoch >= 1820:
                    self.eta_max = [0.5 * eta_max * self.step_scale for eta_max in self.eta_max]
                else:
                    self.eta_max = [eta_max * self.step_scale for eta_max in self.eta_max]
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
                    # 每次重启时，更新 eta_max
                    self.eta_max = [eta_max * self.step_scale for eta_max in self.eta_max]
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        self.last_epoch = math.floor(epoch)

        class _enable_get_lr_call:
            def __init__(self, o):
                self.o = o
            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self
            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self
        with _enable_get_lr_call(self):
            for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
                param_group, lr = data
                param_group['lr'] = lr
                self.print_lr(self.verbose, i, lr, epoch)
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


def get_k_fold_data(features, labels, adj, k=10):
    """Generate K-fold cross-validation data"""
    num_samples = labels.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    kf = KFold(n_splits=k)
    k_fold_data = []
    for train_idx, test_idx in kf.split(indices):
        train_idx = torch.LongTensor(train_idx)
        test_idx = torch.LongTensor(test_idx)
        k_fold_data.append((train_idx, test_idx))
    return k_fold_data
