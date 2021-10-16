import datetime
import dgl
import errno
import numpy as np
import os
import random
import torch
from scipy import sparse


def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    try:
        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise

def get_date_postfix():
    """Get a date based postfix for directory name.
    Returns
    -------
    post_fix : str
    """
    dt = datetime.datetime.now()
    post_fix = '{}_{:02d}-{:02d}-{:02d}'.format(
        dt.date(), dt.hour, dt.minute, dt.second)

    return post_fix

def setup_log_dir(args, sampling=False):
    """Name and create directory for logging.
    Parameters
    ----------
    args : dict
        Configuration
    Returns
    -------
    log_dir : str
        Path for logging directory
    sampling : bool
        Whether we are using sampling based training
    """
    date_postfix = get_date_postfix()
    log_dir = os.path.join(
        args['log_dir'],
        '{}_{}'.format('toronto', date_postfix))

    if sampling:
        log_dir = log_dir + '_sampling'

    mkdir_p(log_dir)
    return log_dir


# The configuration below is from the paper.
default_configure = {
    'lr': 0.005,             # Learning rate
    'num_heads': [8],        # Number of attention heads for node-level attention
    'hidden_units': 12,
    'dropout': 0.6,
    'weight_decay': 0.0005,
    'num_epochs': 1000,
    'patience': 800
}

sampling_configure = {
    'batch_size': 5000
}


def setup(args):
    args.update(default_configure)
    set_random_seed(args['seed'])
    args['device'] = 'cuda: 0' if torch.cuda.is_available() else 'cpu'
    args['log_dir'] = setup_log_dir(args)
    return args


def setup_for_sampling(args):
    args.update(default_configure)  # 将学习率等配置添加进args
    args.update(sampling_configure)  # 每批学习的数据数量
    set_random_seed()
    args['device'] = 'cuda: 0' if torch.cuda.is_available() else 'cpu'
    args['log_dir'] = setup_log_dir(args, sampling=True)  # 文件保存地址
    return args

# mask机制，暂不需要修改
def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.bool()  # 在高性能平台上用这个


def load_data(path):
    print('Loading data...', flush=True)
    idx_features_labels = np.genfromtxt("{}n_feature.txt".format(path), dtype=np.float32)
    # 根据读取的特征文件，生成features,listprice,labels
    feature_size = idx_features_labels.shape[1]  # 特征维度
    data_size = idx_features_labels.shape[0]  # 所有月的总房屋个数
    features = idx_features_labels[:, 0: -1]  # 特征，去掉挂牌价和成交价
    listprice = idx_features_labels[:, -2]  # 挂牌价（训练过程中暂时不用）
    labels = idx_features_labels[:, -1]  # 成交价
    listprice = listprice[:, np.newaxis]
    labels = labels[:, np.newaxis]  # 列向量转列向量矩阵

    # listprice的最后一维添加索引
    index = np.array(range(0, data_size)).T
    index = index[:, np.newaxis]
    listprice = np.hstack((listprice, index))
    print('feature size: ' + str(features.shape))
    print('label size: ' + str(labels.shape))
    print('listprice size: ' + str(listprice.shape))
    # 转化为tensor形式
    features = torch.FloatTensor(features)
    listprice = torch.FloatTensor(listprice)
    labels = torch.FloatTensor(labels)
    # Adjacency matrices for meta path based neighbors
    # 从npz文件中读取数据
    edges_C = sparse.load_npz("{}H_community_H.npz".format(path))
    edges_P = sparse.load_npz("{}H_postal_H.npz".format(path))
    edges_garge_Community = sparse.load_npz("{}H_Garge_C_H.npz".format(path))
    edges_fsa = sparse.load_npz("{}H_fsa_H.npz".format(path))
    edges_M = sparse.load_npz("{}H_municipality_H.npz".format(path))
    # 传入DGL 构造异构图
    C_g = dgl.graph(edges_C, ntype='House', etype='Community')
    P_g = dgl.graph(edges_P, ntype='House', etype='postal')
    garge_Community_g = dgl.graph(edges_garge_Community, ntype='House', etype='garge')
    F_g = dgl.graph(edges_fsa, ntype='House', etype='fsa')
    M_g = dgl.graph(edges_M, ntype='House', etype='fsa')
    # 根据邻接矩阵构造异构图
    gs = [C_g, F_g, garge_Community_g, P_g, M_g]
    print('graph num:'+str(len(gs)), flush=True)
    # 制作训练集和测试集的索引
    train_index = range(25000)
    test_index = range(25000, 31000)
    train_index = np.array(train_index)
    test_index = np.array(test_index)
    num_nodes = C_g.number_of_nodes()  # 节点数量
    # 转化为tensor形式
    train_idx = torch.from_numpy(train_index).long()
    test_idx = torch.from_numpy(test_index).long()
    # mask操作
    train_mask = get_binary_mask(num_nodes, train_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)
    return gs, features, labels, train_idx, test_idx, train_mask, test_mask, listprice


class EarlyStopping(object):
    def __init__(self, patience=100):
        dt = datetime.datetime.now()
        self.filename = 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
            dt.date(), dt.hour, dt.minute, dt.second)
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, model):
        if self.best_loss is None:
            self.best_loss = loss
            self.save_checkpoint(model)
        elif loss > self.best_loss:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if loss <= self.best_loss:
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load(self.filename))