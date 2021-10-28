import torch
import datetime
import numpy as np
import random


class EarlyStopping(object):
    """
    早停机制
    """
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



# 训练参数设置
default_configure = {
    'lr': 0.005,             # Learning rate
    'num_heads': [8],        # Number of attention heads for node-level attention
    'hidden_units': 8,
    'dropout': 0.5,
    'weight_decay': 0.0005,
    'num_epochs': 1000,
    'patience': 800,
    "num_layers": 2,
    "negative_slope": 0.2,
    "residual": False,
    "fastmode": False
}

sampling_configure = {
    'batch_size': 5000
}


def setup(args):
    """
    更新配置
    :param args:
    :return:
    """
    args.update(default_configure)
    set_random_seed(args['seed'])
    args['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return args


def setup_for_sampling(args):
    """
    :param args:
    :return:
    """
    args.update(default_configure)  # 将学习率等配置添加进args
    args.update(sampling_configure)  # 每批学习的数据数量
    set_random_seed()  # 设置随机数
    args['device'] = 'cuda: 0' if torch.cuda.is_available() else 'cpu'
    return args