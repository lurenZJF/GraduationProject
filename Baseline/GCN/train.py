#!/usr/bin/python
# -*- encoding:utf-8 -*-
import sys
import time
import argparse	 # 命令行解析的标准模块，可以让我们直接在命令行中就可以向程序中传入参数并让程序运行
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import warnings
warnings.filterwarnings("ignore")
sys.path.append('/home/penghao/Project/')
from Baseline.GCN.utils import load_data, score
from Baseline.GCN.models import GCN


# 训练设置
parser = argparse.ArgumentParser()  # 创建一个ArgumentParser对象
# 通过调用add_argument()来给一个ArgumentParser添加程序参数信息
"""
第一个参数 - 选项字符串，用于作为标识
action - 当参数在命令行中出现时使用的动作基本类型
default - 当参数未在命令行中出现时使用的值
type - 命令行参数应当被转换成的类型
help - 一个此选项作用的简单描述
"""
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')  # 随机种子
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')  # 权重衰减（参数L2损失）
parser.add_argument('--hidden', type=int, default=32, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
args = parser.parse_args()  # ArgumentParser通过parse_args()方法解析参数,将参数进行关联
# 这个是在确认是否使用gpu的参数，作为是否使用cpu的判定
args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)  # 产生随机种子，以使得结果是确定的
torch.manual_seed(args.seed)  # 为CPU设置随机种子用于生成随机数，以使得结果是确定的
if args.cuda:
    torch.cuda.manual_seed(args.seed)  # 为GPU设置随机种子
# 此处修改至关重要
adj, features, labels, idx_train, idx_val, idx_test = load_data()


def nll_loss(preds, y):
    """
    多分类损失函数
    :param preds:
    :param y:
    :return:
    """
    # 修改这里有无问题呢
    y = y.max(1)[1]
    return F.nll_loss(preds, y)


# 模型和优化器
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.shape[1],
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
if args.cuda:  # 如果使用GUP则执行这里，数据写入cuda，便于后续加速
    model.cuda()  # 模型放到GPU上跑
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch):
    """
    定义训练函数
    :param epoch: 训练轮数
    """
    t = time.time()
    model.train()
    optimizer.zero_grad()  # 梯度置零，把loss关于weight的导数变成0
    output = model(features, adj)  # 前向传播
    loss_train = nll_loss(output[idx_train], labels[idx_train])  # 最大似然/log似然损失函数
    train_acc, train_precision, train_recall, train_F1 = score(output[idx_train], labels[idx_train])  # 准确率
    loss_train.backward()  # 反向传播
    optimizer.step()  # 梯度下降，更新值
    if not args.fastmode:  # 是否在训练期间进行验证
        # 单独评估验证集的性能，在验证运行期间停用dropout
        # 因为nn.functional不像nn模块，在验证运行时不会自动关闭dropout，需要我们自行设置。
        model.eval()
        output = model(features, adj)
    loss_val = nll_loss(output[idx_val], labels[idx_val])
    val_acc, val_precision, val_recall, val_F1 = score(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(train_acc.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(val_acc.item()),
          'time: {:.4f}s'.format(time.time() - t))

# 先将model置为训练状态；梯度清零；将输入送到模型得到输出结果；计算损失与准确率；反向传播求梯度更新参数。


def test():
    """
    测试函数，将训练得到的模型在测试集上运行
    """
    model.eval()  # 不启用BatchNormalization和Dropout
    output = model(features, adj)
    loss_test = nll_loss(output[idx_test], labels[idx_test])
    test_acc, test_precision, test_recall, test_F1 = score(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(test_acc.item()),
          "F1= {:.4f}".format(test_F1.item()),)


t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")	 # 优化完成！
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))  # 已用总时间
# weights = model.W.detach().cpu().numpy().tolist()
# with open("../data/GCNdata/weight.txt", 'w', encoding='utf-8-sig') as f:
#     for w in weights:
#         f.write(str(w) + '\n')
test()