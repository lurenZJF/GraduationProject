import argparse
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from copy import deepcopy
import numpy as np
import time
import torch
import torch.nn.functional as F
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.append('/home/dell/GraduationProject/')
from Baseline.GAT.models import GAT
from Baseline.GAT.utils import setup, EarlyStopping
from Baseline.GAT.data import load_data, load_imdb


def score(logits, labels):
    labels = labels.cpu().numpy()
    preds_probs = logits.cpu().numpy()
    preds = deepcopy(preds_probs)
    preds[np.arange(preds.shape[0]), preds.argmax(1)] = 1.0
    preds[np.where(preds < 1)] = 0.0
    [precision, recall, F1, support] = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return acc, precision, recall, F1


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


def evaluate(model, features, labels, mask, loss_func):
    """
    评估模型效果，计算验证集上模型损失时使用；
    :param model:
    :param g:
    :param features:
    :param labels:
    :param mask:
    :param loss_func:
    :return:
    """
    model.eval()
    with torch.no_grad():
        logits = model(features)
    # print(logits, labels)
    loss = loss_func(logits[mask], labels[mask])
    acc, precision, recall, F1 = score(logits[mask], labels[mask])
    return loss, acc, precision, recall, F1


def main(args):
    # 加载数据
    # g, features, label, train_mask, val_mask, test_mask = load_data()
    g, features, label, train_mask, val_mask, test_mask = load_imdb()
    print("特征维度:", features.shape)
    heads = args["num_heads"] * args["num_layers"]
    # 将数据送到device
    features = features.to(args['device'])
    g = g.to(args['device'])
    labels = label.to(args['device'])
    train_mask = train_mask.to(args['device'])
    val_mask = val_mask.to(args['device'])
    test_mask = test_mask.to(args['device'])
    # 将模型进行复制到GPU
    model = GAT(g,
                args["num_layers"],
                features.shape[1],
                args['hidden_units'],
                labels.shape[1],
                heads,
                F.elu,
                args['dropout'],
                args['dropout'],
                args["negative_slope"],
                args["residual"]).to(args['device'])
    # 定义模型
    stopper = EarlyStopping(patience=args['patience'])  # 早停机制
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    # start training
    val_loss_list = []
    t_total = time.time()
    for epoch in range(args['num_epochs']):
        t = time.time()
        model.train()
        logits = model(features)  # 前向计算
        loss = nll_loss(logits[train_mask], labels[train_mask])  # 计算损失
        # train_acc, train_precision, train_recall, train_F1 = score(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()  # 计算梯度
        optimizer.step()  # 反向传播
        val_loss, val_acc, val_precision, val_recall, val_F1 = evaluate(model, features, labels, val_mask, nll_loss)
        val_loss_list.append(val_loss.item())
        early_stop = stopper.step(val_loss.data.item(), model)
        print('Epoch {:d} | Train Loss {:.4f} '
              '| Val Loss {:.4f} ACC {:.4f} P {:.4f} R {:.4f} F1 {:.4f} |time {:.4f}'.format(
            epoch + 1, loss.item(),
            val_loss.item(), val_acc, val_precision, val_recall, val_F1, time.time() - t), flush=True)
        if early_stop:
            break
    print('total time {:.4f}'.format(time.time() - t_total), flush=True)
    # 测试模型
    stopper.load_checkpoint(model)  # 加载模型
    test_loss, acc, precision, recall, F1 = evaluate(model, features, labels, test_mask, nll_loss)
    print("TEST loss:{:.4f} ACC:{:.4f} P {:.4f} R {:.4f} F1: {:.4f}".format(test_loss, acc, precision, recall, F1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('GAT')
    parser.add_argument('-s', '--seed', type=int, default=1, help='Random seed')
    args = parser.parse_args().__dict__
    args = setup(args)
    print('参数配置', flush=True)
    print(args, flush=True)
    heads = args["num_heads"] * args["num_layers"]
    # print(heads)
    # exit()
    main(args)

