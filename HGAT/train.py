import sys
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import time
from copy import deepcopy
import torch.optim as optim
import warnings
warnings.filterwarnings("ignore")
sys.path.append('/home/penghao/Project/')
from  HGAT.utils import *
from HGAT.print_log import Logger
from HGAT.models import HGAT
import argparse
import pickle as pkl

# 模型输出保存路径
logdir = "log/"
savedir = 'model/'
embdir = 'embeddings/'
# 如果不存在上述文件夹，则进行创建
makedirs([logdir, savedir, embdir])

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
write_embeddings = True
HOP = 2

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=512, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default="HGAT", help='Dataset')
parser.add_argument('--repeat', type=int, default=1, help='Number of repeated trials')
parser.add_argument('--node', action='store_false', default=True, help='Use node-level attention or not. ')
parser.add_argument('--type', action='store_false', default=True, help='Use type-level attention or not. ')
args = parser.parse_args()
dataset = args.dataset
# 判断GPU是否可用
args.cuda = not args.no_cuda and torch.cuda.is_available()
sys.stdout = Logger(logdir+"{}.log".format(dataset))
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

loss_list = dict()


def evaluate(preds_list, y_list):
    """
    评估模型效果
    :param preds_list: 预测标签
    :param y_list: 真实标签
    :return: 准确率、F1值
    """
    nclass = y_list.shape[1]
    preds_list = preds_list[:, :nclass]
    if not preds_list.device == 'cpu':
        preds_list, y_list = preds_list.cpu(), y_list.cpu()
    y_list = y_list.numpy()
    preds_probs = preds_list.detach().numpy()
    preds = deepcopy(preds_probs)
    preds[np.arange(preds.shape[0]), preds.argmax(1)] = 1.0
    preds[np.where(preds < 1)] = 0.0
    print(preds)
    print(y_list)
    exit()
    [precision, recall, F1, support] = precision_recall_fscore_support(y_list, preds, average='macro')
    ER = accuracy_score(y_list, preds) * 100
    print(' Ac: %6.2f' % ER,
          'P: %5.1f' % (precision*100),
          'R: %5.1f' % (recall*100),
          'F1: %5.1f' % (F1*100),
          end="")
    return ER, F1


def train(epoch, input_adj_train, input_features_train,
          idx_out_train, idx_train, input_adj_val, input_features_val,
          idx_out_val, idx_val):
    """
    训练模型
    """
    print('Epoch: {:04d}'.format(epoch+1), end='')
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(input_features_train, input_adj_train)
    if isinstance(output, list):
        O, L = output[0][idx_out_train], labels[idx_train]
    else:
        O, L = output[idx_out_train], labels[idx_train]
    loss_train = nll_loss(O, L)
    print(' | loss: {:.4f}'.format(loss_train.item()), end='')
    acc_train, f1_train = evaluate(O, L)
    loss_train.backward()
    optimizer.step()
    model.eval()  # 停止训练
    output = model(input_features_val, input_adj_val)
    # print("经过softmax后的标签输出：", output[0].shape)
    if isinstance(output, list):
        loss_val = nll_loss(output[0][idx_out_val], labels[idx_val])
        print(' | loss: {:.4f}'.format(loss_val.item()), end='')
        results = evaluate(output[0][idx_out_val], labels[idx_val])
    else:
        loss_val = nll_loss(output[idx_out_val], labels[idx_val])
        print(' | loss: {:.4f}'.format(loss_val.item()), end='')
        results = evaluate(output[idx_out_val], labels[idx_val])
    print(' | time: {:.4f}s'.format(time.time() - t))
    loss_list[epoch] = [loss_train.item()]

    acc_val, f1_val = results
    return float(acc_val.item()), float(f1_val.item())


def test(epoch, input_adj_test, input_features_test, idx_out_test, idx_test):
    """
    测试模型
    """
    print(' '*65, end='')
    t = time.time()
    model.eval()  # 停止训练模型
    output = model(input_features_test, input_adj_test)
    if isinstance(output, list):
        loss_test = nll_loss(output[0][idx_out_test], labels[idx_test])
        print(' | loss: {:.4f}'.format(loss_test.item()), end='')
        results = evaluate(output[0][idx_out_test], labels[idx_test])
    else:
        loss_test = nll_loss(output[idx_out_test], labels[idx_test])
        print(' | loss: {:.4f}'.format(loss_test.item()), end='')
        results = evaluate(output[idx_out_test], labels[idx_test])
    print(' | time: {:.4f}s'.format(time.time() - t))
    loss_list[epoch] += [loss_test.item()]
    acc_test, f1_test = results
    return float(acc_test.item()), float(f1_test.item())


if __name__ == "__main__":
    # 加载数据
    path = "../Dataset/HGAT_train_data/out/"
    adj, features, labels, idx_train_ori, idx_val_ori, idx_test_ori, idx_map = load_data(path=path)
    adj = np.array(adj)
    features = np.array(features)
    # print(adj.shape)
    # print(features.shape)
    # exit()
    N = len(adj)
    # print("图数量:", str(N))
    input_adj_train, input_features_train, idx_out_train = adj, features, idx_train_ori
    input_adj_val, input_features_val, idx_out_val = adj, features, idx_val_ori
    input_adj_test, input_features_test, idx_out_test = adj, features, idx_test_ori
    idx_train, idx_val, idx_test = idx_train_ori, idx_val_ori, idx_test_ori
    # 将数据迁移到GPU上
    if args.cuda:
        N = len(features)
        for i in range(N):
            if input_features_train[i] is not None:
                input_features_train[i] = input_features_train[i].cuda()
            if input_features_val[i] is not None:
                input_features_val[i] = input_features_val[i].cuda()
            if input_features_test[i] is not None:
                input_features_test[i] = input_features_test[i].cuda()
        for i in range(N):
            for j in range(N):
                if input_adj_train[i][j] is not None:
                    input_adj_train[i][j] = input_adj_train[i][j].cuda()
                if input_adj_val[i][j] is not None:
                    input_adj_val[i][j] = input_adj_val[i][j].cuda()
                if input_adj_test[i][j] is not None:
                    input_adj_test[i][j] = input_adj_test[i][j].cuda()
        labels = labels.cuda()
        idx_train, idx_out_train = idx_train.cuda(), idx_out_train.cuda()
        idx_val, idx_out_val = idx_val.cuda(), idx_out_val.cuda()
        idx_test, idx_out_test = idx_test.cuda(), idx_out_test.cuda()
    FINAL_RESULT = []
    for i in range(args.repeat):
        # HGAT and optimizer
        print("\n\nNo. {} test.\n".format(i + 1))
        model = HGAT(nfeat_list=[i.shape[1] for i in features],
                     type_attention=args.type, node_attention=args.node, nhid=args.hidden,
                     nclass=labels.shape[1], dropout=args.dropout, gamma=0.1, orphan=True)

        print(len(list(model.parameters())))
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        if args.cuda:
            model.cuda()
        print([i.size() for i in model.parameters()])
        # Train model
        t_total = time.time()
        vali_max = [0, [0, 0], -1]

        for epoch in range(args.epochs):
            vali_acc, vali_f1 = train(epoch, input_adj_train, input_features_train, idx_out_train, idx_train,
                                      input_adj_val, input_features_val, idx_out_val, idx_val)
            test_acc, test_f1 = test(epoch, input_adj_test, input_features_test, idx_out_test, idx_test)
            if vali_acc > vali_max[0]:
                vali_max = [vali_acc, (test_acc, test_f1), epoch + 1]
                with open(savedir + "{}.pkl".format(dataset), 'wb') as f:
                    pkl.dump(model, f)

                if write_embeddings:
                    makedirs([embdir])
                    with open(embdir + "{}.emb".format(dataset), 'w') as f:
                        for i in model.emb.tolist():
                            f.write("{}\n".format(i))
                    with open(embdir + "{}.emb2".format(dataset), 'w') as f:
                        for i in model.emb2.tolist():
                            f.write("{}\n".format(i))

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        print("The best result is: ACC: {0:.4f} F1: {1:.4f}, where epoch is {2}\n\n".format(
            vali_max[1][0],
            vali_max[1][1],
            vali_max[2]))
        FINAL_RESULT.append(list(vali_max))

    print("\n")
    for i in range(len(FINAL_RESULT)):
        print("{}:\tvali:  {:.5f}\ttest:  ACC: {:.4f} F1: {:.4f}, epoch={}".format(
            i,
            FINAL_RESULT[i][0],
            FINAL_RESULT[i][1][0],
            FINAL_RESULT[i][1][1],
            FINAL_RESULT[i][2]))

