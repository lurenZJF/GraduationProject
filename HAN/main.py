import torch
from utils import load_data, EarlyStopping
import warnings
from model import HAN
import pandas as pd
import time


def evaluate(model, g, features, labels, mask, loss_func):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
    loss = loss_func(logits[mask], labels[mask])
    return loss

def main(args):
    warnings.filterwarnings("ignore")
    # 加载数据集
    g, features, labels, train_idx, test_idx, train_mask, test_mask, listprice= load_data(path='../TorontoM2/')
    
    print('start HAN', flush=True)
    # 将模型复制到GPU
    model = HAN(num_meta_paths=len(g),
                house_nums=features.shape[0],
                in_size=features.shape[1],
                hidden_size=args['hidden_units'],
                out_size=1,
                num_heads=args['num_heads'],
                dropout=args['dropout']).to(args['device'])
    # 多GPU机制
    # if torch.cuda.device_count() > 1: #判断是不是有多个GPU
    #     print('use ',torch.cuda.device_count(),'GPUs!')
    #     model = torch.nn.DataParallel(model,device_ids = range(torch.cuda.device_count()))
    # model = model.to(args['device'])
    features = features.to(args['device'])
    labels = labels.to(args['device'])
    listprice = listprice.to(args['device'])
    train_mask = train_mask.to(args['device'])
    test_mask = test_mask.to(args['device'])
    stopper = EarlyStopping(patience=args['patience'])  # 早停机制
    loss_fcn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    print('start training!', flush=True)
    # start training
    val_loss_list = []
    t_total = time.time()
    for epoch in range(args['num_epochs']):
        t = time.time()
        model.train()
        logits = model(g, features)  # 前向计算
        loss = loss_fcn(logits[train_mask], labels[train_mask])  #计算损失
        optimizer.zero_grad()  
        loss.backward()  # 计算梯度
        optimizer.step()  # 反向传播
        val_loss = evaluate(model, g, features, labels, test_mask, loss_fcn)
        val_loss_list.append(val_loss.item())
        early_stop = stopper.step(val_loss.data.item(), model)
        print('Epoch {:d} | Train Loss {:.4f}| Val Loss {:.4f}|time {:.4f}'.format(epoch + 1, loss.item(), val_loss.item(),time.time()-t), flush=True)
        if early_stop:
            break
    val_data = pd.DataFrame(val_loss_list)
    # start test
    stopper.load_checkpoint(model)  # 加载模型
    model.eval()
    with torch.no_grad():
        test_logits = model(g, features)
    test_loss = loss_fcn(test_logits[test_mask], labels[test_mask])
    print('Test loss {:.4f} '.format(test_loss.item()), flush=True)
    result = []
    for i in test_idx:
        i = i.item()
        result.append([i, listprice[i].item(), test_logits[i].item(), labels[i].item(),
                       abs(test_logits[i].item() - listprice[i].item()),
                       abs(listprice[i].item() - labels[i].item())])
    data = pd.DataFrame(result)
    val_data = pd.DataFrame(val_data)
    data.columns = ['index', 'list', 'predict', 'sold', 'sold-predict', 'list-price']
    csv_path = args['log_dir']+'/predict.csv'
    val_path = args['log_dir']+'/val_loss.csv'
    data.to_csv(csv_path, index=False)
    val_data.to_csv(val_path, index=False)
    print('total time {:.4f}'.format(time.time()-t_total), flush=True)


if __name__ == '__main__':
    import argparse
    from utils import setup
    parser = argparse.ArgumentParser('HAN')
    parser.add_argument('-s', '--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('-ld', '--log-dir', type=str, default='results',
                        help='Dir for saving training results')
    parser.add_argument('--hetero', action='store_true',
                        help='Use metapath coalescing with DGL\'s own dataset')
    args = parser.parse_args().__dict__
    args = setup(args)
    print('参数配置', flush=True)
    print(args, flush=True)
    main(args)