from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import os
import glob
import time
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import f1_score

from utils import load_data, load_data_pubmed, accuracy, macro_f1, CosineAnnealingWarmRestartsWithStepScale
from model import HAT

import torch.distributed as dist
import torch.multiprocessing as mp



# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
# 不用余弦退火，恒定lr为0.0001
# parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
# 前500个epoch用lr=0.001，之后用lr=0.0001
# 前500个epoch用lr=0.001,501-1500用lr=0.0005,1501以后用lr=0.0001
# 尝试和余弦退火类似的方法，用四阶段，初始lr=0.01，50epoch降为0.005,150epoch降为0.001，1500epoch降为0.0001
# parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=20000, help='Patience')
args, unknown = parser.parse_known_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

model_nums = 3       # KAGAT, GAT, HAT
acc = 0              # 0: bestLoss, 1: bestAcc, 2: bestMacroF1
dataname = 'cora'    # cora, citeseer or pubmed
ratio = '205001000'  # 622 or 205001000
foldername = 'test'
max_test = 10
args.epochs = 500
nhid_kan = 4
nhid_gat = 4
nhid_hat = 4
wd_kan = args.weight_decay
wd_gat = args.weight_decay
wd_hat = args.weight_decay

step_scale = 0.65
# stop_restart = 1819
stop_restart = 99999999
total_kan_gat_loss = 0
total_kan_gat_accuracy = 0
total_kan_gat_macrof1 = 0
total_gat_loss = 0
total_gat_accuracy = 0
total_gat_macrof1 = 0
total_hat_loss = 0
total_hat_accuracy = 0
total_hat_macrof1 = 0
res_report = []


def train_model(model, optimizer, scheduler, features, adj, labels, idx_train, idx_val, idx_test, epochs, patience, res_report, test):
    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    features, adj, labels = Variable(features), Variable(adj), Variable(labels)

    acc_train_list = []
    acc_val_list = []
    loss_train_list = []
    loss_val_list = []
    macro_f1_train_list = []
    macro_f1_val_list = []
    lr_list = []
    best_acc_val = 0.0
    best_epoch = 0
    bad_counter = 0

    def train(epoch):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        # 要配合logsoftmax
        # loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        # 改nll_loss 为crossentropy，配合softmax
        criterion = nn.CrossEntropyLoss()
        loss_train = criterion(output[idx_train], labels[idx_train])
        loss_train.backward()
        acc_train = accuracy(output[idx_train], labels[idx_train])
        # 假设 y_true 是真实标签，y_pred 是模型预测的标签
        macro_f1_train = macro_f1(output[idx_train], labels[idx_train])
        optimizer.step()
        # 更新学习率
        # 尝试不用余弦退火，lr固定成0.0001
        # 0.0001太小了（但此时KAN效果也能更好，平均loss小，acc大），loss不好下降，5000个epoch时loss还不收敛。lr改用0.001
        # 0.001好像太大了，KAN的loss和acc都在1500、2000、3000、3400和4500明显抖动，而且前1000epoch中acc最大的区间似乎更窄，可能因为学习率太大反复跳出最优值
        # lr改用0.0005（效果和0.001一样，KAN的loss和acc不好，虽然曲线不再抖动）
        # lr改回0.0001（10次平均KAN比GAT的acc好0.3%左右，但还是loss不收敛）
        # 尝试前500个epoch用lr=0.001，之后用lr=0.0001（loss开始下降明显，但后来不太收敛，最好的acc在1500以后，而且部分test的KAN和GAT相差悬殊，平均下来KAN的acc稍好0.11%）
        # 为了解决不收敛问题，500-1500个epoch改lr=0.0005，1-500还是0.001，然后1500以后还是0.0001
        # KAN的acc略小于GAT（约0.03%），考虑1-1000用0.001，然后用0.0001（恢复成两阶段）
        # KAN比GAT的acc高约0.3%，但GAT的loss还是不收敛
        # 尝试和余弦退火类似的方法，用四阶段，初始lr=0.01，50epoch降为0.005,150epoch降为0.001，1500epoch降为0.0001
        # if epoch+1 == 50: #第50个epoch修改lr=0.005
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = 0.005
        # if epoch+1 == 150: #第150个epoch修改lr=0.001
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = 0.001     
        # if epoch+1 == 1500: #第1500个epoch修改lr=0.0001
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = 0.0001

        # 固定lr=0.001
        # for param_group in optimizer.param_groups:
        #         param_group['lr'] = 0.001
        # if epoch < stop_restart:
        #     scheduler.step()
        # else:
        #     pass
        # lr_list.append(scheduler.get_last_lr()[0])
        lr_list.append(optimizer.param_groups[0]['lr'])
        if not args.fastmode:
            model.eval()
            output = model(features, adj)

        # loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        criterion = nn.CrossEntropyLoss()
        loss_val = criterion(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        macro_f1_val = macro_f1(output[idx_val], labels[idx_val])

        loss_train_list.append(loss_train.data.item())
        loss_val_list.append(loss_val.data.item())
        acc_train_list.append(acc_train.data.item())
        acc_val_list.append(acc_val.data.item())
        macro_f1_train_list.append(macro_f1_train)
        macro_f1_val_list.append(macro_f1_val)

        # print('Epoch: {:04d}'.format(epoch + 1),
        #       'loss_train: {:.4f}'.format(loss_train.data.item()),
        #       'acc_train: {:.4f}'.format(acc_train.data.item()),
        #       'loss_val: {:.4f}'.format(loss_val.data.item()),
        #       'acc_val: {:.4f}'.format(acc_val.data.item()),
        #       'time: {:.4f}s'.format(time.time() - t))

        return loss_val.data.item()

    t_total = time.time()
    best_acc_val_epoch = 0
    temp_best_acc_val_epoch = 0
    temp_best_acc_val = 0
    temp_best_loss_val_epoch = 0
    temp_best_loss_val = 999999
    temp_best_macrof1_val_epoch = 0
    temp_best_macrof1_val = 0.0
    bad_counter = 0
    first_save = True
    last_val_loss = 100000
    model_name = type(model).__name__
    
    with tqdm(total=epochs, desc=f'{test+1}th test {model_name} Training Progress') as pbar:
        for epoch in range(epochs):
            # 余弦退火最小lr=0.00001不应该保存全局验证集loss最小的epoch（因为用了余弦退火，loss的V形曲线在谷底之后会产生loss更小的epoch，但这时候已经过拟合了）
            # 调整一下余弦退火（1800左右的重启（可能因为最小lr设成0.00001太小，重启一下又到了0.001）会导致KAN-GAT的loss和accuracy在1800和2500左右抖动，产生意外的最小loss）
            # 余弦退火压低最后一次重启的lr上限（到大约0.0005），下限0.0001，以平滑曲线
    
            # early-stopping策略（会错过loss最小的epoch）
            # this_val_loss = train(epoch)
            # if this_val_loss >= last_val_loss:
            #     bad_counter += 1
            # else:
            #     temp_best_loss_val_epoch = epoch
            #     temp_best_loss_val = this_val_loss
            #     files = glob.glob('*.pkl')
            #     for file in files:
            #         os.remove(file)
            #     torch.save(model.state_dict(), '{}.pkl'.format(temp_best_loss_val_epoch))
            # last_val_loss = this_val_loss
            
            global acc
            if acc == 0:
            # 保存全局验证集loss最小的epoch（KAN-GAT效果不太好（可能因为验证集上loss最小点位于accuracy最大点以后，存在过拟合？））
                train(epoch)
                if epoch > 0:
                    this_val_loss = loss_val_list[-1]
                    if this_val_loss >= temp_best_loss_val:
                        bad_counter += 1
                    else:
                        temp_best_loss_val_epoch = epoch+1
                        temp_best_loss_val = this_val_loss
                        files = glob.glob('*.pkl')
                        matching_files = [f for f in files if model_name in os.path.basename(f)]
                        for file in matching_files:
                            os.remove(file)
                        torch.save(model.state_dict(), f'{model_name}_{temp_best_loss_val_epoch}.pkl')
            elif acc == 1:
            # 保存全局验证集accuracy最大的epoch（偶尔会在很大的epoch，但通常在几百以内的epoch）
            # KAN-GAT的训练曲线不如GAT稳定
                train(epoch)
                if epoch > 0:
                    this_val_acc = acc_val_list[-1]
                    if this_val_acc <= temp_best_acc_val:
                        bad_counter += 1
                    else:
                        temp_best_acc_val_epoch = epoch+1
                        temp_best_acc_val = this_val_acc
                        files = glob.glob('*.pkl')
                        matching_files = [f for f in files if model_name in os.path.basename(f)]
                        for file in matching_files:
                            os.remove(file)
                        torch.save(model.state_dict(), f'{model_name}_{temp_best_acc_val_epoch}.pkl')
            else:
            # 保存全局验证集macro_f1最大的epoch
            # KAN-GAT的训练曲线不如GAT稳定
                train(epoch)
                if epoch > 0:
                    this_val_macrof1 = macro_f1_val_list[-1]
                    if this_val_macrof1 <= temp_best_macrof1_val:
                        bad_counter += 1
                    else:
                        temp_best_macrof1_val_epoch = epoch+1
                        temp_best_macrof1_val = this_val_macrof1
                        files = glob.glob('*.pkl')
                        matching_files = [f for f in files if model_name in os.path.basename(f)]
                        for file in matching_files:
                            os.remove(file)
                        torch.save(model.state_dict(), f'{model_name}_{temp_best_macrof1_val_epoch}.pkl')
            pbar.update(1)  # 更新进度条
                    
            if epoch+1 == epochs:
                print("Optimization Finished!")
                print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
                if acc == 1:
                    print('Loading {}th epoch'.format(temp_best_acc_val_epoch))
                    model.load_state_dict(torch.load(f'{model_name}_{temp_best_acc_val_epoch}.pkl'))
                elif acc == 0:
                    print('Loading {}th epoch'.format(temp_best_loss_val_epoch))
                    model.load_state_dict(torch.load(f'{model_name}_{temp_best_loss_val_epoch}.pkl'))
                else:
                    print('Loading {}th epoch'.format(temp_best_macrof1_val_epoch))
                    model.load_state_dict(torch.load(f'{model_name}_{temp_best_macrof1_val_epoch}.pkl'))
            
                model.eval()
                output = model(features, adj)
                # loss_test = F.nll_loss(output[idx_test], labels[idx_test])
                criterion = nn.CrossEntropyLoss()
                loss_test = criterion(output[idx_test], labels[idx_test])
                acc_test = accuracy(output[idx_test], labels[idx_test])
                macrof1_test = macro_f1(output[idx_test], labels[idx_test])
                print("Test set results:",
                      "loss= {:.4f}".format(loss_test.data.item()),
                      "accuracy= {:.4f}".format(acc_test.data.item()),
                      "macro f1= {:.4f}".format(macrof1_test))
                if model_name == 'KAN_GAT':
                    global total_kan_gat_loss
                    global total_kan_gat_accuracy
                    global total_kan_gat_macrof1
                    total_kan_gat_loss += loss_test.data.item()
                    total_kan_gat_accuracy += acc_test.data.item()
                    total_kan_gat_macrof1 += macrof1_test
                elif model_name == 'GAT':
                    global total_gat_loss
                    global total_gat_accuracy
                    global total_gat_macrof1
                    total_gat_loss += loss_test.data.item()
                    total_gat_accuracy += acc_test.data.item()
                    total_gat_macrof1 += macrof1_test
                elif model_name == 'HAT':
                    global total_hat_loss
                    global total_hat_accuracy
                    global total_hat_macrof1
                    total_hat_loss += loss_test.data.item()
                    total_hat_accuracy += acc_test.data.item()
                    total_hat_macrof1 += macrof1_test
                else:
                    pass
                files = glob.glob('*.pkl')
                matching_files = [f for f in files if model_name in os.path.basename(f)]
                for file in matching_files:
                    os.remove(file)
                if acc == 1:
                    res_report.append(f"Test set results {type(model).__name__} ({temp_best_acc_val_epoch}th epoch): loss= {loss_test.data.item():.4f} accuracy= {acc_test.data.item():.4f}, macro f1-score= {macrof1_test:.4f}")
                elif acc == 0:
                    res_report.append(f"Test set results {type(model).__name__} ({temp_best_loss_val_epoch}th epoch): loss= {loss_test.data.item():.4f} accuracy= {acc_test.data.item():.4f}, macro f1-score= {macrof1_test:.4f}")
                else:
                    res_report.append(f"Test set results {type(model).__name__} ({temp_best_macrof1_val_epoch}th epoch): loss= {loss_test.data.item():.4f} accuracy= {acc_test.data.item():.4f}, macro f1-score= {macrof1_test:.4f}")
    return acc_train_list, acc_val_list, loss_train_list, loss_val_list, macro_f1_train_list, macro_f1_val_list, best_acc_val_epoch, best_acc_val, lr_list



for test in range(max_test):
    # Load data
    # adj, features, labels, idx_train, idx_val, idx_test = load_data(f'{dataname}')
    if dataname != 'pubmed':
        adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset=dataname)
    else:
        adj, features, labels, idx_train, idx_val, idx_test = load_data_pubmed()
    global nhid_kan, nhid_gat, nhid_hat, wd_kan, wd_gat, wd_hat, lr_hat
    # Model and optimizer for KAN_GAT
    model_kan_gat = KAN_GAT(nfeat=features.shape[1],
                            nhid=nhid_kan,
                            nclass=int(labels.max()) + 1,
                            dropout=args.dropout, 
                            nheads=1,
                            n_nodes = features.shape[0],
                            alpha=args.alpha)
    
    optimizer_kan_gat = optim.AdamW(model_kan_gat.parameters(),
                                   lr=args.lr,
                                   weight_decay=wd_kan)
    scheduler_kan_gat = CosineAnnealingWarmRestartsWithStepScale(optimizer_kan_gat, T_0=5,T_mult=3,eta_min=0.0001,step_scale=step_scale)
    
    # Model and optimizer for GAT
    model_gat = GAT(nfeat=features.shape[1],
                    nhid=nhid_gat,
                    nclass=int(labels.max()) + 1,
                    dropout=args.dropout,
                    nheads=1,
                    n_nodes = features.shape[0],
                    alpha=args.alpha)
    
    optimizer_gat = optim.AdamW(model_gat.parameters(),
                               lr=args.lr,
                               weight_decay=wd_gat)
    scheduler_gat = CosineAnnealingWarmRestartsWithStepScale(optimizer_gat, T_0=5,T_mult=3,eta_min=0.0001,step_scale=step_scale)

    # Model and optimizer for HAT
    model_hat = HAT(nfeat=features.shape[1],
                    nhid=nhid_hat,
                    nclass=int(labels.max()) + 1,
                    dropout=args.dropout,
                    nheads=1,
                    n_nodes = features.shape[0],
                    alpha=args.alpha)
    
    optimizer_hat = optim.AdamW(model_hat.parameters(),
                               lr=args.lr,
                               weight_decay=wd_hat)
                                # 

    scheduler_hat = CosineAnnealingWarmRestartsWithStepScale(optimizer_hat, T_0=5,T_mult=3,eta_min=0.0001,step_scale=step_scale)
    
    # Train all models
    acc_train_kan_gat, acc_val_kan_gat, loss_train_kan_gat, loss_val_kan_gat, macrof1_train_kan_gat, macrof1_val_kan_gat, best_acc_val_epoch_kan_gat, best_acc_val_kan_gat, lr_list_kan = train_model(
        model_kan_gat, optimizer_kan_gat, scheduler_kan_gat, features, adj, labels, idx_train, idx_val, idx_test, args.epochs, args.patience, res_report, test)
    
    acc_train_gat, acc_val_gat, loss_train_gat, loss_val_gat, macrof1_train_gat, macrof1_val_gat, best_acc_val_epoch_gat, best_acc_val_gat, lr_list_gat = train_model(
        model_gat, optimizer_gat, scheduler_gat, features, adj, labels, idx_train, idx_val, idx_test, args.epochs, args.patience, res_report, test)

    acc_train_hat, acc_val_hat, loss_train_hat, loss_val_hat, macrof1_train_hat, macrof1_val_hat, best_acc_val_epoch_hat, best_acc_val_hat, lr_list_hat = train_model(
        model_hat, optimizer_hat, scheduler_hat, features, adj, labels, idx_train, idx_val, idx_test, args.epochs, args.patience, res_report, test)

    # 前500个epoch用大学习率0.001，500个epoch后用小学习率0.0001（lr1=0.001lr2=0.0001）
    # 前500个epoch用lr=0.001,501-1500用lr=0.0005,1501以后用lr=0.0001（lr3stage）
    # 前1000个epoch用lr=0.001，1001以后用lr=0.0001（lr2stage1000）
    # 尝试和余弦退火类似的方法，用四阶段，初始lr=0.01，50epoch降为0.005,150epoch降为0.001，1500epoch降为0.0001（lr4stage）
    global foldername
    result_folder_name = foldername
    if not os.path.exists(result_folder_name):
        os.makedirs(result_folder_name)
    
    res_report_path = os.path.join(result_folder_name, '00'+f'{result_folder_name}.txt')
    with open(res_report_path, 'a', encoding='utf-8') as file:
        seq = ''
        if (test+1)%10 == 1 and (test+1)%100 != 11:
            seq = 'st'
        elif (test+1)%10 == 2 and (test+1)%100 != 12:
            seq = 'nd'
        elif (test+1)%10 == 3 and (test+1)%100 != 13:
            seq = 'rd'
        else:
            seq = 'th'
        file.write(f'{test+1}{seq} test:' + '\n')
        global model_nums
        for res in res_report[test*model_nums:]:
            print(res)
            file.write(res + '\n')# 将每一条记录写入文件并换行
        file.write('\n')
    
    
    # Plot accuracy and loss curves for both models
    
    # Accuracy plots
    plt.plot(acc_train_kan_gat, label='KAN_GAT Training Accuracy')
    plt.plot(acc_val_kan_gat, label='KAN_GAT Validation Accuracy')
    plt.plot(acc_train_gat, label='GAT Training Accuracy')
    plt.plot(acc_val_gat, label='GAT Validation Accuracy')
    plt.plot(acc_train_hat, label='HAT Training Accuracy')
    plt.plot(acc_val_hat, label='HAT Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    best_acc_val_epoch_kan_gat = np.argmax(acc_val_kan_gat)+1
    best_acc_val_kan_gat = acc_val_kan_gat[best_acc_val_epoch_kan_gat-1]
    plt.scatter(best_acc_val_epoch_kan_gat, best_acc_val_kan_gat, color='red')
    plt.text(best_acc_val_epoch_kan_gat, best_acc_val_kan_gat, f'Best KAN_GAT Epoch: {best_acc_val_epoch_kan_gat}\nAcc: {best_acc_val_kan_gat:.4f}',
             horizontalalignment='left', verticalalignment='bottom')
    
    best_acc_val_epoch_gat = np.argmax(acc_val_gat)+1
    best_acc_val_gat = acc_val_gat[best_acc_val_epoch_gat-1]
    plt.scatter(best_acc_val_epoch_gat, best_acc_val_gat, color='blue')
    plt.text(best_acc_val_epoch_gat, best_acc_val_gat, f'Best GAT Epoch: {best_acc_val_epoch_gat}\nAcc: {best_acc_val_gat:.4f}',
             horizontalalignment='left', verticalalignment='bottom')

    best_acc_val_epoch_hat = np.argmax(acc_val_hat)+1
    best_acc_val_hat = acc_val_hat[best_acc_val_epoch_hat-1]
    plt.scatter(best_acc_val_epoch_hat, best_acc_val_hat, color='blue')
    plt.text(best_acc_val_epoch_hat, best_acc_val_hat, f'Best HAT Epoch: {best_acc_val_epoch_hat}\nAcc: {best_acc_val_hat:.4f}',
             horizontalalignment='left', verticalalignment='bottom')
    
    plt.title('Training and Validation Accuracy')
    picname = f'{test+1}th test accuracy.png'
    pic_path = os.path.join(result_folder_name, picname)
    plt.savefig(pic_path)  # 保存图像到文件
    plt.show()
    
    # Loss plots
    plt.plot(loss_train_kan_gat, label='KAN_GAT Training Loss')
    plt.plot(loss_val_kan_gat, label='KAN_GAT Validation Loss')
    plt.plot(loss_train_gat, label='GAT Training Loss')
    plt.plot(loss_val_gat, label='GAT Validation Loss')
    plt.plot(loss_train_hat, label='HAT Training Loss')
    plt.plot(loss_val_hat, label='HAT Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    best_loss_val_epoch_kan_gat = np.argmin(loss_val_kan_gat)+1
    best_loss_val_kan_gat = loss_val_kan_gat[best_loss_val_epoch_kan_gat-1]
    plt.scatter(best_loss_val_epoch_kan_gat, best_loss_val_kan_gat, color='red')
    plt.text(best_loss_val_epoch_kan_gat, best_loss_val_kan_gat, f'Best KAN_GAT Epoch: {best_loss_val_epoch_kan_gat}\nLoss: {best_loss_val_kan_gat:.4f}',
             horizontalalignment='left', verticalalignment='bottom')
    
    best_loss_val_epoch_gat = np.argmin(loss_val_gat)+1
    best_loss_val_gat = loss_val_gat[best_loss_val_epoch_gat-1]
    plt.scatter(best_loss_val_epoch_gat, best_loss_val_gat, color='blue')
    plt.text(best_loss_val_epoch_gat, best_loss_val_gat, f'Best GAT Epoch: {best_loss_val_epoch_gat}\nLoss: {best_loss_val_gat:.4f}',
             horizontalalignment='left', verticalalignment='bottom')

    best_loss_val_epoch_hat = np.argmin(loss_val_hat)+1
    best_loss_val_hat = loss_val_hat[best_loss_val_epoch_hat-1]
    plt.scatter(best_loss_val_epoch_hat, best_loss_val_hat, color='blue')
    plt.text(best_loss_val_epoch_hat, best_loss_val_hat, f'Best HAT Epoch: {best_loss_val_epoch_hat}\nLoss: {best_loss_val_hat:.4f}',
             horizontalalignment='left', verticalalignment='bottom')
    
    plt.title('Training and Validation Loss')
    picname = f'{test+1}th test loss.png'
    pic_path = os.path.join(result_folder_name, picname)
    plt.savefig(pic_path)  # 保存图像到文件
    plt.show()

    # Macro F1-score plots
    plt.plot(macrof1_train_kan_gat, label='KAN_GAT Training Macro F1-score')
    plt.plot(macrof1_val_kan_gat, label='KAN_GAT Validation Macro F1-score')
    plt.plot(macrof1_train_gat, label='GAT Training Macro F1-score')
    plt.plot(macrof1_val_gat, label='GAT Validation Macro F1-score')
    plt.plot(macrof1_train_hat, label='HAT Training Macro F1-score')
    plt.plot(macrof1_val_hat, label='HAT Validation Macro F1-score')
    plt.xlabel('Epochs')
    plt.ylabel('Macro F1-score')
    plt.legend()
    best_macrof1_val_epoch_kan_gat = np.argmax(macrof1_val_kan_gat)+1
    best_macrof1_val_kan_gat = macrof1_val_kan_gat[best_macrof1_val_epoch_kan_gat-1]
    plt.scatter(best_macrof1_val_epoch_kan_gat, best_macrof1_val_kan_gat, color='red')
    plt.text(best_macrof1_val_epoch_kan_gat, best_macrof1_val_kan_gat, f'Best KAN_GAT Epoch: {best_macrof1_val_epoch_kan_gat}\nMacro F1-score: {best_macrof1_val_kan_gat:.4f}',
             horizontalalignment='left', verticalalignment='bottom')
    
    best_macrof1_val_epoch_gat = np.argmax(macrof1_val_gat)+1
    best_macrof1_val_gat = macrof1_val_gat[best_macrof1_val_epoch_gat-1]
    plt.scatter(best_macrof1_val_epoch_gat, best_macrof1_val_gat, color='blue')
    plt.text(best_macrof1_val_epoch_gat, best_macrof1_val_gat, f'Best GAT Epoch: {best_macrof1_val_epoch_gat}\nMacro F1-score: {best_macrof1_val_gat:.4f}',
             horizontalalignment='left', verticalalignment='bottom')

    best_macrof1_val_epoch_hat = np.argmax(macrof1_val_hat)+1
    best_macrof1_val_hat = macrof1_val_hat[best_macrof1_val_epoch_hat-1]
    plt.scatter(best_macrof1_val_epoch_hat, best_macrof1_val_hat, color='blue')
    plt.text(best_macrof1_val_epoch_hat, best_macrof1_val_hat, f'Best HAT Epoch: {best_macrof1_val_epoch_hat}\nMacro F1-score: {best_macrof1_val_hat:.4f}',
             horizontalalignment='left', verticalalignment='bottom')
    
    plt.title('Training and Validation Macro F1-score')
    picname = f'{test+1}th test macrof1.png'
    pic_path = os.path.join(result_folder_name, picname)
    plt.savefig(pic_path)  # 保存图像到文件
    plt.show()

    if test+1 == max_test:
        pass
        # 学习率曲线只保存一次就够了
        # plt.plot(lr_list_kan, label='KAN and GAT Learning Rate')
        # plt.xlabel('Epochs')
        # plt.ylabel('Learning Rate')
        # plt.legend()
        # picname = f'learning_rate.png'
        # pic_path = os.path.join(result_folder_name, picname)
        # plt.savefig(pic_path)  # 保存图像到文件
        # plt.show()


print(f'{max_test} tests KAN_GAT: ')
print(f'average loss= {total_kan_gat_loss/max_test}')
print(f'average accuracy= {total_kan_gat_accuracy/max_test}')
print(f'average macro f1-score= {total_kan_gat_macrof1/max_test}' + '\n')
print(f'{max_test} tests GAT: ')
print(f'average loss= {total_gat_loss/max_test}')
print(f'average accuracy= {total_gat_accuracy/max_test}')
print(f'average macro f1-score= {total_gat_macrof1/max_test}'+'\n')
print(f'{max_test} tests HAT: ')
print(f'average loss= {total_hat_loss/max_test}')
print(f'average accuracy= {total_hat_accuracy/max_test}')
print(f'average macro f1-score= {total_hat_macrof1/max_test}'+'\n')

res_report_path = os.path.join(result_folder_name, '00'+f'{result_folder_name}.txt')
with open(res_report_path, 'a', encoding='utf-8') as file:
    file.write(f'{max_test} tests KAN_GAT: ' + '\n')
    file.write(f'average loss= {total_kan_gat_loss/max_test}' + '\n')
    file.write(f'average accuracy= {total_kan_gat_accuracy/max_test}' + '\n')
    file.write(f'average macro f1-score= {total_kan_gat_macrof1/max_test}' + '\n\n')
    file.write(f'{max_test} tests GAT: ' + '\n')
    file.write(f'average loss= {total_gat_loss/max_test}' + '\n')
    file.write(f'average accuracy= {total_gat_accuracy/max_test}' + '\n')
    file.write(f'average macro f1-score= {total_gat_macrof1/max_test}' + '\n\n')
    file.write(f'{max_test} tests HAT: ' + '\n')
    file.write(f'average loss= {total_hat_loss/max_test}' + '\n')
    file.write(f'average accuracy= {total_hat_accuracy/max_test}' + '\n')
    file.write(f'average macro f1-score= {total_hat_macrof1/max_test}' + '\n\n')
