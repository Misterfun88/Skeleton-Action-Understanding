
from dataset import get_finetune_training_set, get_finetune_validation_set
import argparse
import os
import random
import warnings

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.distributed
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

from umurl import Downstream
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# change for action recogniton


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=80, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=30., type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[50, 70, ], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

parser.add_argument('--pretrained', default='', type=str,
                    help='path to moco pretrained checkpoint')
parser.add_argument('--finetune-dataset', default='ntu60', type=str,
                    help='which dataset to use for finetuning')

parser.add_argument('--protocol', default='cross_view', type=str,
                    help='traiining protocol of ntu')


parser.add_argument('--knn-neighbours', default=1, type=int,
                    help='number of neighbours used for KNN.')

best_acc1 = 0

# initilize weight
def weights_init_gru(model):
    with torch.no_grad():
        for child in list(model.children()):
            print("init ", child)
            for param in list(child.parameters()):
                if param.dim() == 2:
                    nn.init.xavier_uniform_(param)
    print('PC weight initial finished!')


def load_pretrained(model, pretrained):
    if os.path.isfile(pretrained):
        print("=> loading checkpoint '{}'".format(pretrained))
        checkpoint = torch.load(pretrained, map_location="cpu")

        # rename pre-trained keys
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            if k.startswith('module.'):
                # remove prefix
                state_dict[k[len("module."):]] = state_dict[k]
            else:
                pass
            del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        print("message", msg)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

        print("=> loaded pre-trained model '{}'".format(pretrained))
    else:
        print("=> no checkpoint found at '{}'".format(pretrained))


def knn(data_train, data_test, label_train, label_test, nn=9):
    label_train = np.asarray(label_train)
    label_test = np.asarray(label_test)
    print("Number of KNN Neighbours = ", nn)
    print("training feature and labels", data_train.shape, len(label_train))
    print("test feature and labels", data_test.shape, len(label_test))

    Xtr_Norm = preprocessing.normalize(data_train)
    Xte_Norm = preprocessing.normalize(data_test)

    knn = KNeighborsClassifier(n_neighbors=nn,
                               metric='cosine',
                               n_jobs=24)  # , metric='cosine'#'mahalanobis', metric_params={'V': np.cov(data_train)})
    knn.fit(Xtr_Norm, label_train)
    pred = knn.predict(Xte_Norm)
    acc = accuracy_score(pred, label_test)

    return acc


def test_extract_hidden(model, data_train, data_eval):
    model.eval()
    print("Extracting training features")
    label_train_list = []
    hidden_array_train_list = []
    for ith, (jt, js, bt, bs, mt, ms, label) in enumerate(data_train):
        jt = jt.float().cuda(non_blocking=True)
        js = js.float().cuda(non_blocking=True)
        bt = bt.float().cuda(non_blocking=True)
        bs = bs.float().cuda(non_blocking=True)
        mt = mt.float().cuda(non_blocking=True)
        ms = ms.float().cuda(non_blocking=True)
        
        en_hi = model(jt, js, bt, bs, mt, ms, knn_eval=True)
         
        label_train_list.append(label)
        hidden_array_train_list.append(en_hi[:, :].detach().cpu().numpy())
        
    label_train = np.hstack(label_train_list)
    hidden_array_train = np.vstack(hidden_array_train_list)

    print("Extracting validation features")
    label_eval_list = []
    hidden_array_eval_list = []
    for ith, (jt, js, bt, bs, mt, ms, label) in enumerate(data_eval):
        jt = jt.float().cuda(non_blocking=True)
        js = js.float().cuda(non_blocking=True)
        bt = bt.float().cuda(non_blocking=True)
        bs = bs.float().cuda(non_blocking=True)
        mt = mt.float().cuda(non_blocking=True)
        ms = ms.float().cuda(non_blocking=True)

        en_hi = model(jt, js, bt, bs, mt, ms, knn_eval=True)

        label_eval_list.append(label)
        hidden_array_eval_list.append(en_hi[:, :].detach().cpu().numpy())
        
    label_eval = np.hstack(label_eval_list)
    hidden_array_eval = np.vstack(hidden_array_eval_list)

    return hidden_array_train, hidden_array_eval, label_train, label_eval


class MyAutoDataset(Dataset):
    def __init__(self, data, label):

        self.data = data
        self.label = label
        #self.xy = zip(self.data, self.label)

    def __getitem__(self, index):
        sequence = self.data[index, :]
        label = self.label[index]

        return sequence, label

    def __len__(self):
        return len(self.label)


def train_autoencoder(hidden_train, hidden_eval, label_train,
                      label_eval, middle_size, criterion, lambda1, num_epoches):
    batch_size = 64
    #auto = autoencoder(hidden_train.shape[1], middle_size).to(device)
    auto = autoencoder(hidden_train.shape[1], middle_size).cuda()
    auto_optimizer = optim.Adam(auto.parameters(), lr=0.001)
    auto_scheduler = optim.lr_scheduler.LambdaLR(auto_optimizer, lr_lambda=lambda1)
    criterion_auto = nn.MSELoss()

    autodataset = MyAutoDataset(hidden_train, label_train)
    trainloader = DataLoader(autodataset, batch_size=batch_size, shuffle=True)

    autodataset = MyAutoDataset(hidden_eval, label_eval)
    evalloader = DataLoader(autodataset, batch_size=batch_size, shuffle=True)

    print("Training autoencoder")
    for epoch in tqdm(range(num_epoches)):
        for (data, label) in trainloader:
            # img, _ = data
            # img = img.view(img.size(0), -1)
            # img = Variable(img).cuda()
            #data = torch.tensor(data.clone().detach(), dtype=torch.float).to(device)
            # ===================forward=====================
            data = data.cuda()
            output, _ = auto(data)
            loss = criterion(output, data)
            # ===================backward====================
            auto_optimizer.zero_grad()
            loss.backward()
            auto_optimizer.step()
            auto_scheduler.step()
        for (data, label) in evalloader:
            data = data.cuda()
            # ===================forward=====================
            output, _ = auto(data)
            loss_eval = criterion(output, data)
        # ===================log========================
        # if epoch % 200 == 0:
        #   print('epoch [{}/{}], train loss:{:.4f} eval loass:{:.4f}'
        #         .format(epoch + 1, num_epoches, loss.item(), loss_eval.item()))

    # extract hidden train
    count = 0
    for (data, label) in trainloader:
        data = data.cuda()
        _, encoder_output = auto(data)

        if count == 0:
            np_out_train = encoder_output.detach().cpu().numpy()
            label_train = label
        else:
            label_train = np.hstack((label_train, label))
            np_out_train = np.vstack((np_out_train, encoder_output.detach().cpu().numpy()))
        count += 1

    # extract hidden eval
    count = 0
    for (data, label) in evalloader:
        data = data.cuda()
        _, encoder_output = auto(data)

        if count == 0:
            np_out_eval = encoder_output.detach().cpu().numpy()
            label_eval = label

        else:
            label_eval = np.hstack((label_eval, label))
            np_out_eval = np.vstack((np_out_eval, encoder_output.detach().cpu().numpy()))
        count += 1

    return np_out_train, np_out_eval, label_train, label_eval


class autoencoder(nn.Module):
    def __init__(self, input_size, middle_size):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, middle_size),
            nn.Tanh()
        )

        self.decoder = nn.Sequential(