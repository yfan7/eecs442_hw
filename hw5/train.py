import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import png
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from colormap.colors import Color, hex2rgb
from sklearn.metrics import average_precision_score as ap_score
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

from dataset import FacadeDataset

N_CLASS=5

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.n_class = N_CLASS
        self.l1 = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        self.l2 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.l3 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.l4 = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2,2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.l5 = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2,2),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512,2,stride = 2)
        )
        self.l6 = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256,2,stride = 2)
        )
        
        self.l7 = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 2,stride = 2)
        )
        self.l8 = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 2,stride = 2)
        )
           
        self.l9 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,N_CLASS, 1)
        )  
  

    def forward(self, x):
        t1 = self.l1(x)
        t2 = self.l2(t1)
        t3 = self.l3(t2)
        t4 = self.l4(t3)
        t5 = torch.cat((t4,self.l5(t4)),1)
        t6 = torch.cat((t3,self.l6(t5)),1)
        t7 = torch.cat((t2,self.l7(t6)),1)
        t8 = torch.cat((t1,self.l8(t7)),1)
        t9 = self.l9(t8)
        return t9


def save_label(label, path):
    '''
    Function for ploting labels.
    '''
    colormap = [
        '#000000',
        '#0080FF',
        '#80FF80',
        '#FF8000',
        '#FF0000',
    ]
    assert(np.max(label)<len(colormap))
    colors = [hex2rgb(color, normalise=False) for color in colormap]
    w = png.Writer(label.shape[1], label.shape[0], palette=colors, bitdepth=4)
    with open(path, 'wb') as f:
        w.write(f, label)

def train(trainloader, net, criterion, optimizer, device, epoch):
    '''
    Function for training.
    '''
    start = time.time()
    running_loss = 0.0
    net = net.train()
    for images, labels in tqdm(trainloader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = net(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss = loss.item()
    end = time.time()
    print('[epoch %d] loss: %.3f elapsed time %.3f' %
          (epoch, running_loss, end-start))

def test(testloader, net, criterion, device):
    '''
    Function for testing.
    '''
    losses = 0.
    cnt = 0
    with torch.no_grad():
        net = net.eval()
        for images, labels in tqdm(testloader):
            images = images.to(device)
            labels = labels.to(device)
            output = net(images)
            loss = criterion(output, labels)
            losses += loss.item()
            cnt += 1
    print(losses / cnt)
    return (losses/cnt)


def cal_AP(testloader, net, criterion, device):
    '''
    Calculate Average Precision
    '''
    losses = 0.
    cnt = 0
    with torch.no_grad():
        net = net.eval()
        preds = [[] for _ in range(5)]
        heatmaps = [[] for _ in range(5)]
        for images, labels in tqdm(testloader):
            images = images.to(device)
            labels = labels.to(device)
            output = net(images).cpu().numpy()
            for c in range(5):
                preds[c].append(output[:, c].reshape(-1))
                heatmaps[c].append(labels[:, c].cpu().numpy().reshape(-1))

        aps = []
        for c in range(5):
            preds[c] = np.concatenate(preds[c])
            heatmaps[c] = np.concatenate(heatmaps[c])
            if heatmaps[c].max() == 0:
                ap = float('nan')
            else:
                ap = ap_score(heatmaps[c], preds[c])
                aps.append(ap)
            print("AP = {}".format(ap))
        print('avg AP: ', np.mean(aps))

    # print(losses / cnt)
    return None


def get_result(testloader, net, device, folder='output_train'):
    result = []
    cnt = 1
    with torch.no_grad():
        net = net.eval()
        cnt = 0
        for images, labels in tqdm(testloader):
            images = images.to(device)
            labels = labels.to(device)
            output = net(images)[0].cpu().numpy()
            c, h, w = output.shape
            assert(c == N_CLASS)
            y = np.zeros((h,w)).astype('uint8')
            for i in range(N_CLASS):
                mask = output[i]>0.5
                y[mask] = i
            gt = labels.cpu().data.numpy().squeeze(0).astype('uint8')
            save_label(y, './{}/y{}.png'.format(folder, cnt))
            save_label(gt, './{}/gt{}.png'.format(folder, cnt))
            plt.imsave(
                './{}/x{}.png'.format(folder, cnt),
                ((images[0].cpu().data.numpy()+1)*128).astype(np.uint8).transpose(1,2,0))

            cnt += 1

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # TODO change data_range to include all train/evaluation/test data.
    # TODO adjust batch_size.
    print()
    train_data = FacadeDataset(flag='train', data_range=(0,720), onehot=False)
    train_loader = DataLoader(train_data, batch_size=10)
    test_data = FacadeDataset(flag='test_dev', data_range=(0,114), onehot=False)
    test_loader = DataLoader(test_data, batch_size=1)
    val_data = FacadeDataset(flag='train', data_range=(720,906), onehot=False)
    val_loader = DataLoader(test_data, batch_size=10)
    ap_data = FacadeDataset(flag='test_dev', data_range=(0,114), onehot=True)
    ap_loader = DataLoader(ap_data, batch_size=1)
    print(ap_data.dataset[0][0].shape)

    name = 'starter_net'
    net = Net().to(device)
    criterion = nn.CrossEntropyLoss() #TODO decide loss
    optimizer = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-5)

    print('\nStart training')
    val_loss = []
    t_loss = []
    for epoch in range(15): #TODO decide epochs
        print('-----------------Epoch = %d-----------------' % (epoch+1))
        train(train_loader, net, criterion, optimizer, device, epoch+1)
        # TODO create your evaluation set, load the evaluation set and test on evaluation set
        
        valloss = test(val_loader, net, criterion, device)
        trainloss = test(train_loader, net, criterion, device)
        val_loss.append(valloss)
        t_loss.append(trainloss)

    print('\nFinished Training, Testing on test set')
    test(test_loader, net, criterion, device)

    print('\nGenerating Unlabeled Result')
    result = get_result(test_loader, net, device, folder='output_test')

    torch.save(net.state_dict(), './models/model_{}.pth'.format(name))

    cal_AP(ap_loader, net, criterion, device)
    plt.plot(range(1,16), t_loss, 'b',label='Training loss')
    plt.plot(range(1,16), val_loss, 'r',label='Valitation loss')
    plt.show()
    gen_data = FacadeDataset(flag='eval', data_range=(0,1), onehot=True)
    gen_loader = DataLoader(gen_data, batch_size=1)
    rst = get_result(gen_loader, net, device, folder='output_eval')

if __name__ == "__main__":
    main()
