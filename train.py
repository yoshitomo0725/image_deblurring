import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import transforms, datasets

import os
from pathlib import Path
from math import log10
import matplotlib.pyplot as plt
import numpy as np

from model import SRCNN, UNet_2D, Unet, Win5RB, ResnetGenerator
#from deblurgan_gene import ResnetGenerator, UnetGenerator
from dataset import GoProDataset
from dataset_DMPHN import GoProDataset_DMPHN


# setting
model = ResnetGenerator()
gpu_num = 0 # 使用するGPUを設定


# コマンドライン引数
import argparse
parser = argparse.ArgumentParser(description='Debluring CNN')
#parser.add_argument('--cuda', action='store_true', default=False)
parser.add_argument('-b', '--batch', default=4)
parser.add_argument('-e', '--epoch', default=30)
parser.add_argument('-c', '--checkpoint', action='store_true', default=False)
parser.add_argument('-p', '--checkpoint_epoch')
args = parser.parse_args()

# データセットの読み込み
'''
train_set = GoProDataset("./data/GoPro/train", transform=transforms.ToTensor()
                         , crop=True, crop_size=256, rotation=True, flip=True)
train_loader = DataLoader(dataset=train_set, batch_size = BATCH_SIZE, shuffle=True)

test_set = GoProDataset("./data/GoPro/test", transform=transforms.ToTensor())
test_loader = DataLoader(dataset=test_set, batch_size = 1, shuffle=False)
'''

train_set = GoProDataset_DMPHN(
            blur_image_files = './data/GoPro/train_blur_file.txt',
            sharp_image_files = './data/GoPro/train_sharp_file.txt',
            root_dir = './data/GoPro/',
            crop = True,
            crop_size = 128,
            transform = transforms.Compose([
                transforms.ToTensor()
                ]))

train_loader = DataLoader(train_set, batch_size = args.batch, shuffle=True)

test_set = GoProDataset_DMPHN(
            blur_image_files = './data/GoPro/test_blur_file.txt',
            sharp_image_files = './data/GoPro/test_sharp_file.txt',
            root_dir = './data/GoPro/',
            crop = False,
            transform = transforms.Compose([
                transforms.ToTensor()
                ]))

test_loader = DataLoader(test_set, batch_size = 1, shuffle=False)

# 損失関数
criterion = nn.MSELoss()
device = torch.device("cuda:{}".format(gpu_num) if torch.cuda.is_available() else "cpu")

# deivceを使用
#model = ResnetGenerator()
model = model.cuda(device)
criterion = criterion.cuda(device)

# 学習率
'''optimizer = optim.Adam([{'params': model.conv1.parameters()},
                        {'params': model.conv2.parameters()},
                        {'params': model.conv3.parameters(), 'lr': 1e-5}],
                        lr=1e-4)'''
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# checkpoint読み込み
if args.checkpoint:
    checkpoint_path = 'checkpoint/model_epoch{}.pth'.format(args.checkpoint_epoch)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    psnr = checkpoint['psnr']
    
# グラフ作成用
train_loss_values = []
train_psnr_values = []
valid_loss_values = []
valid_psnr_values = []
valid_x = []

valid_loss_min = np.Inf
    
def plot_graph_train(values1, label1, name):
    plt.plot(range(len(values1)), values1, label=label1)
    plt.legend()
    plt.grid()
    plt.savefig(name + "_train.png")
    plt.clf()

# 使用するGPUの表示
print('-----------------------')
print("Using cuda:{}".format(gpu_num) if torch.cuda.is_available() else "Using CPU")
print('BATCH:{}  EPOCH:{}'.format(args.batch, args.epoch))
print('Model: {}'.format(model.__class__.__name__ ))
print('-----------------------')

# 学習
for epoch in range(args.epoch):
    model.train()
    epoch_loss, epoch_psnr = 0, 0
    test_loss, test_psnr = 0, 0
    
    for batch in train_loader:
        inputs, targets = batch[0], batch[1]
        inputs = inputs.cuda(device)
        targets = targets.cuda(device)

        optimizer.zero_grad()        
        prediction = model(inputs)
        loss = criterion(prediction, targets)
        epoch_loss += loss.data
        epoch_psnr += 10 * log10(1 / loss.data)
        
        loss.backward()
        optimizer.step()
        
    avg_train_loss = epoch_loss / len(train_loader)
    avg_train_psnr = epoch_psnr / len(train_loader)
    
    print('[Epoch {}] Loss: {:.4f}, PSNR: {:.4f} dB'.format(epoch + 1, avg_train_loss, avg_train_psnr))

    train_loss_values.append(avg_train_loss.detach().cpu())
    train_psnr_values.append(avg_train_psnr)


    # 100epochごとに下を実行
    if (epoch + 1) % 10 != 0: continue

    # 評価
    model.eval()

    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch[0], batch[1]
            inputs = inputs.cuda(device)
            targets = targets.cuda(device)
                
            prediction = model(inputs)
            loss = criterion(prediction, targets)
            test_loss += loss.data
            test_psnr += 10 * log10(1 / loss.data)

    avg_valid_loss = test_loss / len(test_loader)
    avg_valid_psnr = test_psnr / len(test_loader)
    
    print("===> Avg. Loss: {:.4f}, PSNR: {:.4f} dB".format(avg_valid_loss, avg_valid_psnr))
    valid_loss_values.append(avg_valid_loss.detach().cpu())
    valid_psnr_values.append(avg_valid_psnr)
    valid_x.append(epoch)
    
    # checkpointの保存
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_valid_loss,
            'psnr': avg_valid_psnr
            }, 'checkpoint/model_epoch{}.pth'.format(epoch+1))


    # 評価データにおいて最高精度のモデルのcheckpointの保存
    if valid_loss_values[-1] <= valid_loss_min:
        print('Valid Loss({:.4f} ===> {:.4f}).  Saving model ...'.format(valid_loss_min,valid_loss_values[-1]))

        torch.save(model.state_dict(), "model_state_dict.pth")
        valid_loss_min = valid_loss_values[-1]
        valid_max_psnr = valid_psnr_values[-1]
    
#torch.save(model.state_dict(), "model_state_dict.pth")

print("---- finish training ----")
print('min Loss: {:.4f}, max PSNR: {:.4f} dB'.format(valid_loss_min, valid_max_psnr))

plot_graph_train(train_loss_values, 'loss(train)', 'loss')
plot_graph_train(train_psnr_values, 'psnr(train)', 'psnr')

print("---- plot graph ----")
