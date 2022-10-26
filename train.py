import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import transforms, datasets
from tensorboardX import SummaryWriter

import os
from pathlib import Path
from math import log10
import matplotlib.pyplot as plt

from model import SRCNN, UNet_2D, Unet
from dataset import GoProDataset


# setting
BATCH_SIZE = 4
EPOCH = 30

# コマンドライン引数
import argparse
parser = argparse.ArgumentParser(description='predictionCNN Example')
parser.add_argument('--cuda', action='store_true', default=False)
opt = parser.parse_args()

# データセットの読み込み
train_set = GoProDataset("./data/GoPro/train", transform=transforms.ToTensor()
                         , crop=True, crop_size=256, rotation=True, flip=True)
train_loader = DataLoader(dataset=train_set, batch_size = BATCH_SIZE, shuffle=True)

test_set = GoProDataset("./data/GoPro/test", transform=transforms.ToTensor())
test_loader = DataLoader(dataset=test_set, batch_size = 1, shuffle=False)


# 損失関数
criterion = nn.MSELoss()
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# モデル
#model = SRCNN()
model = UNet_2D()
#model = Unet()

if opt.cuda:
    model = model.cuda()
    criterion = criterion.cuda()


# 学習率
'''optimizer = optim.Adam([{'params': model.conv1.parameters()},
                        {'params': model.conv2.parameters()},
                        {'params': model.conv3.parameters(), 'lr': 1e-5}],
                        lr=1e-4)'''
optimizer = optim.Adam(model.parameters(), lr=0.001)

# tensorboadの設定
#writer = SummaryWriter(log_dir="./logs")

# demo画像保存用
sample_dir = './sample'
os.makedirs(sample_dir, exist_ok=True)

# グラフ作成用
train_loss_values = []
train_psnr_values = []
valid_loss_values = []
valid_psnr_values = []
valid_x = []

def plot_graph(values1, values2, valid_num, label1, label2, name):
    plt.plot(range(len(values1)), values1, label=label1)
    plt.plot(valid_num, values2, label=label2)
    plt.legend()
    plt.grid()
    plt.savefig(name + ".png")
    #plt.show()
    plt.clf() # 初期化
    
def plot_graph_train(values1, label1, name):
    plt.plot(range(len(values1)), values1, label=label1)
    plt.legend()
    plt.grid()
    plt.savefig(name + "_train.png")
    #plt.show()
    plt.clf() # 初期化


for epoch in range(EPOCH):
    model.train()
    epoch_loss, epoch_psnr = 0, 0
    test_loss, test_psnr = 0, 0
    
    for i, data in enumerate(train_loader):
        inputs, targets = data["blur"], data["sharp"]
        if opt.cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()
        optimizer.zero_grad()        
        prediction = model(inputs)
        loss = criterion(prediction, targets)
        epoch_loss += loss.data
        epoch_psnr += 10 * log10(1 / loss.data)
        
        loss.backward()
        optimizer.step()
    
    '''
    for batch in train_loader:
        inputs, targets = Variable(batch[0]), Variable(batch[1])
        if opt.cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        optimizer.zero_grad()        
        prediction = model(inputs)
        loss = criterion(prediction, targets)
        epoch_loss += loss.data
        epoch_psnr += 10 * log10(1 / loss.data)
        
        loss.backward()
        optimizer.step()
    '''
        
    avg_train_loss = epoch_loss / len(train_loader)
    avg_train_psnr = epoch_psnr / len(train_loader)

    #writer.add_scalar('train/loss', avg_train_loss, global_step=epoch)
    #writer.add_scalar('train/psnr', avg_train_psnr, global_step=epoch)
    
    print('[Epoch {}] Loss: {:.4f}, PSNR: {:.4f} dB'.format(epoch + 1, avg_train_loss, avg_train_psnr))

    train_loss_values.append(avg_train_loss.detach().cpu()) #こっちじゃね?
    train_psnr_values.append(avg_train_psnr)

    # 1000epochごとに下を実行
    if (epoch + 1) % 100 != 0: continue

    model.eval()

    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch[0], batch[1]
            if opt.cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()
                
            prediction = model(inputs)
            loss = criterion(prediction, targets)
            test_loss += loss.data
            test_psnr += 10 * log10(1 / loss.data)

            #save_image(prediction, sample_dir / '{}_epoch{:05}.png'.format(batch[2][0], epoch + 1), nrow=1)

    avg_valid_loss = test_loss / len(test_loader)
    avg_valid_psnr = test_psnr / len(test_loader)
    
    #writer.add_scalar('test/loss', test_loss / len(test_loader), global_step=epoch)
    #writer.add_scalar('test/psnr', test_psnr / len(test_loader), global_step=epoch)
    print("===> Avg. Loss: {:.4f}, PSNR: {:.4f} dB".format(avg_valid_loss, avg_valid_psnr))
    valid_loss_values.append(avg_valid_loss.detach().cpu())
    valid_psnr_values.append(avg_valid_psnr)
    valid_x.append(epoch)
    
torch.save(model.state_dict(), "model_state_dict.pth")
#writer.close()
print("---- finish training ----")

#print('train_loss', train_loss_values)
#print('train_psnr', train_psnr_values)
#print('valid_loss', valid_loss_values)
#print('valid_psnr', valid_psnr_values)
#print(valid_x)

#plot_graph(train_loss_values, valid_loss_values, valid_x, 'loss(train)', 'loss(validate)', 'loss')
#plot_graph(train_psnr_values, valid_psnr_values, valid_x, 'psnr(train)', 'psnr(validate)', 'psnr')
plot_graph_train(train_loss_values, 'loss(train)', 'loss')
plot_graph_train(train_psnr_values, 'psnr(train)', 'psnr')

print("---- plot graph ----")