import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
#from torch.autograd import Variable
#from torchvision.utils import save_image
from torchvision import transforms, datasets


#import os
#from pathlib import Path
from math import log10
import matplotlib.pyplot as plt
import numpy as np
import csv
from tqdm import tqdm

#from model import SRCNN, UNet_2D, Unet, Win5RB, ResnetGenerator
from model_DMPHN import DMPHN_Unet

#from dataset import GoProDataset
#from dataset_DMPHN import GoProDataset_DMPHN, HIDEDataset_DMPHN
from dataset_keypoint import HideDataset_kpt
from loss import VGGLoss, Keypoint_Loss


# setting
model = DMPHN_Unet()
gpu_num = 1 # 使用するGPUを設定
#checkpoint_epoch = 100

torch.backends.cudnn.benchmark = True

# Path
#checkpoint_path = 'checkpoint/model_epoch{}.pth'.format(checkpoint_epoch)
train_result_path = 'result_train.csv'
valid_result_path = 'result_valid.csv'
setting_path = 'result_setting.csv'

dataset_dir = './data/HIDE_dataset/'

# コマンドライン引数
import argparse
parser = argparse.ArgumentParser(description='Debluring CNN')
#parser.add_argument('--cuda', action='store_true', default=False)
parser.add_argument('-b', '--batch', type=int, default=32) # 32or64
parser.add_argument('-e', '--epoch', type=int, default=100)
parser.add_argument('-c', '--checkpoint', action='store_true', default=False)
parser.add_argument('-p', '--checkpoint_epoch')
parser.add_argument('-s', '--crop_size', type=int, default=256)
parser.add_argument('-k', '--keypoint', action='store_true', default=False)
args = parser.parse_args()

# データセットの読み込み
test_set = HideDataset_kpt(
            blur_image_files = './data/HIDE_dataset/test_blur_file.txt',
            sharp_image_files = './data/HIDE_dataset/test_sharp_file.txt',
            kpt_files = './data/HIDE_dataset/test_kpt_file.txt',
            root_dir = './data/HIDE_dataset/',
            keypoint=args.keypoint, 
            test=True, 
            transform = transforms.Compose([
                transforms.ToTensor()
                ]))

test_loader = DataLoader(test_set, batch_size = 1, shuffle=False)

# 損失関数
if args.keypoint:
    MSE_Loss = nn.MSELoss()
    kpt_Loss = Keypoint_Loss()
else:
    criterion = nn.MSELoss()

# deivceを使用
device = torch.device("cuda:{}".format(gpu_num) if torch.cuda.is_available() else "cpu")
model = model.cuda(device)
if args.keypoint:
    MSE_Loss = MSE_Loss.cuda(device)
    kpt_Loss = kpt_Loss.cuda(device)
else:
    criterion = criterion.cuda(device)

# 学習率
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# checkpoint読み込み
if args.checkpoint:
    checkpoint_path = 'model_epoch{}.pth'.format(args.checkpoint_epoch) # 'checkpoint/model_epoch{}.pth'
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    f = open(train_result_path, mode="a")
    g = open(valid_result_path, mode="a")
    h = open(setting_path, mode="a", newline="")
    writer_f = csv.writer(f)
    writer_g = csv.writer(g)
    writer_h = csv.writer(h)
                
    start_epoch = checkpoint['epoch'] + 1
    loss = checkpoint['loss']
    psnr = checkpoint['psnr']
    valid_loss_min = checkpoint['min_loss']
    
    writer_h.writerow(['[Epoch]', start_epoch, ' >>> ', args.epoch])
    
else:
    start_epoch = 0
    f = open(train_result_path, mode="w", newline="")
    g = open(valid_result_path, mode="w", newline="")
    h = open(setting_path, mode="w", newline="")
    writer_f = csv.writer(f)
    writer_g = csv.writer(g)
    writer_h = csv.writer(h)
    writer_f.writerow(['Epoch', 'Loss', 'PSNR'])
    writer_g.writerow(['Epoch', 'Loss', 'PSNR'])
    valid_loss_min = np.Inf
    

# 設定の表示
print('-----------------------')
print("Using cuda: {}".format(gpu_num) if torch.cuda.is_available() else "Using CPU")
print('BATCH: {}  EPOCH: {}'.format(args.batch, args.epoch))
print('CropSize: {}'.format(args.crop_size))
print('Model: {}'.format(model.__class__.__name__ ))
print('With Keypoint'if args.keypoint else 'Without Keypoint')
print('Using Checkpoint (Epoch: {})'.format(args.checkpoint_epoch) if args.checkpoint else 'First training')
print('-----------------------')

# 設定をcsvに保存
setting = [['BATCH', args.batch], 
           ['EPOCH', args.epoch], 
           ['Model', model.__class__.__name__],
           ['Train_Dataset', dataset_dir]]
writer_h.writerow(['cuda', gpu_num]if torch.cuda.is_available() else ['CPU'])
writer_h.writerow(['With keypoint']if args.keypoint else ['Without keypoint'])
for data in setting:
    writer_h.writerow(data)

h.close()

# 学習
for epoch in range(start_epoch, args.epoch):
    train_set = HideDataset_kpt(
            blur_image_files = './data/HIDE_dataset/train_blur_file.txt',
            sharp_image_files = './data/HIDE_dataset/train_sharp_file.txt',
            kpt_files = './data/HIDE_dataset/train_kpt_file.txt',
            root_dir = './data/HIDE_dataset/',
            keypoint=args.keypoint, 
            test=False,
            crop = True,
            #rotation=True, 
            transform = transforms.Compose([
                transforms.ToTensor()
                ]))

    train_loader = DataLoader(train_set, batch_size = args.batch, shuffle=True) # , collate_fn=collate_fn)
    
    model.train()
    epoch_loss, epoch_psnr = 0, 0
    test_loss, test_psnr = 0, 0
    
    pbar = tqdm(train_loader, desc = 'description')

    for batch in pbar:
        if args.keypoint:
            inputs, targets, keypoint = batch[0], batch[1], batch[2]
            inputs = inputs.cuda(device)
            targets = targets.cuda(device)
            keypoint = keypoint.cuda(device)
        else:
            inputs, targets = batch[0], batch[1]
            inputs = inputs.cuda(device)
            targets = targets.cuda(device)

        optimizer.zero_grad()        
        prediction = model(inputs)
        
        if args.keypoint:
            loss_MSE = MSE_Loss(prediction, targets)
            loss_kpt = kpt_Loss(prediction, targets, keypoint)
            loss = loss_MSE + 0.1 * loss_kpt
            epoch_loss += loss.item()
            epoch_psnr += 10 * log10(1 / loss_MSE.item())
        else:
            loss = criterion(prediction, targets)
            epoch_loss += loss.data
            epoch_psnr += 10 * log10(1 / loss.data)
        
        loss.backward()
        optimizer.step()
        pbar.set_description("[Train {}] total Loss: {:.2f}, total PSNR: {:.2f}dB"
                             .format(epoch + 1, epoch_loss/len(train_loader), epoch_psnr/len(train_loader)))
        
    avg_train_loss = epoch_loss / len(train_loader)
    avg_train_psnr = epoch_psnr / len(train_loader)
    
    print('[Epoch {}] Loss: {:.6f}, PSNR: {:.6f} dB'
          .format(epoch + 1, avg_train_loss, avg_train_psnr))

    
    writer_f.writerow([epoch+1, avg_train_loss, avg_train_psnr])

    # 10epochごとに評価
    if (epoch + 1) % 10 == 0 or epoch == 0:

        # 評価
        model.eval()

        with torch.no_grad():
            pbar = tqdm(test_loader, desc = 'description')
            for batch in pbar:
                if args.keypoint:
                    inputs, targets, keypoint = batch[0], batch[1], batch[2]
                    keypoint = keypoint.cuda(device)
                else:
                    inputs, targets = batch[0], batch[1]
                inputs = inputs.cuda(device)
                targets = targets.cuda(device)
                    
                prediction = model(inputs)
                if args.keypoint:
                    loss_MSE = MSE_Loss(prediction, targets)
                    loss_kpt = kpt_Loss(prediction, targets, keypoint)
                    loss = loss_MSE + 0.1 * loss_kpt
                    test_loss += loss.item()
                    test_psnr += 10 * log10(1 / loss_MSE.item())
                else:
                    loss = criterion(prediction, targets)
                    test_loss += loss.item()
                    test_psnr += 10 * log10(1 / loss.item())
                pbar.set_description("[Valid {}]".format(epoch + 1))

        avg_valid_loss = test_loss / len(test_loader)
        avg_valid_psnr = test_psnr / len(test_loader)
        
        print("===> Avg. Loss: {:.6f}, PSNR: {:.6f} dB".format(avg_valid_loss, avg_valid_psnr))

        writer_g.writerow([epoch+1, avg_valid_loss, avg_valid_psnr])
        
        # 評価データにおいて最高精度のモデルのcheckpointの保存
        if avg_valid_loss <= valid_loss_min:
            print('Valid Loss({:.6f} ===> {:.6f}). Saving model'
                  .format(valid_loss_min, avg_valid_loss))

            torch.save(model.state_dict(), "model_state_dict.pth")
            valid_loss_min = avg_valid_loss
            valid_max_psnr = avg_valid_psnr
        
        # checkpointの保存
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_valid_loss,
                'psnr': avg_valid_psnr,
                'min_loss':valid_loss_min
                }, 'model_epoch{}.pth'.format(epoch+1)) # checkpoint/model_epoch{}.pth

    
#torch.save(model.state_dict(), "model_state_dict.pth")

f.close()
g.close()

print("---- finish training ----")
print('min Loss: {:.6f}, max PSNR: {:.6f} dB'.format(valid_loss_min, valid_max_psnr))
