import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
from torchvision import transforms, datasets

from pathlib import Path
from math import log10

from model import SRCNN, UNet_2D
from dataset import GoProDataset

import argparse
parser = argparse.ArgumentParser(description='predictionCNN Example')
parser.add_argument('--cuda', action='store_true', default=False)
parser.add_argument('--weight_path', type=str, default='model_state_dict.pth')
#parser.add_argument('--save_dir', type=str, default='./sample/')
opt = parser.parse_args()

test_set = GoProDataset("./data/GoPro/test", transform=transforms.ToTensor())
test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)

model = UNet_2D()
#model = SRCNN()

criterion = nn.MSELoss()
if opt.cuda:
    model = model.cuda()
    criterion = criterion.cuda()

model.load_state_dict(torch.load(opt.weight_path, map_location='cuda' if opt.cuda else 'cpu'))

model.eval()
total_loss, total_psnr = 0, 0
total_loss_b, total_psnr_b = 0, 0
n = 0

with torch.no_grad():
    for i, data in enumerate(test_loader):
        inputs, targets = data["blur"], data["sharp"]
        if opt.cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()      
        prediction = model(inputs)
        loss = criterion(prediction, targets)
        total_loss += loss.data
        total_psnr += 10 * log10(1 / loss.data)

        loss = criterion(inputs, targets)
        total_loss_b += loss.data
        total_psnr_b += 10 * log10(1 / loss.data)
        n += 1
        if n % 100 == 0:
            save_image(prediction, '{}_pred.png'.format(n), nrow=1)
            save_image(inputs, '{}_blur.png'.format(n), nrow=1)
            save_image(targets, '{}_sharp.png'.format(n), nrow=1)
            

#save_image(prediction, 'pred.png', nrow=1)
#save_image(inputs, 'blur.png', nrow=1)
#save_image(targets, 'sharp.png', nrow=1)

print("===> [blur] Avg. Loss: {:.4f}, PSNR: {:.4f} dB".format(total_loss_b / len(test_loader), total_psnr_b / len(test_loader)))
print("===> [prediction] Avg. Loss: {:.4f}, PSNR: {:.4f} dB".format(total_loss / len(test_loader), total_psnr / len(test_loader)))