import torch.utils.data as data
from PIL import Image, ImageOps
import glob
import os
import torchvision.transforms as transforms
from torchvision.transforms import functional as tvf
import numpy as np
import random
    
"""   
.
└─ data
   |
   └── GoPro
         ├─ test
         |   |- GoPro384_11_00
         |   |     |- blur
         |   |     |- sharp
         |   |
         |   |- GoPro384_11_05
         |   ...
         |
         └─ train
             |- GoPro372_07_00
             ...
"""

class GoProDataset(data.Dataset):
    """
    self.img_paths   画像ペアの入ってるフォルダの一つ上のディレクトリへのパス
    self.imgs_list   画像ペアの入ってるフォルダすべてのList
    self.transform   指定したtransform
    
    # data/GoPro/{train or test}/GOPRO***_11_**/{blur or sharp}/ファイル名.png
    """
    def __init__(self, img_dir, transform, crop=False, crop_size=256, rotation=False, flip=False):
        self.img_paths = img_dir # ./data/GoPro/{train or test}
        self.blur_list = glob.glob(self.img_paths + '/*/blur/*.png')
        self.sharp_list = glob.glob(self.img_paths + '/*/sharp/*.png')
        self.transform = transform
        self.crop = crop
        self.crop_size = crop_size
        self.rotation = rotation
        self.flip = flip
        self.datanum = len(self.blur_list)
        

    def __getitem__(self, index):
        img_blur = Image.open(os.path.join(self.blur_list[index])).convert('RGB')
        img_sharp = Image.open(os.path.join(self.sharp_list[index])).convert('RGB')
        
        if self.rotation:
            degree = random.choice([10, 20, 30])
            img_blur = tvf.rotate(img_blur, degree) 
            img_sharp = tvf.rotate(img_sharp, degree)
            
        if self.transform:
            img_blur = self.transform(img_blur)
            img_sharp = self.transform(img_sharp) 
            
        
        if self.crop:
            W = img_blur.size()[1]
            H = img_blur.size()[2] 

            Ws = np.random.randint(0, W-self.crop_size-1, 1)[0]
            Hs = np.random.randint(0, H-self.crop_size-1, 1)[0]
            
            img_blur = img_blur[:,Ws:Ws+self.crop_size,Hs:Hs+self.crop_size]
            img_sharp = img_sharp[:,Ws:Ws+self.crop_size,Hs:Hs+self.crop_size]
        
        if self.flip:
            for flip_num in range(2):
                img_blur = transforms.RandomHorizontalFlip(p=flip_num)(img_blur)
                img_sharp = transforms.RandomHorizontalFlip(p=flip_num)(img_sharp)
 
            
        #data = {"blur": img_blur, "sharp": img_sharp}
        #return data

        return img_blur , img_sharp

    def __len__(self):  # データセット数を返す
        return self.datanum
        #return len(glob.glob(self.img_paths + "/*/blur/*.png"))
    

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.autograd import Variable

#train_set = GoProDataset("./data/GoPro/train", transform= transforms.Compose([transforms.ToTensor()]), crop=True, crop_size=256, rotation=True, flip=True)
                         
train_set = GoProDataset("./data/GoPro/train", 
                         transform= transforms.Compose([transforms.RandomCrop(128),
                                                        transforms.RandomHorizontalFlip(p=0.5),
                                                        transforms.RandomRotation(30),
                                                        transforms.ToTensor()]))
train_loader = DataLoader(dataset=train_set, batch_size = 2, shuffle=True)
print(len(train_loader.dataset)) # 2103
print(len(train_loader)) # 1052
        
def image_show(data_loader,n):

  #Augmentationした画像データを読み込む
  tmp = iter(data_loader)
  blur,sharp = tmp.next()

  #画像をtensorからnumpyに変換
  blur_images = blur.numpy()
  sharp_images = sharp.numpy()

  #n枚の画像を1枚ずつ取り出し、表示する
  for i in range(n):
    blur = np.transpose(blur_images[i],[1,2,0])
    sharp = np.transpose(sharp_images[i],[1,2,0])
    plt.subplot(121).imshow(blur)
    plt.subplot(122).imshow(sharp)
    plt.show()
    
def show_img(dataset):
    plt.figure(figsize=(15, 3))
    for i in range(5):
        blur, sharp = dataset[i]
        blur = blur.permute(1, 2, 0)
        sharp = sharp.permute(1, 2, 0)
        plt.subplot(2, 5, i+1)
        plt.imshow(blur)
        plt.imshow(sharp)
    plt.show()
    
image_show(train_loader,10)
#show_img(train_set)


'''
# 以下、確認用
train_dataset = GoProDataset("./data/GoPro/train", transform=transforms.ToTensor())

blur, sharp = train_dataset[0]
print(blur.size())
print(sharp.size())  # torch.Size([3, 720, 1280]) torch.Size([3, 720, 1280])
print(len(train_dataset))

test_dataset = GoProDataset("./data/GoPro/test", transform=transforms.ToTensor())

blur, sharp = test_dataset[0]
print(blur.size())
print(sharp.size())
print(len(test_dataset))

showdata = GoProDataset("./data/GoPro/test", transform=None)
blur, sharp = showdata[1110]
sharp.show() # 画像表示
'''

