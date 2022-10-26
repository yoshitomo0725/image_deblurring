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
    
    # data/GoPro/{train or test}/GOPRO***_11_**/{blur or sharp}/ファイル名.png

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
 
            
        data = {"blur": img_blur, "sharp": img_sharp}
        return data
        #return img_blur , img_sharp

    def __len__(self):  # データセット数を返す
        return self.datanum
    

