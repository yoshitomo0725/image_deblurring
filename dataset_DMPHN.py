import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
from skimage import io, transform
import os
import numpy as np
import random

import matplotlib.pyplot as plt

class GoProDataset_DMPHN(Dataset):
    def __init__(self, blur_image_files, sharp_image_files, root_dir, crop=False, crop_size=256, multi_scale=False, rotation=False, color_augment=False, transform=None):
        """
        Args:
             split_file: Path to the split file
             root_dir: Directory with all the images
             transform: Optional transform to be appeared on a sample
        """
        blur_file = open(blur_image_files, 'r')
        self.blur_image_files = blur_file.readlines()
        sharp_file = open(sharp_image_files, 'r')
        self.sharp_image_files = sharp_file.readlines()
        self.root_dir = root_dir
        self.transform = transform        
        self.crop = crop
        self.crop_size = crop_size
        self.multi_scale = multi_scale
        self.rotation = rotation
        self.color_augment = color_augment

    def __len__(self):
        return len(self.blur_image_files)

    def __getitem__(self, idx):
        image_name = self.blur_image_files[idx][0:-1].split('/')
        blur_image = Image.open(os.path.join(self.root_dir, image_name[0], image_name[1], image_name[2], image_name[3])).convert('RGB')
        sharp_image = Image.open(os.path.join(self.root_dir, image_name[0], image_name[1], 'sharp', image_name[3])).convert('RGB')
        
        if self.rotation:
            degree = random.choice([90, 180, 270])
            blur_image = transforms.functional.rotate(blur_image, degree) 
            sharp_image = transforms.functional.rotate(sharp_image, degree)

        if self.color_augment:
            blur_image = transforms.functional.adjust_gamma(blur_image, 1)
            sharp_image = transforms.functional.adjust_gamma(sharp_image, 1)                           
            sat_factor = 1 + (0.2 - 0.4*np.random.rand())
            blur_image = transforms.functional.adjust_saturation(blur_image, sat_factor)
            sharp_image = transforms.functional.adjust_saturation(sharp_image, sat_factor)
            
        if self.transform:
            blur_image = self.transform(blur_image)
            sharp_image = self.transform(sharp_image)

        if self.crop:
            W = blur_image.size()[1]
            H = blur_image.size()[2] 

            Ws = np.random.randint(0, W-self.crop_size-1, 1)[0]
            Hs = np.random.randint(0, H-self.crop_size-1, 1)[0]
            
            blur_image = blur_image[:,Ws:Ws+self.crop_size,Hs:Hs+self.crop_size]
            sharp_image = sharp_image[:,Ws:Ws+self.crop_size,Hs:Hs+self.crop_size]
                       
        return (blur_image, sharp_image)
        #return {'blur_image': blur_image, 'sharp_image': sharp_image}
        

"""
train_dataset = GoProDataset2(
            blur_image_files = './data/GoPro/train_blur_file.txt',
            sharp_image_files = './data/GoPro/train_sharp_file.txt',
            root_dir = './data/GoPro/',
            crop = True,
            crop_size = 256,
            transform = transforms.Compose([
                transforms.ToTensor()
                ]))

train_dataloader = DataLoader(train_dataset, batch_size = 2, shuffle=True)

blur, sharp = train_dataset.__getitem__(0)
print("train_set.__len__(): ", train_dataset.__len__())
print("blur.shape: ", blur.shape)
print("blur.shape: ", sharp.shape)

def format_image(img):
    img = np.array(np.transpose(img, (1,2,0)))
    return img
def format_mask(mask):
    mask = np.squeeze(np.transpose(mask, (1,2,0)))
    return mask

def visualize_dataset(n_images, predict=None):
  images = [0, 1, 2]
  #images = random.sample(range(0, 670), n_images)
  figure, ax = plt.subplots(nrows=len(images), ncols=2, figsize=(5, 8))
  print('select img: ', images)
  for i in range(0, len(images)):
    img_no = images[i]
    blur, sharp = train_dataset.__getitem__(img_no)
    blur = format_image(blur)
    sharp = format_mask(sharp)
    ax[i, 0].imshow(blur)
    ax[i, 1].imshow(sharp)
    ax[i, 0].set_title("Blur Image")
    ax[i, 1].set_title("Sharp Image")
    ax[i, 0].set_axis_off()
    ax[i, 1].set_axis_off()
  plt.tight_layout()
  plt.show()
visualize_dataset(3)

for epoch in range(1):
    for inputs, targets in train_dataloader:
        print(blur.shape) # torch.Size([3, 256, 256])
        print(blur)
        print(sharp.shape) # torch.Size([3, 256, 256])
        print(sharp)

"""