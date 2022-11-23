import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
#from skimage import io, transform
import os
import numpy as np
import random

import matplotlib.pyplot as plt
import json
from scipy.stats import multivariate_normal

def fill_circle(image, center, radius):
    coord = np.fromfunction(lambda y, x: np.dstack((y + 0.5, x + 0.5)), image.shape[:2])
    dist2 = (coord[:,:,1] - center[0]) ** 2 + (coord[:,:,0] - center[1]) ** 2
    condition = dist2 <= radius ** 2
    if image.ndim == 3:
        condition = np.tile(condition.reshape(condition.shape + (1,)), (1, 1, image.shape[2]))
    return np.where(condition, 1, image)

def array_to_img(x):
    '''
    データを [0, 255] で表される画像に変換する。
    '''
    #x += max(-np.min(x), 0)  # 非負の値にする。
    #x_max = np.max(x)  # [0, 1] で正規化する。
    #if x_max != 0: x /= x_max
    x *= 255  # [0, 255] にする。
    return x.astype(np.uint8)
        
class HideDataset_kpt(Dataset):
    def __init__(self, blur_image_files, sharp_image_files, kpt_files, root_dir, 
                 test=False, keypoint=False, crop=False, crop_size=256, rotation=False, color_augment=False, transform=None):
        """
        Args:
             split_file: 分割ファイルへのパス
             root_dir: 全画像ファイルを格納したディレクトリ
             transform: Optional transform to be appeared on a sample
             
        blur_image_files = './data/HIDE_dataset/train_blur_file.txt',
        sharp_image_files = './data/HIDE_dataset/train_sharp_file.txt',
        kpt_files = './data/HIDE_dataset/train_kpt_file.txt'
        root_dir = './data/HIDE_dataset/',
        """
        blur_file = open(blur_image_files, 'r')
        self.blur_image_files = blur_file.readlines()
        sharp_file = open(sharp_image_files, 'r')
        self.sharp_image_files = sharp_file.readlines()
        kpt_file = open(kpt_files, 'r')
        self.kpt_files = kpt_file.readlines()
        
        self.test = test
        self.keypoint = keypoint
        
        self.root_dir = root_dir
        self.transform = transform   
        self.crop = crop
        self.crop_size = crop_size
        self.rotation = rotation
        self.color_augment = color_augment

    def __len__(self):
        return len(self.blur_image_files)

    def __getitem__(self, idx):
        image_name = self.blur_image_files[idx][0:-1].split('/')

        if self.test:
            blur_image = Image.open(os.path.join(self.root_dir, image_name[0], image_name[1], image_name[2])).convert('RGB')
            sharp_image = Image.open(os.path.join(self.root_dir, 'GT', image_name[2])).convert('RGB')
        else:
            blur_image = Image.open(os.path.join(self.root_dir, image_name[0], image_name[1])).convert('RGB')
            sharp_image = Image.open(os.path.join(self.root_dir, 'GT', image_name[1])).convert('RGB')
        
        if self.keypoint:
            json_path = self.kpt_files[idx][0:-1].split('/')
            
            if self.test:
                with open(os.path.join(self.root_dir, 'kpt_test', json_path[1], json_path[2]), 'r') as f:
                    kpt = json.load(f)
            else:
                with open(os.path.join(self.root_dir, 'kpt_train', json_path[1]), 'r') as f:
                    kpt = json.load(f)
            
            #img_w, img_h = 1280, 720
            #XY = np.c_[np.mgrid[:img_h, :img_w].ravel(), np.mgrid[:img_h, :img_w].ravel()]
            keypoint_list = [3, 4, 6, 7, 10, 11, 13, 14]

            sum_img = np.zeros((720,1280))
            #sigma = np.eye(2) * 2000 # 分散
            #sigma = np.array([[1, 0], [0, 3]]) * 3000 #* (dist_18/100)**2
            
            for d in kpt['people']:
                kpt = np.array(d['pose_keypoints_2d']).reshape(25, 3) # 25行*3列 に変換
                
                # 首から腰までの距離算出
                kpt_Neck = np.array([kpt[1][0], kpt[1][1]])
                kpt_waist = np.array([kpt[8][0], kpt[8][1]])
                d_Neck2waist = np.linalg.norm(kpt_Neck - kpt_waist)
                
                # 円
                for i in keypoint_list:
                    if kpt[i][2] > 0: # 信頼度が...より大きい場合
                        p_x, p_y = kpt[i][0], kpt[i][1]
                        img = fill_circle(sum_img, np.array([p_x, p_y]), d_Neck2waist/3)
                        
                '''
                # 二次元ガウス分布    
                for i in keypoint_list:
                    if kpt[i][2] > 0.6:
                        p_x, p_y = kpt[i][0], kpt[i][1]
                        ax1.scatter(p_x, p_y)
                        ax2.scatter(p_x, p_y)
                        Z = multivariate_normal.pdf(x=XY, mean=np.array([p_y, p_x]), cov=sigma)
                        Z = Z.reshape(img_h, img_w)
                        sum_img = sum_img + Z
                '''
                        
            img = sum_img*= 255
            img.astype(np.uint8)
            kpt_image = Image.fromarray(img.astype(np.uint8))
        
        if self.rotation:
            degree = random.choice([90, 180, 270])
            blur_image = transforms.functional.rotate(blur_image, degree) 
            sharp_image = transforms.functional.rotate(sharp_image, degree)
            if self.keypoint:
                kpt_image = transforms.functional.rotate(kpt_image, degree)

        if self.color_augment:
            blur_image = transforms.functional.adjust_gamma(blur_image, 1)
            sharp_image = transforms.functional.adjust_gamma(sharp_image, 1)                           
            sat_factor = 1 + (0.2 - 0.4*np.random.rand())
            blur_image = transforms.functional.adjust_saturation(blur_image, sat_factor)
            sharp_image = transforms.functional.adjust_saturation(sharp_image, sat_factor)
            
        if self.transform:
            blur_image = self.transform(blur_image)
            sharp_image = self.transform(sharp_image)
            if self.keypoint:
                kpt_image = self.transform(kpt_image)

        if self.crop:
            W = blur_image.size()[1]
            H = blur_image.size()[2] 

            Ws = np.random.randint(0, W-self.crop_size-1, 1)[0]
            Hs = np.random.randint(0, H-self.crop_size-1, 1)[0]
            
            blur_image = blur_image[:,Ws:Ws+self.crop_size,Hs:Hs+self.crop_size]
            sharp_image = sharp_image[:,Ws:Ws+self.crop_size,Hs:Hs+self.crop_size]
            if self.keypoint:
                kpt_image = kpt_image[:,Ws:Ws+self.crop_size,Hs:Hs+self.crop_size]
                       
        if self.keypoint:
            return (blur_image, sharp_image, kpt_image)
        else:
            return (blur_image, sharp_image)
