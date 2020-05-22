from PIL import Image as io
import os
import numpy as np
import pandas as pd
import glob
import random
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


class pokemon_loader(Dataset):
    def __init__(self,root,transform=None):
        """ Intialize the dataset """
        self.filenames = []
        self.root = root
        self.transform = transform
        # read filenames
        self.filenames = glob.glob(os.path.join(root,'*.png'))
        # img_labels = pd.read_csv(os.path.join(root,'pokemon.csv'))
        # np_img_label = np.array(img_labels['Male'])
        # np_img_label = np_img_label.astype(float)
        # for fn in filenames:
        #     img_num = int(os.path.basename(fn).split('.')[0])
        #     self.filenames.append((fn, np_img_label[img_num], img_num))
        self.len = len(self.filenames)

    def  __getitem__(self, index):
        image_fn = self.filenames[index]
        img = io.open(image_fn)
        
#        img = self.data_preprocess(img)
        
        if self.transform is not None:
            img = self.transform(img)
            
        sample = {'image': img}       
        return sample

#    def data_preprocess(self,img):
#        #flip
#        flip_type = random.randint(1,2) #normal=2,3,4, flip = 1 horizontal
#        if flip_type<2:
#            img = cv2.flip(img,flip_type)
#        return img

    def __len__(self):
        return self.len

def test(idx=0,root='./Data/images/'):
    floader = pokemon_loader(root)
    img = floader[idx]['image']
    plt.imshow(img)
