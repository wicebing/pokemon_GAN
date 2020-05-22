import pandas as pd
import numpy as np
from PIL import Image
import cv2
import math 
import glob
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import dataloader, dataset
import torch.utils.model_zoo as model_zoo
import torchvision
import torchvision.transforms as T


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.gen1= nn.Sequential(nn.ConvTranspose2d(100,1024,kernel_size=3, stride=1, padding=0),
                                 nn.BatchNorm2d(1024),
                                 nn.LeakyReLU(0.1),
                                 
                                 nn.ConvTranspose2d(1024,512,kernel_size=3, stride=2, padding=1),
                                 nn.BatchNorm2d(512),
                                 nn.LeakyReLU(0.1),
                                 )
        self.rc1 = nn.Sequential(nn.ConvTranspose2d(512,256,kernel_size=3, stride=1, padding=1),
                                 nn.BatchNorm2d(256),
                                 nn.LeakyReLU(0.1),
                                 nn.ConvTranspose2d(256,256,kernel_size=3, stride=1, padding=1),
                                 nn.BatchNorm2d(256),
                                 nn.LeakyReLU(0.1),
                                 )
        self.gen2= nn.Sequential(nn.ConvTranspose2d(768,256,kernel_size=3, stride=2, padding=1),
                                 nn.BatchNorm2d(256),
                                 nn.LeakyReLU(0.1),                                 
                                 nn.ConvTranspose2d(256,128,kernel_size=3, stride=2, padding=1),
                                 nn.BatchNorm2d(128),
                                 nn.LeakyReLU(0.1),
                                 )
        self.rc2 = nn.Sequential(nn.ConvTranspose2d(128,64,kernel_size=3, stride=1, padding=1),
                                 nn.BatchNorm2d(64),
                                 nn.LeakyReLU(0.1),
                                 nn.ConvTranspose2d(64,64,kernel_size=3, stride=1, padding=1),
                                 nn.BatchNorm2d(64),
                                 nn.LeakyReLU(0.1),
                                 )
        self.gen3= nn.Sequential(nn.ConvTranspose2d(192,64,kernel_size=3, stride=2, padding=1),
                                 nn.BatchNorm2d(64),
                                 nn.LeakyReLU(0.1),
                                 )
        self.out = nn.Sequential(nn.ConvTranspose2d(64,3,kernel_size=4,stride=2,padding=2),
                                 nn.Sigmoid()
                                 )
    def forward(self,x):
        x = self.gen1(x)
        xrc = self.rc1(x)
        x = torch.cat([x,xrc],1).contiguous()
        x = self.gen2(x)
        xrc = self.rc2(x)
        x = torch.cat([x,xrc],1).contiguous()
        x = self.gen3(x)
        x = self.out(x)        
#        print(x.shape)
        return x       

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.gen1= nn.Sequential(nn.Conv2d(3,32,kernel_size=4,stride=2,padding=2),
                                 nn.LeakyReLU(0.2),
                                 nn.Dropout(0.5),
                                 
                                 nn.Conv2d(32,64,kernel_size=3,stride=2,padding=1),
                                 nn.BatchNorm2d(64),
                                 nn.LeakyReLU(0.2),
                                 nn.Dropout(0.5),
                                 )
        self.rc1 = nn.Sequential(nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
                                 nn.BatchNorm2d(64),
                                 nn.LeakyReLU(0.2),
                                 nn.Dropout(0.5),
                                 nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
                                 nn.BatchNorm2d(64),
                                 nn.LeakyReLU(0.2),
                                 nn.Dropout(0.5),
                                 )       
        self.gen2= nn.Sequential(nn.Conv2d(128,128,kernel_size=3,stride=2,padding=1),
                                 nn.BatchNorm2d(128),
                                 nn.LeakyReLU(0.2),
                                 nn.Dropout(0.5),    

                                 nn.Conv2d(128,256,kernel_size=3,stride=2,padding=1),
                                 nn.BatchNorm2d(256),
                                 nn.LeakyReLU(0.2),
                                 nn.Dropout(0.5), 
                                 )
        self.rc2 = nn.Sequential(nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
                                 nn.BatchNorm2d(256),
                                 nn.LeakyReLU(0.2),
                                 nn.Dropout(0.5),
                                 nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
                                 nn.BatchNorm2d(256),
                                 nn.LeakyReLU(0.2),
                                 nn.Dropout(0.5),
                                 )  
        self.gen3= nn.Sequential(nn.Conv2d(512,512,kernel_size=3,stride=2,padding=1),
                                 nn.BatchNorm2d(512),
                                 nn.LeakyReLU(0.2),
                                 nn.Dropout(0.5),  
                                 nn.Conv2d(512,1024,kernel_size=3,stride=1,padding=1),
                                 nn.BatchNorm2d(1024),
                                 nn.LeakyReLU(0.2),
                                 nn.Dropout(0.5),
                                 )
        self.out = nn.Sequential(nn.Conv2d(1024,1,kernel_size=3,stride=1,padding=0),
                                 nn.Sigmoid()
                                 )
    def forward(self,x):
        x = self.gen1(x)
        xrc = self.rc1(x)
        x = torch.cat([x,xrc],1).contiguous()
        x = self.gen2(x)
        xrc = self.rc2(x)
        x = torch.cat([x,xrc],1).contiguous()        
        x = self.gen3(x)
        x = self.out(x)
#        print(x.shape)
        x = x.view(-1,1)
        return x 

def test():
    GEN = Generator()
    z = torch.randn(1,100,1,1)
    output = GEN(z)
    print(output.shape)
    
    DEC = Discriminator()
    output1 = DEC(output)
    print(output1.shape)
    
if __name__ == '__main__':
    test()