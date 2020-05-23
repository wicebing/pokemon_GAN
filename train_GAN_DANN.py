import pandas as pd
import numpy as np
import cv2
import math 
import glob
import os
import matplotlib.pyplot as plt
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision

import GAN_v01
import Dataloader_pokemon

filepath = './Data/images/'
device1 ='cuda:1'
batch_size = 8

transform = T.Compose([T.ToPILImage(), 
                       T.RandomHorizontalFlip(p=0.5),
                       T.Resize((64,64)),
                       T.ToTensor(),
                       T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                       ])  
trainset = Dataloader_pokemon.pokemon_loader(filepath,
                                       transform=transform,
                                       )
trainset_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=12)

def z_creator(batch_size= batch_size):
    z = 2*torch.rand(batch_size,100,1,1)-1
    return z

def fix_z(batch_size= batch_size):
    np.random.seed(seed=3)
    z = 2*np.random.rand(batch_size,100,1,1)-1
    z = torch.tensor(z).float()
    return z

def target_real_fake(batch_size= batch_size, soft = 0.85):
    t = torch.ones(batch_size,1) 
    return soft*t, 1 - soft*t , t

def gen_fake_label(batch_size= batch_size):
    fl = torch.randint(0,2,[batch_size,1,1,1])
    return fl.float()

def gen_fake_label2(batch_size= batch_size):
    fl_m = torch.ones([batch_size,1,1,1])
    fl_f = torch.zeros([batch_size,1,1,1])
    fl = torch.cat([fl_m,fl_f],0)
    return fl.float()
    
def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer' : optimizer.state_dict()}
    torch.save(state, os.path.join('./checkpoint_in',checkpoint_path))
    name_='W_'+checkpoint_path
    torch.save(model,os.path.join('./checkpoint_in',name_))
    print('model saved to %s' % checkpoint_path)
    
def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(os.path.join('./checkpoint_in',checkpoint_path))
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)

def imshow(img,ep=0, outfile ='./output'):
    npimg = img.numpy()
    filename = os.path.join(outfile,str(ep)+'.jpg')
    plt.figure(figsize=[15,15])
    plt.axis = None
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(filename)   

def imshow2(img_m,img_f,ep=0, outfile ='./output'):
    npimg_m = img_m.numpy()
    npimg_f = img_f.numpy()
    filename = os.path.join(outfile,str(ep)+'.jpg')

    plt.figure(figsize=(20,10),dpi=150)
    plt.subplot(121)
    plt.axis = None
    plt.imshow(np.transpose(npimg_m, (1, 2, 0)))
    plt.subplot(122)
    plt.axis = None
    plt.imshow(np.transpose(npimg_f, (1, 2, 0)))    
    plt.savefig(filename) 

class ReverseLayerF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None
                
def train_AC_GAN_V(GAN_G,GAN_D,soft=0.85,
                   epoch=1,lr=1e-4,log_interval=100,Gr=10,startnum=36,plotep=10):
    #GAN
    GAN_G.to(device1)
    GAN_D.to(device1)        
    optimizer_GAN_G = optim.Adam(GAN_G.parameters(), lr=lr)
    optimizer_GAN_D = optim.Adam(GAN_D.parameters(), lr=lr)

    try:
        load_checkpoint('Detector.pth',GAN_D,optimizer_GAN_D)
    except:
        print('***Dectector === no trained model ===')
    try:
        load_checkpoint('Gnerator.pth',GAN_G,optimizer_GAN_G)
    except:
        print('***Gnerator === no trained model ===')
        
    criterion_GAN = nn.MSELoss().to(device1)
   
    iteration = 0
    for ep in range(epoch):
        t0 = time.time()
        GAN_G.train()
        for batch_idx, sample in enumerate(trainset_loader):
            real_img = sample['image'].to(device1)
         
            batch_s = len(real_img)
            target_real, target_fake, target_ones = target_real_fake(batch_size=batch_s,soft=soft)
            target_real, target_fake = target_real.to(device1), target_fake.to(device1)
            target_ones = target_ones.to(device1)
        
            target_real+=(1.1*(1-soft)*torch.rand_like(target_real))
            target_fake-=(1.1*(1-soft)*torch.rand_like(target_fake))
            
            fake_label = gen_fake_label(batch_size=batch_s)

            
            #train Discriminator every batch
            if batch_idx % 1 ==0:
                optimizer_GAN_D.zero_grad()
                output = GAN_D(real_img)
                D_loss_real = criterion_GAN(output.float(), target_real.float()) 
                              
                z = z_creator(batch_size=batch_s).to(device1)
                fake_img = GAN_G(z)

                reverse_fake_img= ReverseLayerF.apply(fake_img, 1.2)
                output = GAN_D(reverse_fake_img)
                D_loss_fake = criterion_GAN(output.float(), target_fake.float())
                
                D_loss = D_loss_real+D_loss_fake
                D_loss.backward()
                optimizer_GAN_D.step()
                optimizer_GAN_G.step()
                                           
            #train Generator every ?? batch
            # if batch_idx % Gr ==0:
            #     optimizer_GAN_G.zero_grad()
            #     z = z_creator(batch_size=batch_s).to(device1)
            #     fake_img = GAN_G(z)
            #     output = GAN_D(fake_img)
            #     loss_fake = criterion_GAN(output.float(), target_ones.float())
            #     loss_fake.backward()            
            #     optimizer_GAN_G.step()
                                                    

            if iteration % log_interval == 0:
                print('Ep:{} [{} ({:.0f}%)] Dr:{:.4f} Df:{:.4f}'.format(
                    ep, batch_idx * len(real_img),
                    100. * batch_idx / len(trainset_loader),
                    D_loss_real.item(),D_loss_fake.item()))   
            iteration +=1

        if ep % plotep ==0:
            save_checkpoint('Gnerator.pth', GAN_G, optimizer_GAN_G)
            save_checkpoint('Detector.pth', GAN_D, optimizer_GAN_D)

            print('======= epoch:%i ========'%ep)
            imshow(torchvision.utils.make_grid(fake_img[:36].detach().cpu(),6),
                   ep+0+int(startnum),outfile ='../output/GAN_in/')

            GAN_G.eval()
            with torch.no_grad():       
                z_fix1 = fix_z(batch_size=36).to(device1)                
                fix_GAN = GAN_G(z_fix1)
                imshow(torchvision.utils.make_grid(fix_GAN[:36].detach().cpu(),6),
                      'GAN'+str(ep+0+int(startnum)),outfile ='../output/fix_in/')

            
        print('++ Ep Time: {:.1f} Secs ++'.format(t0-time.time()))

    save_checkpoint('Gnerator.pth', GAN_G, optimizer_GAN_G)
    save_checkpoint('Detector.pth', GAN_D, optimizer_GAN_D)

G_gen = GAN_v01.Generator()
G_dis = GAN_v01.Discriminator()

print('==== Start Training ====')
print('batch_size = ',batch_size)
train_AC_GAN_V(G_gen,G_dis,
               soft=0.9,
               epoch=10000,lr=2e-4,
               log_interval=int(5000/batch_size),Gr=1,
               startnum=0,plotep=8)
    