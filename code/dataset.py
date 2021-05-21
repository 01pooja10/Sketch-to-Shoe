import os
import PIL
from PIL import Image
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

class ShoeData(Dataset):
    def __init__(self,root_directory):
        super(ShoeData,self).__init__()
        self.root = root_directory
        self.files = os.listdir(root_directory)
        print(files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self,idx):
        imfile = self.files[idx]
        impath = os.path.join(self.root,impath)
        img = Image.open(impath)
        img = np.array(img)
        inp = img[:,:256,:]
        out = img[:,256:,:]
        return

'''
img = np.array(Image.open(r'C:\Users\Pooja\Documents\ML_projects\Sketch-to-Shoe\data\28_AB.jpg'))
ipimg = img[:,:256,:]
opimg = img[:,256:,:]

ipimg = Image.fromarray(ipimg)
ipimg.show()

opimg = Image.fromarray(opimg)
opimg.show()
'''
