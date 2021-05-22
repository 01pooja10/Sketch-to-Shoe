from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from model import Discriminator, Generator
from dataset import ShoeData
from save import save_images,save_model
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast,GradScaler

#hyperparameters
lr=0.001
device = 'cuda'
train_root = r'C:/Users/Pooja/Downloads/train'
batch_size = 16
epochs = 100
l1_lambda = 100


def train(d,g,optd,optg,dscaler,gscaler):
    train_data = ShoeData(root_directory = train_root)

    transforms_list = transforms.Compose([
                transforms.ColorJitter(brightness=0.2,saturation=0.4),
                transforms.Normalize((0.5,),(0.5,),max_pixel_value=255.0,),
                transforms.ToTensor()
                ])

    train_loader = DataLoader(train_data,transforms=transforms_list,batch_size=16,shuffle=True)
    bce_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()



def Main():
    gen = Generator(in_channels=3).to(device)
    disc = Discriminator(in_channels=3).to(device)
    optd = nn.Adam(discriminator.parameters(),lr=lr)
    optg = nn.Adam(generator.parameters(),lr=lr)
    dscaler = GradScaler()
    gscaler = GradScaler()
    train(gen,disc,optd,optg,dscaler,gscaler)

if __name__=='Main':
    Main()
