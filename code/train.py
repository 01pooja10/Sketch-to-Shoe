import torch
#used for CUDA run of out memory error
torch.cuda.empty_cache()
import torchvision.transforms as transforms
from model import Discriminator, Generator
from dataset import ShoeData
from save import save_samples,save_model
import torch.nn as nn
import config
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast,GradScaler


def train(d,g,optd,optg,dscaler,gscaler):

    transforms_list = transforms.Compose([
                transforms.ColorJitter(brightness=0.2,saturation=0.4),
                transforms.ToTensor(),
                transforms.Normalize((0.5,),(0.5,))
                ])

    train_data = ShoeData(root_directory=config.train_root, transforms=transforms_list)

    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    bce_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    for epoch in range(config.epochs):

        for idx, (x,y) in enumerate(tqdm(train_loader)):
            x = x.to(config.device)
            y = y.to(config.device)

            #training the discriminator
            with autocast():
                yfake = g(x)
                dreal = d(x,y)
                dfake = d(x,yfake.detach())
                drloss = bce_loss(dreal,torch.ones_like(dreal))
                dfloss = bce_loss(dfake,torch.ones_like(dfake))
                dloss = (drloss+dfloss)/2
            optd.zero_grad()
            dscaler.scale(dloss).backward()
            dscaler.step(optd)
            dscaler.update()

            #training the generator
            with autocast():
                dfake = d(x,yfake)
                gfloss = bce_loss(dfake,torch.ones_like(dfake))
                l1 = l1_loss(yfake,y)*(config.l1_lambda)
                gloss = gfloss+l1
            optg.zero_grad()
            gscaler.scale(gloss).backward()
            gscaler.step(optg)
            gscaler.update()
        print('Epoch: ',str(epoch),'Disc Loss: ',str(dloss.item()), 'Gen Loss: ',str(gloss.item()))

        #loading validation results as images into a separate folder
        transform_val = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))
        ])
        val_data = ShoeData(root_directory=config.val_root,transforms=transform_val)
        val_loader = DataLoader(val_data,batch_size=config.batch_size,shuffle=False)
        save_samples(g,val_loader,config.save_folder,epoch)
    print('Discriminator Loss: ',str(dloss.item()), 'Generator Loss: ',str(gloss.item()))

    save_model(d,optd,config.dmodel_path)
    save_model(g,optg,config.gmodel_path)
    print('Model saved :)')

def main():

    gen = Generator(in_channels=3).to(config.device)
    disc = Discriminator(in_channels=3).to(config.device)
    optd = optim.Adam(disc.parameters(),lr=config.lr)
    optg = optim.Adam(gen.parameters(),lr=config.lr)
    dscaler = GradScaler()
    gscaler = GradScaler()
    train(disc,gen,optd,optg,dscaler,gscaler)
    print('Training Complete!')


if __name__=='__main__':
    main()
