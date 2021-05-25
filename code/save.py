import torch
import torch.nn as nn
from torchvision.utils import save_image


def save_samples(model,data,folder,i):
    '''
    model - generator model
    data - the data loader(val/test)
    folder - the destination folder's path
    i - sample number
    '''
    print('Generating samples...')
    device = 'cuda'
    _,y = next(iter(data))
    y = y.to(device)
    model.eval()
    with torch.no_grad():
        fake = model(y)
        #remove 0 mean and std deviation of 1
        fake = fake*0.5 + 0.5
        save_image(fake,folder+'/fake_'+str(i)+'.png')
    model.train()

def save_model(model,optim,path):
    '''
    model - generator or discriminator
    optim - optimizer's current state
    filename - name - default: saved_model.pth
    '''
    print('Saving your model....')
    model_details = {'model':model.state_dict(),'optimizer':optim.state_dict()}
    torch.save(model_details,path+'/saved_model.pth')
