import torch
import torch.nn as nn
import torchvision

#default CNN block
class CNN(nn.Module):
    '''
    the default CNN code used in Discriminator
    accepts in and out channels
    '''
    def __init__(self,in_channels,out_channels,stride):
        super(CNN,self).__init__()
        self.model = nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size=4,stride=stride,
                                    padding=1,bias=False,padding_mode='reflect'),
                            nn.BatchNorm2d(out_channels),
                            nn.LeakyReLU(0.2))

    def forward(self,x):
        x = self.model(x)
        return x

#Discriminator model
class Discriminator(nn.Module):
    def __init__(self,in_channels=3,features=[64,128,256,512]):
        super(Discriminator,self).__init__()
        '''
        in_channels - RGB - 3 channels by default but for 2 images i.e. sketches and shoes

        features - list of desirable output feature maps with sizes ranging from 64-512

        build a separate Conv2d model without batch_norm
        '''
        self.initial_model = nn.Sequential(
                                nn.Conv2d(in_channels*2,features[0], padding=1,kernel_size=4,
                                        stride=2,padding_mode='reflect'),
                                nn.LeakyReLU(0.2))
        '''
        set in channels=64 for the first iteration with out channels=128

        setting stride = 1 for output 512 as mentioned in the paper
        '''
        layers = []
        in_channels = features[0]

        for feature in features[1:]:
            print('Each feature: ',feature)
            layers.append(CNN(in_channels,feature,stride=1 if feature==512 else 2))
            #update in channel with each iteration
            in_channels = feature
        #final model
        print('List of layers: ',layers)

        '''
        adding another conv2d layer to output only 1 channel as Discriminator produces
        probability of image being real or fake
        '''
        layers.append(nn.Conv2d(in_channels,1, padding=1,kernel_size=4,
                      stride=2,padding_mode='reflect'))

        self.model = nn.Sequential(*layers)


    def forward(self,x,y):
        x = torch.cat([x,y], dim=1)
        x = self.initial_model(x)
        x = self.model(x)
        return x

#test case - batches, channels, image size
xt = torch.randn((1, 3, 256, 256))
yt = torch.randn((1, 3, 256, 256))

mod = Discriminator(in_channels=3)
output = mod(xt,yt)
print(output.size())



class Generator(nn.Module):
    def __init__(self,in_size,channels):
        super(Generator,self).__init__()
