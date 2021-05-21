import torch
import torch.nn as nn
import torchvision

#Discriminator

class DCNN(nn.Module):
    '''
    the default CNN code block used in Discriminator
    is used iteratively by the latter according to the
    no. of input/output channels needed
    '''
    def __init__(self,in_channels,out_channels,stride):
        super(DCNN,self).__init__()
        self.model = nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size=4,stride=stride,
                                    padding=1,bias=False,padding_mode='reflect'),
                            nn.BatchNorm2d(out_channels),
                            nn.LeakyReLU(0.2))

    def forward(self,x):
        x = self.model(x)
        return x


class Discriminator(nn.Module):
    def __init__(self,in_channels=3,features=[64,128,256,512]):
        super(Discriminator,self).__init__()
        '''
        in_channels - RGB - 3 channels by default but for 2 images i.e. sketches and shoes

        features - list of desirable output feature maps with sizes ranging from 64-512

        build a separate Conv2d model without batch_norm as the initial model
        '''
        self.initial_model = nn.Sequential(
                                nn.Conv2d(in_channels*2,features[0], padding=1,kernel_size=4,
                                        stride=2,padding_mode='reflect'),
                                nn.LeakyReLU(0.2))
        '''
        set in channels=64 for the first iteration with out channels=128
        update in_channels inside for loop
        setting stride = 1 for output 512 as mentioned in the original idea
        '''
        layers = []
        in_channels = features[0]

        for feature in features[1:]:
            print('Each feature: ',feature)
            layers.append(DCNN(in_channels,feature,stride=1 if feature==512 else 2))
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


#Generator

class GCNN(nn.Module):
    def __init__(self, in_channels, out_channels, down=True,
                    activation='relu', keep_dropout=False):
        super(GCNN,self).__init__()
        self.model = nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size=4,
                                    stride=2,padding=1,padding_mode='reflect') if down
                                    else nn.ConvTranspose2d(in_channels,out_channels,bias=False,
                                    kernel_size=4,stride=2,padding=1),
                                    nn.BatchNorm2d(out_channels),
                                    nn.ReLU(0.2) if activation=='relu'
                                    else nn.LeakyReLU(0.2))
        self.keep_dropout = keep_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self,x):
        x = self.model(x)
        if self.keep_dropout:
            return self.dropout(x)
        else:
            return x

class Generator(nn.Module):
    def __init__(self,in_channels,features=64):
        super(Generator,self).__init__()
        self.d1 = nn.Sequential(nn.Conv2d(in_channels,features,kernel_size=4,
                                    stride=2,padding=1,padding_mode='reflect'),
                                    nn.LeakyReLU(0.2)) #128x128
        '''
        require a separate downward layer (d1) as the initial cnn
        beginning the downward part of the generator with 64 features
        '''

        self.d2 = GCNN(features,features*2,down=True,activation='leakyrelu',keep_dropout=False)
        #64x64
        self.d3 = GCNN(features*2,features*4,down=True,activation='leakyrelu',keep_dropout=False)
        #32x32
        self.d4 = GCNN(features*4,features*8,down=True,activation='leakyrelu',keep_dropout=False)
        #16x16
        self.d5 = GCNN(features*8,features*8,down=True,activation='leakyrelu',keep_dropout=False)
        #8x8
        self.d6 = GCNN(features*8,features*8,down=True,activation='leakyrelu',keep_dropout=False)
        #4x4
        self.d7 = GCNN(features*8,features*8,down=True,activation='leakyrelu',keep_dropout=False)
        #2x2
        self.bottleneck = nn.Sequential(nn.Conv2d(features*8,features*8,kernel_size=4,
                                            stride=2,padding=1,padding_mode='reflect'),
                                        nn.ReLU(0.2))
        '''
        after passing through bottleneck, images become -> 1x1
        the fully down-sampled images are passed to the upward block of the generator
        '''

        self.u1 = GCNN(features*8,features*8,down=False,activation='relu',keep_dropout=True)
        #2x2
        self.u2 = GCNN(features*8*2,features*8,down=False,activation='relu',keep_dropout=True)
        #4x4
        self.u3 = GCNN(features*8*2,features*8,down=False,activation='relu',keep_dropout=True)
        #8x8
        self.u4 = GCNN(features*8*2,features*8,down=False,activation='relu',keep_dropout=False)
        #16x16
        self.u5 = GCNN(features*8*2,features*4,down=False,activation='relu',keep_dropout=False)
        #32x32
        self.u6 = GCNN(features*4*2,features*2,down=False,activation='relu',keep_dropout=False)
        #64x64
        self.u7 = GCNN(features*2*2,features,down=False,activation='relu',keep_dropout=False)
        #128x128
        self.ufinal = nn.Sequential(nn.ConvTranspose2d(features*2,in_channels,
                                        kernel_size=4,stride=2,padding=1),
                                    nn.Tanh())
        '''
        after getting passed through the ufinal layer, the images get restored
        to their original dimensions
        '''

    def forward(self,x):
        '''
        begin the forward process - first send x through downward part of generator
        first up - initial down layer (d1)
        then send x through consequent downward layers
        finally push x through the bottleneck
        '''
        print('Downward layer: ')
        x1 = self.d1(x)
        print(x1.size())
        x2 = self.d2(x1)
        print(x2.size())
        x3 = self.d3(x2)
        print(x3.size())
        x4 = self.d4(x3)
        print(x4.size())
        x5 = self.d5(x4)
        print(x5.size())
        x6 = self.d6(x5)
        print(x6.size())
        x7 = self.d7(x6)
        print(x7.size())
        x8 = self.bottleneck(x7)
        '''
        send x to 1st up layer, consecutive upward layers
        final up layer helps upsample to original size
        '''
        print('Upward layer: ')
        y1 = self.u1(x8)
        print(y1.size())
        y2 = self.u2(torch.cat([y1,x7], dim=1))
        print(y2.size())
        y3 = self.u3(torch.cat([y2,x6], dim=1))
        print(y3.size())
        y4 = self.u4(torch.cat([y3,x5], dim=1))
        print(y4.size())
        y5 = self.u5(torch.cat([y4,x4], dim=1))
        print(y5.size())
        y6 = self.u6(torch.cat([y5,x3], dim=1))
        print(y6.size())
        y7 = self.u7(torch.cat([y6,x2], dim=1))
        print(y7.size())
        y8 = self.ufinal(torch.cat([y7,x1], dim=1))
        return y8


#test case - batches, channels, image size
xt = torch.randn((1, 3, 256, 256))

mod = Generator(in_channels=3)
output = mod(xt)
print('Final size: ',output.size())
