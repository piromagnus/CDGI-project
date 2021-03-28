from model import generator, discriminator
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

#from model import _netlocalD,_netG
import utils
epochs=100
epochs=100
Batch_Size=64
lr=0.0002   
beta1=0.5
over=4
#parser = argparse.ArgumentParser()
#parser.add_argument('--dataroot',  default='../TrainingData', help='path to dataset')
#opt = parser.parse_args()
try:
    os.makedirs("result/train/cropped")
    os.makedirs("result/train/real")
    os.makedirs("result/train/recon")
    os.makedirs("model")
except OSError:
    pass

transform = transforms.Compose([transforms.Scale(128),
                                    transforms.CenterCrop(128),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = dset.ImageFolder(root="../TrainingData", transform=transform )
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=Batch_Size,
                                         shuffle=True, num_workers=2)

#ngpu = int(opt.ngpu)

wtl2 = 0.999

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


resume_epoch=0

netG = generator()
netG.apply(weights_init)


netD = discriminator()
netD.apply(weights_init)

criterion = nn.BCELoss()
criterionMSE = nn.MSELoss()

input_real = torch.FloatTensor(Batch_Size, 3, 128, 128)
input_cropped = torch.FloatTensor(Batch_Size, 3, 128, 128)
label = torch.FloatTensor(Batch_Size)
real_label = 1
fake_label = 0

real_center = torch.FloatTensor(Batch_Size, 3, 64,64)


netD.cuda()
netG.cuda()
criterion.cuda()
criterionMSE.cuda()
input_real, input_cropped,label = input_real.cuda(),input_cropped.cuda(), label.cuda()
real_center = real_center.cuda()

input_real = Variable(input_real)
input_cropped = Variable(input_cropped)
label = Variable(label)

real_center = Variable(real_center)

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))