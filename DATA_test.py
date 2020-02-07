import torch
import torch.nn as nn
import os
import numpy as np
import hickle as hkl
from torch.utils.data import DataLoader
from torch.autograd import Variable
from data import DATA
from HPNet import HPNet
import torchvision


def save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    from PIL import Image
    im = Image.fromarray(np.rollaxis(tensor.numpy(), 0, 3))
    im.save(filename)


batch_size = 4
P_channels = (3, 48, 96, 192)
I_channels = (3, 48, 96, 192)
R_channels = (48, 96, 192)
extrap_start_time = 20

DATA_DIR = './KTH_data'
test_file = os.path.join(DATA_DIR, 'X_test.hkl')
test_sources = os.path.join(DATA_DIR, 'sources_test.hkl')

nt = 8
nc = 5

DATA_test = DATA(test_file, test_sources, nt * nc)

test_loader = DataLoader(DATA_test, batch_size=batch_size, shuffle=False)

model = HPNet(P_channels, I_channels, R_channels, nc, output_mode='prediction', extrap_start_time=extrap_start_time)
model = nn.DataParallel(model)
model.load_state_dict(torch.load('models/training_weights.pt'))

if torch.cuda.is_available():
    print('Using GPU.')
    model.cuda()

for i, inputs in enumerate(test_loader):
    inputs = inputs.permute(0, 4, 1, 2, 3) # batch x channel x time_steps x width x height
    inputs = Variable(inputs.cuda())
    origin = (inputs.data.cpu()[:, :, :] * 255).byte()
    origin = origin.permute(0, 2, 1, 3, 4).contiguous().view(-1, origin.shape[1], origin.shape[3], origin.shape[4])


    pred = model(inputs)


    pred = (pred.data.cpu()[:, :, :] * 255).byte()
    pred = pred.permute(0, 2, 1, 3, 4).contiguous().view(-1, pred.shape[1], pred.shape[3], pred.shape[4]) # (batch x time_steps) x channel x width x height

    origin = torchvision.utils.make_grid(origin, nrow=5)
    pred = torchvision.utils.make_grid(pred, nrow=5)
    save_image(origin, 'origin.png')
    save_image(pred, 'predicted.png')
    break

