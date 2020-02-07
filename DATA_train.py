from __future__ import print_function
import os
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import DATA
from HPNet import HPNet
from debug import info


num_epochs = 100
batch_size = 4
P_channels = (3, 48, 96, 192)
I_channels = (3, 48, 96, 192)
R_channels = (48, 96, 192)
lr = 0.001 
nt = 8 
nc = 5 


layer_loss_weights = Variable(torch.FloatTensor([[1.], [0.], [0.], [0.]]).cuda())
time_loss_weights = 1./(nt - 1) * torch.ones(nt, 1)
time_loss_weights[0] = 0
time_loss_weights = Variable(time_loss_weights.cuda())

DATA_DIR = './KTH_data/'

train_file = os.path.join(DATA_DIR, 'X_train.hkl')
train_sources = os.path.join(DATA_DIR, 'sources_train.hkl')
val_file = os.path.join(DATA_DIR, 'X_val.hkl')
val_sources = os.path.join(DATA_DIR, 'sources_val.hkl')


DATA_train = DATA(train_file, train_sources, nt * nc)
DATA_val = DATA(val_file, val_sources, nt * nc)

train_loader = DataLoader(DATA_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(DATA_val, batch_size=batch_size, shuffle=True)

model = HPNet(P_channels, I_channels, R_channels, nc, output_mode='error')
model = nn.DataParallel(model)
if torch.cuda.is_available():
    print('Using GPU.')
    sys.stdout.flush()
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)


def lr_scheduler(optimizer, epoch):
    if epoch < num_epochs //2:
        return optimizer
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001
        return optimizer


for epoch in range(num_epochs):
    optimizer = lr_scheduler(optimizer, epoch)
    for i, inputs in enumerate(train_loader):
        inputs = inputs.permute(0, 4, 1, 2, 3)  # batch x channel x time_steps x width x height
        inputs = Variable(inputs.cuda())
        errors = model(inputs)  # batch x n_layers x nt
        loc_batch = errors.size(0)
        errors = torch.mm(errors.view(-1, nt), time_loss_weights)  # batch*n_layers x 1
        errors = torch.mm(errors.view(loc_batch, -1), layer_loss_weights)

        errors = torch.mean(errors)

        optimizer.zero_grad()

        errors.backward()

        optimizer.step()
        if i % 10 == 0:
            print('Epoch: {}/{}, step: {}/{}, errors: {:.4f}e-3'.format(epoch, num_epochs, i, len(DATA_train)//batch_size,
                                                                 errors.data[0] * 1e3))
            sys.stdout.flush()
    if (epoch + 1) % 10 == 0:
        cp_paths = ['models', 'training_{}.pt'.format(epoch + 1)]
        if not os.path.exists(cp_paths[0]):
            os.mkdir(cp_paths[0])
        torch.save(model.state_dict(), '/'.join(cp_paths))
