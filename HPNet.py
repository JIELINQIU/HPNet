import torch
import torch.nn as nn
from torch.nn import functional as F
from convlstmcell import Conv3dLSTM
from torch.autograd import Variable
from debug import info
import pdb


class HPNet(nn.Module):
    def __init__(self, P_channels, I_channels, R_channels, num_chunks, output_mode='error', extrap_start_time=-1):
        super(HPNet, self).__init__()
        self.p_channels = P_channels + (0, )
        self.i_channels = I_channels
        self.r_channels = R_channels
        self.n_layers = len(P_channels)
        self.output_mode = output_mode
        self.num_chunks = num_chunks
        self.extrap_start_time = extrap_start_time

        default_output_modes = ['prediction', 'error']
        assert output_mode in default_output_modes, 'Invalid output_mode: ' + str(output_mode)

        for i in range(self.n_layers):
            if i == 0:
                inp_channels = 2 * self.i_channels[i] + self.p_channels[i+1]
            else:
                inp_channels = 2 * self.i_channels[i] + self.r_channels[i-1] + \
                               2 * self.i_channels[i-1] + self.p_channels[i+1]
            cell = Conv3dLSTM(inp_channels, self.p_channels[i], (3, 3, 3))
            setattr(self, 'cell{}'.format(i), cell)

        for i in range(self.n_layers):
            conv = nn.Sequential(nn.Conv3d(self.p_channels[i], self.i_channels[i], 3, padding=1), nn.ReLU())
            if i == 0:
                conv.add_module('satlu', SatLU())
            setattr(self, 'conv{}'.format(i), conv)

            spconv_e = nn.Sequential(nn.Conv3d(2*self.i_channels[i], 2*self.i_channels[i], 3, padding=1))
            setattr(self, 'spconv_e{}'.format(i), spconv_e)

        self.upsample = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear')
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.downsample = nn.Sequential(nn.ReLU(), self.maxpool)

        for l in range(self.n_layers - 1):
            update_I = nn.Sequential(self.maxpool, nn.ReLU())
            setattr(self, 'update_I{}'.format(l), update_I)

            spconv_r = nn.Sequential(nn.Conv3d(self.i_channels[l], self.r_channels[l], 3, padding=1))
            setattr(self, 'spconv_r{}'.format(l), spconv_r)

        self.reset_parameters()

    def reset_parameters(self):
        for l in range(self.n_layers):
            cell = getattr(self, 'cell{}'.format(l))
            cell.reset_parameters()

    def forward(self, input):

        P_seq = [None] * self.n_layers
        H_seq = [None] * self.n_layers
        E_seq = [None] * self.n_layers
        dE_seq = [None] * self.n_layers
        I_seq = [None] * self.n_layers
        R_seq = [None] * (self.n_layers - 1)

        w, h = input.size(-2), input.size(-1)
        batch_size = input.size(0)

        for l in range(self.n_layers):
            E_seq[l] = Variable(torch.zeros(batch_size, 2*self.i_channels[l], self.num_chunks, w, h)).cuda()
            dE_seq[l] = Variable(torch.zeros(batch_size, 2*self.i_channels[l], self.num_chunks, w, h)).cuda()
            P_seq[l] = Variable(torch.zeros(batch_size, self.p_channels[l], self.num_chunks, w, h)).cuda()
            I_seq[l] = Variable(torch.zeros(batch_size, self.i_channels[l], self.num_chunks, w, h)).cuda()
            if l != self.n_layers - 1:
                R_seq[l] = Variable(torch.zeros(batch_size, self.r_channels[l], self.num_chunks, w, h)).cuda()
            w = w//2
            h = h//2
        time_steps = input.size(2) // self.num_chunks
        assert time_steps * self.num_chunks == input.size(2)
        total_error = []

        frame_prediction = []
        for t in range(time_steps):
            if self.output_mode == 'error' or self.extrap_start_time > t * self.num_chunks:
                I = input[:, :, t*self.num_chunks:(t+1)*self.num_chunks]
                I = I.type(torch.cuda.FloatTensor)
            else:  # extrapolation mode - prediction result as next input
                assert len(frame_prediction) != 0
                I = frame_prediction[-1]
            I_seq[0] = I

            # Top down
            for l in reversed(range(self.n_layers)):
                cell = getattr(self, 'cell{}'.format(l))
                if t == 0:
                    P = P_seq[l]
                    hx = (P, P)
                else:
                    hx = H_seq[l]
                if l == self.n_layers - 1:
                    tmp = torch.cat((E_seq[l],
                                     self.downsample(R_seq[l - 1]),
                                     self.downsample(E_seq[l - 1])),
                                    dim=1)
                elif l != 0:
                    tmp = torch.cat((E_seq[l],
                                     self.downsample(R_seq[l - 1]),
                                     self.downsample(E_seq[l - 1]),
                                     self.upsample(P_seq[l + 1])),
                                    dim=1)
                else:
                    tmp = torch.cat((E_seq[l],
                                     self.upsample(P_seq[l + 1])),
                                    dim=1)
                P, hx = cell(tmp, hx)
                P_seq[l] = P
                H_seq[l] = hx

            # Bottom up
            for l in range(self.n_layers):
                conv = getattr(self, 'conv{}'.format(l))
                spconv_e = getattr(self, 'spconv_e{}'.format(l))
                I_hat = conv(P_seq[l])

                if l == 0:
                    frame_prediction.append(I_hat)

                pos = F.relu(I_hat - I)
                neg = F.relu(I - I_hat)
                dE = torch.cat([pos, neg],1)
                dE_seq[l] = dE
                E = spconv_e(dE)
                E_seq[l] = E

                if l < self.n_layers - 1:
                    spconv_r = getattr(self, 'spconv_r{}'.format(l))
                    dI = I_seq[l] - I
                    dR = spconv_r(dI)
                    R = R_seq[l] + dR
                    R_seq[l] = R

                    update_I = getattr(self, 'update_I{}'.format(l))
                    I = update_I(R)
                    I_seq[l + 1] = I
            if self.output_mode == 'error':
                mean_error = torch.cat([torch.mean(e.view(e.size(0), -1), 1, keepdim=True) for e in dE_seq], 1)
                total_error.append(mean_error)

        if self.output_mode == 'error':
            return torch.stack(total_error, 2)  # batch x n_layers x nt
        elif self.output_mode == 'prediction':
            return torch.cat(frame_prediction, dim=2)


class SatLU(nn.Module):

    def __init__(self, lower=0, upper=255, inplace=False):
        super(SatLU, self).__init__()
        self.lower = lower
        self.upper = upper
        self.inplace = inplace

    def forward(self, input):
        return F.hardtanh(input, self.lower, self.upper, self.inplace)

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + ' ('\
            + 'min_val=' + str(self.lower) \
            + ', max_val=' + str(self.upper) \
            + inplace_str + ')'
