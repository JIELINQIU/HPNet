import hickle as hkl
import pickle as pkl
import torch
import torch.utils.data as data
import scipy.ndimage as simg


class DATA(data.Dataset):
    def __init__(self, datafile, sourcefile, nt):
        self.datafile = open(datafile + '.pkl', mode='rb')
        self.sourcefile = open(sourcefile + '.pkl', mode='rb')
        self.X = pkl.load(self.datafile, encoding='bytes')
        self.sources = pkl.load(self.sourcefile, encoding='bytes')

        self.nt = nt
        cur_loc = 0
        possible_starts = []
        while cur_loc < self.X.shape[0] - self.nt + 1:
            if self.sources[cur_loc] == self.sources[cur_loc + self.nt - 1]:
                possible_starts.append(cur_loc)
                cur_loc += self.nt
            else:
                cur_loc += 1
        self.possible_starts = possible_starts

    def __getitem__(self, index):
        loc = self.possible_starts[index]
        img = self.X[loc:loc + self.nt] / 255
        zoom_fh = 64 / img.shape[1]
        zoom_fw = 64 / img.shape[2]
        img = simg.zoom(img, [1., zoom_fh, zoom_fw, 1.])
        assert img.shape[1] == 64 and img.shape[2] == 64 and img.shape[3] == 3
        return img

    def __len__(self):
        return len(self.possible_starts)
