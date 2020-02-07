
import os
import PIL
import numpy as np
from imageio import imread
from scipy.misc import imresize
import hickle as hkl

from PIL import Image

DATA_DIR = './KTH/KTH_data/'

desired_im_sz = (64, 64)
categories = ['walking','running','handwaving','jogging','handclapping','boxing']

test_recordings = [('walking', 'person25_walking_d1'),('walking', 'person24_walking_d1'),('walking', 'person23_walking_d1'),('walking', 'person22_walking_d1'),\
('running', 'person21_running_d1'),('running', 'person20_running_d1'),('running', 'person19_running_d1'),('running', 'person18_running_d1'),\
('handwaving', 'person17_handwaving_d1'),('handwaving', 'person16_handwaving_d1'),('handwaving', 'person15_handwaving_d1'),('handwaving', 'person14_handwaving_d1'),\
('jogging', 'person13_jogging_d1'),('jogging', 'person12_jogging_d1'),('jogging', 'person11_jogging_d1'),('jogging', 'person10_jogging_d1'),\
('handclapping', 'person09_handclapping_d1'),('handclapping', 'person08_handclapping_d1'),('handclapping', 'person07_handclapping_d1'),('handclapping', 'person06_handclapping_d1'),\
('boxing', 'person05_boxing_d1'),('boxing', 'person04_boxing_d1'),('boxing', 'person03_boxing_d1'),('boxing', 'person02_boxing_d1')]
val_recordings = [('walking', 'person01_walking_d1'),\
('running', 'person02_running_d1'),('handwaving', 'person03_handwaving_d1'),\
('jogging', 'person04_jogging_d1'),('handclapping', 'person05_handclapping_d1'),('boxing', 'person06_boxing_d1')]

if not os.path.exists(DATA_DIR): os.mkdir(DATA_DIR)


def download_data():
    base_dir = os.path.join(DATA_DIR, 'raw/')
    if not os.path.exists(base_dir): os.mkdir(base_dir)
    for c in categories:
        print( "Downloading set: " + c)
        c_dir = base_dir + c + '/'
        if not os.path.exists(c_dir): os.mkdir(c_dir)

def extract_data():
    for c in categories:
        c_dir = os.path.join(DATA_DIR, 'raw/', c + '/')
        files = list(os.walk(c_dir, topdown=False))[-1][-1]
        for f in files:
            spec_folder = f[:19] + '\\' 
            os.system(command)

def process_data():
    splits = {s: [] for s in ['train', 'test', 'val']}
    splits['val'] = val_recordings
    splits['test'] = test_recordings
    not_train = splits['val'] + splits['test']
    for c in categories:
        c_dir = os.path.join(DATA_DIR, 'raw', c + '/')
        folders= list(os.walk(c_dir, topdown=False))[-1][-2]
        splits['train'] += [(c, f) for f in folders if (c, f) not in not_train]

    for split in splits:
        im_list = []
        source_list = []
        for category, folder in splits[split]:
            im_dir = os.path.join(DATA_DIR, 'raw/', category, folder, '')
            files = list(os.walk(im_dir, topdown=False))[-1][-1]
            im_list += [im_dir + f for f in sorted(files)]
            source_list += [category + '-' + folder] * len(files)

        print( 'Creating ' + split + ' data: ' + str(len(im_list)) + ' images')
        X = np.zeros((len(im_list),) + desired_im_sz + (3,), np.uint8)
        for i, im_file in enumerate(im_list):
            im = imread(im_file)
            X[i] = process_im(im, desired_im_sz)

        hkl.dump(X, os.path.join(DATA_DIR, 'X_' + split + '.hkl'))
        hkl.dump(source_list, os.path.join(DATA_DIR, 'sources_' + split + '.hkl'))

def process_im(im, desired_sz):
    target_ds = float(desired_sz[0])/im.shape[0]
    im = imresize(im, (desired_sz[0], int(np.round(target_ds * im.shape[1]))))
    d = int((im.shape[1] - desired_sz[1]) / 2)
    im = im[:, d:d+desired_sz[1]]
    return im


if __name__ == '__main__':
    download_data()
    extract_data()
    process_data()
