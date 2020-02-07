import os
import sys
import hickle as hkl
import pickle as pkl


f='X_train.hkl'
a=hkl.load(f)
print(f, type(a))
newf = f+'.pkl'
ff = open(newf, mode='wb')
pkl.dump(a, ff, protocol=0)