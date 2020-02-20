#!/usr/bin/env python

import sys
import cPickle

def load_batch(fpath, label_key='labels'):
    with open(fpath,'rb') as f:
        d = cPickle.load(f)

    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)

    return data, labels


if __name__=='__main__':

    path = 'cifar-10-batches-py/'
    file = 'data_batch_1'

    data, labels = load_batch(path+file)

    print(data, labels)
