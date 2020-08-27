import tensorflow as tf
import pickle
import numpy as np
from graphnmr import *
import os
import sys
from nmrdata import *
from graphnmr import *

if len(sys.argv) != 3:
    print('Usage: [input_filename] [embeddings]')
    exit()

fn = sys.argv[1]
embeddings = load_embeddings(sys.argv[2])
# [count, mean, std]
standards = {k: [0., 0., 0.] for k in embeddings['atom'].values()}


init_data_op, data = load_records(fn, batch_size=1)
with tf.Session() as sess:
    sess.run(init_data_op)
    try:
        i = 0
        while True:
            args = sess.run([data['features'],
                             data['mask'],
                             data['peaks']])
            # slice out so we don't have batch index
            args = [a[0] for a in args]
            # overwrite mask with weights
            # taking the sklearn equation
            # convert mask to binary in case it's weighted
            args[1] = (args[1] > 1e-10) * 1
            for k, v in standards.items():
                m = (args[0] == k) * args[1]
                c = np.sum(m)
                if c == 0:
                    continue
                v[0] += c
                d = np.sum((args[2] - v[1]) * m)
                v[1] += d / c
                v[2] += d * d
            i += 1
            print('\r.......{}'.format(i), end='')
    except tf.errors.OutOfRangeError:
        print('Dataset complete')
        pass


for k, v in standards.items():
    if v[0] == 0:
        continue
    else:
        # final computation of stddev
        v[2] /= v[0]
        v[2] = np.sqrt(v[2])

with open('peak_standards.pb', 'wb') as f:
    pickle.dump(standards, f)

rv = {v: k for k, v in embeddings['atom'].items()}
for k, v in standards.items():
    print(rv[k], v)
