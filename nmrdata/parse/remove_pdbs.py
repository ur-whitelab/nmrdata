import tensorflow as tf
import pickle
import numpy as np
from graphnmr import *
import matplotlib.pyplot as plt
import os, sys


if len(sys.argv) != 6:
    print('Usage: [input_filename records] [embeddings] [record_info] [pdb id list] [output_filename without pdbs]')
    exit()
if sys.argv[1] == sys.argv[5]:
    print('Will not overwrite')
    exit()
fn = sys.argv[1]
embeddings = load_embeddings(sys.argv[2])

# now load records to connect pdb ids to record indices
with open(sys.argv[4], 'r') as f:
    pdb_set = set([l.split()[0] for l in f.readlines()])
print(pdb_set)
with open(sys.argv[3], 'r') as f:
    rinfo = [l.split() for l in f.readlines()]
    rinfo = rinfo[1:]
non_matches, matches = [],[]
pdbs_added = set()
for i in range(len(rinfo)):
    if rinfo[i][0].split('.')[0] in pdb_set:
        matches.append(rinfo[i][-3:])
        pdbs_added.add(rinfo[i][0].split('.')[0])
    else:
        matches.append(rinfo[i][-3:])
# only keep model id
matches = [rinfo[i][-3] for i in range(len(rinfo)) ]
non_matches = [rinfo[i][-3] for i in range(len(rinfo)) if rinfo[i][0].split('.')[0] not in pdb_set]

# get number of unique pdbs
print('These were not found in the dataset: ', pdb_set - pdbs_added)

# now we try writing records


init_data_op, data = load_records(fn, batch_size=1)
with tf.Session() as sess, tf.python_io.TFRecordWriter(sys.argv[3],
                                                       options=tf.io.TFRecordCompressionType.GZIP) as writer:
    sess.run(init_data_op)
    try:
        i = 0
        skipped = 0
        while True:
            args = sess.run([data['features'], 
                             data['mask'], 
                             data['nlist'], 
                             data['peaks'],
                             data['class'],
                             data['name'],
                             data['index']])
            args = [a[0] for a in args]
            # make sure index matches
            if data['index'][0] in matches:
                skipped += 1
                continue
            record = make_tfrecord(*args)
            writer.write(record.SerializeToString())
            i += 1
            print('\r.......{}'.format(i), end='')
    except tf.errors.OutOfRangeError:
        print('Dataset complete. Skipped {}'.format(skipped))
        pass


