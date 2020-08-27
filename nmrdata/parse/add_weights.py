import tensorflow as tf
import pickle
import numpy as np
from graphnmr import *
import os
import sys
from graphnmr import *

if len(sys.argv) != 4:
    print('Usage: [input_filename] [embeddings] [output_filename]')
    exit()
if sys.argv[1] == sys.argv[3]:
    print('Will not overwrite')
    exit()
fn = sys.argv[1]
embeddings = load_embeddings(sys.argv[2])

name_counts = count_names(fn, embeddings)


# now we try writing records


total = sum(name_counts)
class_number = sum([1 if x > 0 else 0 for x in name_counts])

print('Found {} classes out of {} names. Total counts is {}'.format(
    class_number, len(name_counts), total))

init_data_op, data = load_records(fn, batch_size=1)
with tf.Session() as sess, tf.python_io.TFRecordWriter(sys.argv[3],
                                                       options=tf.io.TFRecordCompressionType.GZIP) as writer:
    sess.run(init_data_op)
    try:
        i = 0
        while True:
            args = sess.run([data['features'],
                             data['mask'],
                             data['nlist'],
                             data['peaks'],
                             data['class'],
                             data['name'],
                             data['index']])
            # slice out so we don't have batch index
            args = [a[0] for a in args]
            # overwrite mask with weights
            # taking the sklearn equation
            args[1] = args[1] * (total / class_number /
                                 np.maximum(name_counts[args[5].astype(np.int32)], 1))
            record = make_tfrecord(*args)
            writer.write(record.SerializeToString())
            i += 1
            print('\r.......{}'.format(i), end='')
    except tf.errors.OutOfRangeError:
        print('Dataset complete')
        pass
