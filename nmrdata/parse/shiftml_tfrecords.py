import tensorflow as tf
import numpy as np
import pickle
import io
import tqdm
from nmrdata import *

def make_nlist(pos, embeddings, neighbor_number):
    # all nlists are non-bonded here
    N = pos.shape[0]
    pos_nlist = nlist_model(pos, neighbor_number)
    nlist = np.zeros( (N, neighbor_number, 3) , dtype=np.float32)
    # a 0 -> non-bonded
    for index in range(N):
        for ni in range(len(pos_nlist[index])):
            if pos_nlist[index, ni, 0] >= 100: # this is a large distance sentinel indicating not part of nlist
                continue
            j = int(pos_nlist[index, ni, 1])
            # / 10 to get to nm
            nlist[index, ni, 0] = pos_nlist[index, ni, 0] / 10
            nlist[index, ni, 1] = j
            nlist[index,ni,2] = embeddings['nlist']['nonbonded']
        # pad out the nlist
        for index in range(N, N):
            for ni in range(neighbor_number):
                nlist[index, ni, 0] = 0
                nlist[index, ni, 1] = 0
                nlist[index, ni, 2] = embeddings['nlist']['none']
    return nlist


def parse_xyz(ifile, embeddings):
    while True:
        line = ifile.readline()
        if not line:
            break
        N = int(line)
        name = ifile.readline().split('NAME=')[1]
        features = np.empty((N), dtype=np.int64)
        atom_names = np.empty((N), dtype=np.int64)
        pos = np.empty((N, 3), dtype=np.float32)
        peaks = np.empty((N), dtype=np.float32)
        for i in range(N):
            sline = ifile.readline().split()
            e = sline[0]
            if e not in embeddings['atom']:
                embeddings['atom'][e] = len(embeddings['atom'])
            features[i] = embeddings['atom'][e]
            atom_name = 'DFT-' + e
            if atom_name not in embeddings['name']:
                print('*******Adding new atom name*********')
                embeddings['name'][atom_name] = len(embeddings['name'])
            atom_names[i] = embeddings['name'][atom_name]
            pos[i, :] = [float(s) for s in sline[1:4]]
            peaks[i] = float(sline[4])
        yield features, atom_names, pos, peaks

@click.command()
@click.argument('xyz_file')
@click.argument('output_name')
@click.option('--embeddings', default=None, help='path to custom embeddings file')
@click.option('--neighbor-number', default=16, help='The model specific size of neighbor lists')
def parse_shiftml(xyz_file, output_name, embeddings, neighbor_number):

    embeddings = load_embeddings(embeddings)
    with tf.io.TFRecordWriter(f'shiftml-{output_name}.tfrecord',
                                     options=tf.io.TFRecordOptions(compression_type='GZIP')) as writer,\
        open(xyz_file) as f:
        successes = 0
        pbar = tqdm.tqdm(parse_xyz(f, embeddings))
        for rd in pbar:
            atom_data, name_data, pos, peaks = rd
            class_label = 'DFT'
            if class_label not in embeddings['class']:
                embeddings['class'][class_label] = len(embeddings['class'])
            mask_data = np.ones_like(atom_data).astype(np.float32)
            nlist = make_nlist(pos, embeddings, neighbor_number)
            pbar.set_description('DFT with {} atoms. Successes: {}'.format(len(atom_data), successes))
            record = make_tfrecord(atom_data, mask_data, nlist, peaks, embeddings['class'][class_label], name_data)
            writer.write(record.SerializeToString())
            successes += 1
    save_embeddings(embeddings, 'final-embeddings.pb')

