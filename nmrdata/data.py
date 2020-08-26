import tensorflow as tf
import pickle
import os
import gsd.hoomd
import numpy as np


MAX_ATOM_NUMBER = 256
#MAX_ATOM_NUMBER = 384
#MAX_ATOM_NUMBER = 32
#NEIGHBOR_NUMBER = 8
NEIGHBOR_NUMBER = 16

def load_records(filename, batch_size=1):
    data = tf.data.TFRecordDataset(filename, compression_type='GZIP').map(data_parse)
    data = data.batch(batch_size)
    iterator = tf.data.Iterator.from_structure(data.output_types, data.output_shapes)
    init_op = iterator.make_initializer(data)
    bond_inputs, atom_inputs, peak_inputs, mask_inputs,name_inputs, class_input, record_index = iterator.get_next()
    return init_op, {'features': atom_inputs,
            'nlist': bond_inputs,
            'peaks': peak_inputs,
            'mask': mask_inputs,
            'name': name_inputs,
            'class': class_input,
            'index': record_index}



def count_records(filename, batch_size=32):
    '''Counts the number of records total in the filename
    '''
    tf.reset_default_graph()
    init_data_op, data = load_records(filename, batch_size=batch_size)    
    count = 0

    
    with tf.Session() as sess:
        sess.run(init_data_op)
        try:
            while True:
                _ = sess.run([data['name']])
                count += batch_size
                print('\rCounting records...{}'.format(count), end='')
        except tf.errors.OutOfRangeError:
            print('Dataset complete')
            pass
    return count



def nlist_tf_model(positions, NN, sorted=False):
    M = tf.shape(positions)[0]
    # adjust NN
    NN = tf.minimum(NN, M)
    # Making 3 dim CG nlist
    qexpand = tf.expand_dims(positions, 1)  # one column
    qTexpand = tf.expand_dims(positions, 0)  # one row
    # repeat it to make matrix of all positions
    qtile = tf.tile(qexpand, [1, M, 1])
    qTtile = tf.tile(qTexpand, [M, 1, 1])
    # subtract them to get distance matrix
    dist_mat = qTtile - qtile
    # mask distance matrix to remove things beyond cutoff and zeros
    dist = tf.norm(dist_mat, axis=2)
    mask = (dist >= 5e-4)
    mask_cast = tf.cast(mask, dtype=dist.dtype)
    dist_mat_r = dist * mask_cast + (1 - mask_cast) * 1000
    topk = tf.math.top_k(-dist_mat_r, k=NN, sorted=sorted)
    return tf.stack([-topk.values, tf.cast(topk.indices, tf.float32)], 2)

def write_record_traj(positions, atom_data, mask_data, 
                      nlist, peak_data, residue, atom_names,
                      embedding_dicts):
    snap = gsd.hoomd.Snapshot()
    # subtract 1 because the Z is at the end.
    N = MAX_ATOM_NUMBER - sum(atom_data == embedding_dicts['atom']['X']) - 1
    snap.particles.N = N
    #need keys sorted by index
    atom_types = list(embedding_dicts['atom'].items())
    atom_types.sort(key=lambda x: x[1])
    snap.particles.types = [x[0] for x in atom_types]
    snap.particles.typeid = atom_data[:N]
    # need to recenter, convert and compute box size for positions
    trans_pos = positions[:N] * 10
    trans_pos -= np.mean(trans_pos, axis=0)
    box = np.max(trans_pos, axis=0) - np.min(trans_pos, axis=0)
    snap.particles.position = trans_pos
    snap.configuration.box = list(box) + [0, 0, 0]
    snap.particles.charge = peak_data[:N] * mask_data[:N]
    
    #process bonds    
    bond_types = list(embedding_dicts['nlist'].items())
    bond_types.sort(key=lambda x: x[1])
    #need to translate to single chars
    bt_trans = {'none': 'X', 'nonbonded': 'B', 1: 'S', 2: '2', 3:'3', 4:'4', 5:'5', 6:'6'}
    snap.bonds.types = [bt_trans[x[0]] for x in bond_types]
    bonds = np.hstack((np.repeat(np.arange(N), NEIGHBOR_NUMBER).reshape(-1, 1), nlist[:N,:,1].reshape(-1, 1))).astype(np.int)
    bond_ids = nlist[:N,:,2].reshape(-1)
    # filter out duplicate bonds
    bond_ids = bond_ids[bonds[:,0] > bonds[:,1]]
    bonds = bonds[bonds[:,0] > bonds[:,1],:]
    Nb = bonds.shape[0]
    snap.bonds.N = Nb
    snap.bonds.group = bonds
    snap.bonds.typeid = bond_ids
    return snap


def nlist_model(NN, sess):
    # creates a function we call to build nlist for different position number
    p = tf.placeholder(tf.float32, shape=[None, 3], name='positions')
    nlist = nlist_tf_model(p, NN)
    def compute(positions):
        return sess.run(nlist, feed_dict={p: positions})
    return compute

def data_parse(proto):
    features = {'bond-data': tf.FixedLenFeature([MAX_ATOM_NUMBER, NEIGHBOR_NUMBER, 3], tf.float32),
                'atom-data': tf.FixedLenFeature([MAX_ATOM_NUMBER], tf.int64),
                'peak-data': tf.FixedLenFeature([MAX_ATOM_NUMBER], tf.float32),
                'mask-data': tf.FixedLenFeature([MAX_ATOM_NUMBER], tf.float32),
                'name-data': tf.FixedLenFeature([MAX_ATOM_NUMBER], tf.int64),
                'residue': tf.FixedLenFeature([1], tf.int64),
                'indices': tf.FixedLenFeature([3], tf.int64)
               }
    parsed_features = tf.parse_single_example(proto, features) 
    return (parsed_features['bond-data'], 
           parsed_features['atom-data'], 
           parsed_features['peak-data'], 
           parsed_features['mask-data'], 
           parsed_features['name-data'],
           parsed_features['residue'],
           parsed_features['indices'])

def create_datasets(filenames, skips, swap=False):
    '''Swap is used if you want to plot the training data, instead of usual validation. 
    Returns (train, validation)
    '''
    datasets = []
    for f,s in zip(filenames, skips):
        d = tf.data.TFRecordDataset([f], compression_type='GZIP').map(data_parse)
        if swap:
            datasets.append( (d.take(s), d.skip(s)) )
        else:
            datasets.append( (d.skip(s), d.take(s)) )

    return datasets

def make_tfrecord(atom_data, mask_data, nlist, peak_data, residue, atom_names, weights=None, indices=np.zeros((3,1), dtype=np.int64)):
    '''
    Write out the TF record.
      
      atom_data - N ints containing atom indixes
      mask_data - N floats containing 1/0 for which atoms are begin considered
      nlist     - N x M x 3 floats neighbor list:
                    :,:,0 -> distance
                    :,:,1 -> neighbor index
                    :,:,2 -> bond count/type
      peak_data - N containing peak data for training (0 for prediction)
      residue     - N ints indicating the residue of the atom (for validation)
      atom_names  - N ints indicating the name of the atom  (for validation)
      indices       - 3 ints indices to attach to record (for validation)
    '''
    features = {}
    # nlist

    assert atom_data.shape[0] == MAX_ATOM_NUMBER
    assert mask_data.shape[0] == MAX_ATOM_NUMBER
    assert nlist.shape[0] == MAX_ATOM_NUMBER and nlist.shape[1] == NEIGHBOR_NUMBER and nlist.shape[2] == 3
    assert peak_data.shape[0] == MAX_ATOM_NUMBER
    assert atom_names.shape[0] == MAX_ATOM_NUMBER
    assert len(indices) == 3
    if np.any(np.isnan(peak_data)):
        raise ValueError('Found nan in your data!')
        # Use code below if you do not care about nans
        peak_data[np.isnan(peak_data)] = 0
        mask_data[np.isnan(peak_data)] = 0
    if np.any(np.abs(peak_data) > 10000):
        raise ValueError('Found very large peaks, |v| > 10000')
    features['bond-data'] = tf.train.Feature(float_list=tf.train.FloatList(value=nlist.flatten()))
    features['atom-data'] = tf.train.Feature(int64_list=tf.train.Int64List(value=atom_data.flatten()))
    features['peak-data'] = tf.train.Feature(float_list=tf.train.FloatList(value=peak_data.flatten()))
    features['mask-data'] = tf.train.Feature(float_list=tf.train.FloatList(value=mask_data.flatten()))
    features['name-data'] = tf.train.Feature(int64_list=tf.train.Int64List(value=atom_names.flatten()))
    features['residue'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[residue]))
    features['indices'] = tf.train.Feature(int64_list=tf.train.Int64List(value=indices.flatten()))
    # Make training example
    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example

def load_embeddings(path):
    if not os.path.exists(path):
        embedding_dicts = {}
    else:
        with open(path, 'rb') as f:
            embedding_dicts = pickle.load(f) 
    if 'atom' not in embedding_dicts:
        elements = ['X', 'Z'] # no atom (X), other atom Z
        atom_dictionary = dict(zip(elements, range(len(elements))))
        embedding_dicts['atom'] = atom_dictionary
    if 'bond' not in embedding_dicts:
        bond_dictionary = dict(zip(['none', 'single', 'double', 'triple', 'aromatic','self'], range(5)))
        embedding_dicts['bond'] = bond_dictionary
    if 'exp' not in embedding_dicts:
        exp_dictionary = {'': 0}
        embedding_dicts['exp'] = exp_dictionary
    if 'class' not in embedding_dicts:
        aas = ['ala','arg','asn','asp','cys','glu','gln','gly','his', 'ile', 'leu','lys','met','phe','pro','ser','thr','trp','tyr','val', 'MB']
        aas = [s.upper() for s in aas]
        embedding_dicts['class'] = dict(zip(aas, range(len(aas))))
    if 'nlist' not in embedding_dicts:
        nlist = ['none', 'nonbonded'] + list(1)
        embedding_dicts['nlist'] = dict(zip(nlist, range(len(nlist))))
    if 'name' not in embedding_dicts:
        embedding_dicts['name'] = {'X': 0}

    return embedding_dicts

def save_embeddings(embeddings, path):
    with open(path, 'wb') as f:
        pickle.dump(embeddings, f)
