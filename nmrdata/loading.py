import warnings
import MDAnalysis as md
from nmrdata import *
import tensorflow as tf
import pickle
import os
import numpy as np
import click
import pickle


def load_records(filename):
    data = tf.data.TFRecordDataset(
        filename, compression_type='GZIP').map(data_parse)
    return data.map(data_parse_dict)


def data_parse_dict(*record):
    atom_number, neighbor_number, bond_inputs, atom_inputs, peak_inputs, mask_inputs, name_inputs, class_input, record_index = record
    return {'natoms': atom_number,
            'nneigh': neighbor_number,
            'features': atom_inputs,
            'nlist': bond_inputs,
            'peaks': peak_inputs,
            'mask': mask_inputs,
            'name': name_inputs,
            'class': class_input,
            'index': record_index}


@tf.function(experimental_compile=True)
def nlist_model(positions, NN, sorted=False):
    M = tf.shape(input=positions)[0]
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
    dist = tf.norm(tensor=dist_mat, axis=2)
    mask = (dist >= 5e-4)
    mask_cast = tf.cast(mask, dtype=dist.dtype)
    dist_mat_r = dist * mask_cast + (1 - mask_cast) * 1000
    topk = tf.math.top_k(-dist_mat_r, k=NN, sorted=sorted)
    return tf.stack([-topk.values, tf.cast(topk.indices, tf.float32)], 2)


def write_record_traj(positions, atom_data, mask_data,
                      nlist, peak_data, residue, atom_names,
                      embedding_dicts):
    import gsd.hoomd
    snap = gsd.hoomd.Snapshot()
    N = sum(atom_data != embedding_dicts['atom']['X'])
    snap.particles.N = N
    # need keys sorted by index
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

    # process bonds
    bond_types = list(embedding_dicts['nlist'].items())
    bond_types.sort(key=lambda x: x[1])
    # need to translate to single chars
    bt_trans = {'none': 'X', 'nonbonded': 'B', 1: 'S',
                2: '2', 3: '3', 4: '4', 5: '5', 6: '6'}
    snap.bonds.types = [bt_trans[x[0]] for x in bond_types]
    neighbor_number = nlist.shape[1]
    bonds = np.hstack((np.repeat(np.arange(N), neighbor_number).reshape(-1,
                                                                        1), nlist[:N, :, 1].reshape(-1, 1))).astype(np.int)
    bond_ids = nlist[:N, :, 2].reshape(-1)
    # filter out duplicate bonds
    bond_ids = bond_ids[bonds[:, 0] > bonds[:, 1]]
    bonds = bonds[bonds[:, 0] > bonds[:, 1], :]
    Nb = bonds.shape[0]
    snap.bonds.N = Nb
    snap.bonds.group = bonds
    snap.bonds.typeid = bond_ids
    return snap


def data_parse(proto):
    features = {
        'atom-number': tf.io.FixedLenFeature([], tf.int64),
        'neighbor-number': tf.io.FixedLenFeature([], tf.int64),
        'bond-data': tf.io.VarLenFeature(tf.float32),
        'atom-data': tf.io.VarLenFeature(tf.int64),
        'peak-data': tf.io.VarLenFeature(tf.float32),
        'mask-data': tf.io.VarLenFeature(tf.float32),
        'name-data': tf.io.VarLenFeature(tf.int64),
        'residue': tf.io.FixedLenFeature([1], tf.int64),
        'indices': tf.io.FixedLenFeature([3], tf.int64)
    }
    parsed_features = tf.io.parse_single_example(
        serialized=proto, features=features)
    # convert our features from sparse to dense
    atom_number = parsed_features['atom-number']
    neighbor_number = parsed_features['neighbor-number']
    bonds = tf.reshape(tf.sparse.to_dense(
        parsed_features['bond-data'], default_value=0), (atom_number, neighbor_number, 3))

    atoms = tf.sparse.to_dense(
        parsed_features['atom-data'], default_value=0)
    peaks = tf.sparse.to_dense(
        parsed_features['peak-data'], default_value=0)
    mask = tf.sparse.to_dense(
        parsed_features['mask-data'], default_value=0)
    names = tf.sparse.to_dense(
        parsed_features['name-data'], default_value=0)

    return (parsed_features['atom-number'],
            parsed_features['neighbor-number'],
            bonds,
            atoms,
            peaks,
            mask,
            names,
            parsed_features['residue'],
            parsed_features['indices'])


def dataset(tfrecords, embeddings=None, label_info=False, short_records=True):
    '''Create iterator over tfrecords
    '''
    d = tf.data.TFRecordDataset(
        tfrecords, compression_type='GZIP').map(data_parse)
    if short_records:
        d = d.map(lambda *x: data_shorten(*x,
                                          embeddings=embeddings, label_info=label_info))
    return d


def data_shorten(*args, embeddings, label_info=False):
    embeddings = load_embeddings(embeddings)
    N = args[0]
    NN = args[1]
    nlist_full = args[2]
    nodes = args[3]
    labels = args[4]
    mask = args[5]
    names = args[6]
    edges = nlist_full[:, :, 0]
    inv_degree = tf.squeeze(tf.math.divide_no_nan(1.,
                                                  tf.reduce_sum(tf.cast(nlist_full[:, :, 0] > 0, tf.float32), axis=1)))
    nlist = tf.cast(nlist_full[:, :, 1], tf.int32)
    nodes = tf.one_hot(nodes, len(embeddings['atom']))

    if label_info:
        return (nodes, nlist, edges, inv_degree), tf.stack([labels, tf.cast(names, labels.dtype), mask], axis=1), mask
    return (nodes, nlist, edges, inv_degree), labels, mask


def make_tfrecord(atom_data, mask_data, nlist, peak_data, residue, atom_names, weights=None, indices=np.zeros((3, 1), dtype=np.int64)):
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
    N = atom_data.shape[0]
    NN = nlist.shape[1]
    assert mask_data.shape[0] == N
    assert nlist.shape[0] == N and nlist.shape[2] == 3
    assert peak_data.shape[0] == N
    assert atom_names.shape[0] == N
    assert len(indices) == 3
    if np.any(np.isnan(peak_data)):
        raise ValueError('Found nan in your data!')
        # Use code below if you do not care about nans
        peak_data[np.isnan(peak_data)] = 0
        mask_data[np.isnan(peak_data)] = 0
    if np.any(np.abs(peak_data) > 10000):
        raise ValueError('Found very large peaks, |v| > 10000')
    features['atom-number'] = tf.train.Feature(
        int64_list=tf.train.Int64List(value=[N]))
    features['neighbor-number'] = tf.train.Feature(
        int64_list=tf.train.Int64List(value=[NN]))
    features['bond-data'] = tf.train.Feature(
        float_list=tf.train.FloatList(value=nlist.flatten()))
    features['atom-data'] = tf.train.Feature(
        int64_list=tf.train.Int64List(value=atom_data.flatten()))
    features['peak-data'] = tf.train.Feature(
        float_list=tf.train.FloatList(value=peak_data.flatten()))
    features['mask-data'] = tf.train.Feature(
        float_list=tf.train.FloatList(value=mask_data.flatten()))
    features['name-data'] = tf.train.Feature(
        int64_list=tf.train.Int64List(value=atom_names.flatten()))
    features['residue'] = tf.train.Feature(
        int64_list=tf.train.Int64List(value=[residue]))
    features['indices'] = tf.train.Feature(
        int64_list=tf.train.Int64List(value=indices.flatten()))
    # Make training example
    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example


def _load_embeddings():
    from importlib_resources import files
    import nmrdata.data
    fp = files(nmrdata.data).joinpath(
        'embeddings.pb')
    with fp.open('rb') as f:
        e = pickle.load(f)
    return e


def load_embeddings(path=None):

    if path is None:
        return _load_embeddings()

    if not os.path.exists(path):
        raise Warning('Are you really sure you want to remake embeddings??')
        embedding_dicts = {}
    else:
        with open(path, 'rb') as f:
            embedding_dicts = pickle.load(f)
    if 'atom' not in embedding_dicts:
        elements = ['X', 'Z']  # no atom (X), other atom Z
        atom_dictionary = dict(zip(elements, range(len(elements))))
        embedding_dicts['atom'] = atom_dictionary
    if 'bond' not in embedding_dicts:
        bond_dictionary = dict(
            zip(['none', 'single', 'double', 'triple', 'aromatic', 'self'], range(5)))
        embedding_dicts['bond'] = bond_dictionary
    if 'exp' not in embedding_dicts:
        exp_dictionary = {'': 0}
        embedding_dicts['exp'] = exp_dictionary
    if 'class' not in embedding_dicts:
        aas = ['ala', 'arg', 'asn', 'asp', 'cys', 'glu', 'gln', 'gly', 'his', 'ile',
               'leu', 'lys', 'met', 'phe', 'pro', 'ser', 'thr', 'trp', 'tyr', 'val', 'MB']
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


def load_standards():
    from importlib_resources import files
    import nmrdata.data
    fp = files(nmrdata.data).joinpath(
        'standards.pb')
    with fp.open('rb') as f:
        e = pickle.load(f)
    return e


def _oldstyle_mda(pairs, pair_distances, N):
    '''Need to work around change in MDAnalysis neighbor lists. Unfortunate.'''
    ragged_nlist = [[] for _ in range(N)]
    ragged_edges = [[] for _ in range(N)]
    for p, d in zip(pairs, pair_distances):
        ragged_nlist[p[0]].append(p[1])
        ragged_nlist[p[1]].append(p[0])
        ragged_edges[p[0]].append(d)
        ragged_edges[p[1]].append(d)
    return ragged_nlist, ragged_edges


def parse_universe(u, neighbor_number, embeddings, cutoff=None, pbc=None, warn=True):
    '''Converts universe into atoms, edges, nlist
    '''
    N = u.atoms.positions.shape[0]
    if pbc is None:
        pbc = sum(u.dimensions**2) > 0
        warnings.warn(f'Gussing system is{"" if pbc else " not"} pbc')
    new_embeddings = False
    dimensions = u.dimensions
    if cutoff is None:
        cutoff = min(dimensions) / 2.01
        if cutoff == 0:
            # no box defined
            bbox = u.atoms.bbox()
            dimensions = bbox[1] - bbox[0]
            cutoff = min(dimensions) / 2.01
            # make it into proper dimensions
            dimensions = np.array(list(dimensions) + [90, 90, 90])
            u.atoms.wrap(box=dimensions)
            if warn:
                warnings.warn(
                    'Guessing the system dimensions are' + str(dimensions))
    gridsearch = md.lib.nsgrid.FastNS(
        cutoff, u.atoms.positions, dimensions, pbc=pbc)
    results = gridsearch.self_search()
    ragged_nlist, ragged_edges = _oldstyle_mda(
        results.get_pairs(), results.get_pair_distances(), N)
    nlist = np.zeros((N, neighbor_number), dtype=np.int32)
    edges = np.zeros((N, neighbor_number), dtype=np.float32)
    atoms = np.zeros(N, dtype=np.int32)
    # check for elements
    try:
        elements = u.atoms.elements
    except md.exceptions.NoDataError as e:
        if warn:
            warnings.warn('Trying to guess elements from names')
        elements = []
        for i in range(N):
            # find first non-digit character
            elements.append([n for n in u.atoms[i].name if not n.isdigit()][0])
    for i in range(N):
        # sort them
        order = np.argsort(ragged_edges[i])
        nl = np.array(ragged_nlist[i])[order][:neighbor_number]
        el = np.array(ragged_edges[i])[order][:neighbor_number]
        nlist[i, :len(nl)] = nl
        edges[i, :len(nl)] = el
        try:
            atoms[i] = embeddings['atom'][elements[i]]
        except KeyError as e:
            if warn:
                print('Unparameterized element <' +
                      elements[i] + '> will replace with unknown atom')
            atoms[i] = 1
            embeddings['name'][elements[i]] = len(embeddings['name'])
            new_embeddings = True
    edges /= 10  # angstrom to nm. TODO: need more reliable check
    # note we convert atoms to be one hot
    if new_embeddings:
        nef = 'new-embeddings.pb'
        if not os.path.exists(nef):
            print('Writing modified emebddings as new_embeddings')
            save_embeddings(embeddings, 'new-embeddings.pb')
        elif warn:
            print('Will not write modified embeddings because file exists')
    return tf.one_hot(atoms, len(embeddings['atom'])), edges, nlist
