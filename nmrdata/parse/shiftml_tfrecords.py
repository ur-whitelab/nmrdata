import tensorflow as tf
from graphnmr import *
import numpy as np
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
import io
import tqdm

def padto(a, shape):
    pad = tuple((0,shape[i] - a.shape[i]) for i in range(len(shape)))
    if pad[0][1] < 0:
        return None
    return np.pad(a, pad, 'constant')

vdw = {
    'H': 1,
    'C': 1,
    'O': 1,
    'P': 1,
    'N': 1,
    'Cl': 1
}

def guess_bonds(pos_nlist, atom_names):
    bonds = np.zeros( (BOND_MAX, MAX_ATOM_NUMBER,MAX_ATOM_NUMBER), dtype=np.int64)
    for i in range(len(atom_names)):
        for ni

def adj_to_nlist(atoms, A, nlist_model, embeddings):
    bonds = {1: Chem.rdchem.BondType.SINGLE,
            2: Chem.rdchem.BondType.DOUBLE,
            3: Chem.rdchem.BondType.TRIPLE}
    m = Chem.EditableMol(Chem.Mol())
    for a in atoms:
        m.AddAtom(Chem.Atom(a))
    for i in range(len(atoms)):
        for j in range(i, len(atoms)):
            if A[i,j] > 0:
                m.AddBond(i, j, bonds[A[i,j]])
    mol = m.GetMol()
    try:
        AllChem.EmbedMolecule(mol)
    except ValueError as e:
        print('Unable to process')
        print(Chem.MolToSmiles(mol))
        raise e
    for c in mol.GetConformers():
        pos = c.GetPositions()
        N = len(pos)
        np_pos = np.zeros( ( max(N, NEIGHBOR_NUMBER), 3))
        np_pos[:N, :] = pos
        pos_nlist = nlist_model(np_pos)
        nlist = np.zeros( (MAX_ATOM_NUMBER, NEIGHBOR_NUMBER, 3) )


        # compute bond distances
        # bonds contains 1 neighbors, 2 neighbors, etc where "1" means 1 bond away and "2" means two bonds away
        bonds = np.zeros( (BOND_MAX, MAX_ATOM_NUMBER,MAX_ATOM_NUMBER), dtype=np.int64)
        # need to rebuild adjacency matrix with new atom ordering
        for b in mol.GetBonds():
            bonds[0, b.GetBeginAtomIdx(), b.GetEndAtomIdx()] = 1
            bonds[0, b.GetEndAtomIdx(), b.GetBeginAtomIdx()] = 1
        for bi in range(1, BOND_MAX):
            bonds[bi, :, :] = (bonds[0, :, :] @ bonds[bi - 1, :, :]) > 0

        # a 0 -> non-bonded
        for index in range(N):
            for ni in range(NEIGHBOR_NUMBER):
                j = int(pos_nlist[index, ni, 1])
                # / 10 to get to nm
                nlist[index, ni, 0] = pos_nlist[index, ni, 0] / 10
                nlist[index, ni, 1] = j
                # a 0 -> non-bonded
                if sum(bonds[:, index, j]) == 0:
                    nlist[index,ni,2] = embeddings['nlist']['nonbonded']
                else:
                    # add 1 so index 0 -> single bonded
                    bond_dist = (bonds[:, index, j] != 0).argmax(0) + 1
                    nlist[index,ni,2] = embeddings['nlist'][bond_dist]

        for index in range(N, MAX_ATOM_NUMBER):
            for ni in range(NEIGHBOR_NUMBER):
                nlist[index, ni, 0] = 0
                nlist[index, ni, 1] = MAX_ATOM_NUMBER - 1
                nlist[index, ni, 2] = embeddings['nlist']['none']
        yield nlist

def parse_xyz(ifile, embeddings):
    while True:
        line = ifile.readline()
        if not line:
           break
        N = int(line)
        name = ifile.readline().split('NAME=')[1]
        atom_strings = []
        atom_names = np.empty( (N), dtype=np.int64)
        pos = np.empty( (N, 3), dtype=np.float)
        peaks = np.empty( (N), dtype=np.float)
        for i in range(N):
            sline = ifile.readline().split()
            e = sline[0]
            atom_strings.append(e)
            atom_name = 'DFT-' + e
            if atom_name not in embeddings['name']:
                embeddings['name'][atom_name] = len(embeddings['name'])
            atom_names[i] = embeddings['name'][atom_name]
            pos[i, :] = [float(s) for s in sline[1:4]]
            peaks[i] = float(sline[4])
        yield name, atom_strings, atom_names, pos, peaks


DATA_DIR = 'data' + os.sep

embeddings = load_embeddings('embeddings.pb')

files = [DATA_DIR + 'shiftml' + os.sep + 'CSD-2k.xyz', DATA_DIR + 'shiftml' + os.sep + 'CSD-500.xyz']

# turn off GPU for more memory
config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
with tf.python_io.TFRecordWriter('train-structure-metabolite-data-{}-{}.tfrecord'.format(MAX_ATOM_NUMBER, NEIGHBOR_NUMBER),
                                 options=tf.io.TFRecordCompressionType.GZIP) as writer:
    with tf.Session(config=config) as sess:
        nm = nlist_model(NEIGHBOR_NUMBER, sess)
        pbar = tqdm.tqdm()
        for fn in files:
            with open(fn, 'r') as f:
                for name, atom_strings, atom_names, pos, peaks in parse_xyz(f, embeddings):

                    pos_nlist = nm(pos)
                    bonds = guess_bonds(pos_nlist, atom_strings)
                    exit()
            bond_data = padto(prepare_adj(rd), (MAX_ATOM_NUMBER, MAX_ATOM_NUMBER))
            if bond_data is None:
                #bigger than max atom number
                continue
            atom_data, heavies, atoms, names = prepare_features(rd, embeddings)
            class_label = 'MB'
            if class_label not in embeddings['class']:
                embeddings['class'][class_label] = len(embeddings['class'])
            atom_data = padto(atom_data, (MAX_ATOM_NUMBER,))
            peak_data = padto(prepare_labels(rd), (MAX_ATOM_NUMBER, ))
            name_data = padto(names, (MAX_ATOM_NUMBER,))
            nucleus = rd['nucleus'][-1]
            mask_data = (peak_data != 0) * 1.0
            try:
                for ci, nlist in enumerate(adj_to_nlist(atoms, bond_data, nm, embeddings)):
                    pbar.set_description('{}:{}'.format(class_label,ci))
                    record = make_tfrecord(atom_data, mask_data, nlist, peak_data, embeddings['class'][class_label], name_data)
                    writer.write(record.SerializeToString())
            except ValueError:
                continue
save_embeddings(embeddings, 'embeddings.pb')