import tensorflow as tf
from graphnmr import *
import numpy as np
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
import io
import tqdm
import sys

def prepare_features(entry, embedding_dicts):
    N = len(entry['atoms'])
    names = np.zeros((N), dtype=np.int32)
    atoms = []
    F = np.empty((N), dtype=np.int32)
    heavies = 0
    for i in range(N):
        e = entry['atoms'][i]['@elementType']
        if e != 'H':
            heavies += 1
        if e not in embedding_dicts['atom']:
            embedding_dicts['atom'][e] = len(embedding_dicts['atom'])
        F[i] = embedding_dicts['atom'][e]
        atoms.append(e)
        atom_name = 'MB-' + e
        if atom_name not in embedding_dicts['name']:
            embedding_dicts['name'][atom_name] = len(embedding_dicts['name'])
        names[i] = embedding_dicts['name'][atom_name]
    return F, heavies, atoms, names


def prepare_labels(entry):
    N = len(entry['atoms'])
    L = np.zeros((N), dtype=np.float)
    for i in range(N):
        for p in entry['peaks']:
            if 'a{}'.format(i+1) in p['atoms']['@atomRefs'].split():
                if type(p['peakList']['peak']) == list:
                    #print('duplicate peaks', *[x['@center'] for x in p['peakList']['peak']])
                    #print(p)
                    L[i] = p['peakList']['peak'][0]['@center']
                else:
                    L[i] = p['peakList']['peak']['@center']
    return L

def prepare_adj(entry):
    N = len(entry['atoms'])
    A = np.zeros((N,N), dtype=np.int32)
    for b in entry['bonds']:
        i,j = [int(s[1:]) - 1 for s in b['@atomRefs'].split()]
        A[i, j] = int(b['@order'])
        A[j, i] = A[i,j]
    #add self loops
    #for i in range(N):
        #A[i,i] = bond_dictionary['self']
    return A

def prepare_expconditions(entry, exp_dictionary):
    normalized_solvent = entry['solvent'].lower()
    if normalized_solvent not in exp_dictionary:
        exp_dictionary[normalized_solvent] = len(exp_dictionary)
    return np.repeat(exp_dictionary[normalized_solvent], len(entry['atoms']))

def padto(a, shape):
    pad = tuple((0,shape[i] - a.shape[i]) for i in range(len(shape)))
    if pad[0][1] < 0:
        return None
    return np.pad(a, pad, 'constant')

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
        # Not necessary according to current docs
        '''
        mol.UpdatePropertyCache(strict=False)
        for i in range(1000):
            r = AllChem.MMFFOptimizeMolecule(mol, maxIters=100)
            if r == 0:
                break
            if r == -1:
                raise ValueError()
        '''
    except (ValueError,RuntimeError) as e:
        print('Unable to process')
        print(Chem.MolToSmiles(mol))
        raise e
    for c in mol.GetConformers():
        pos = c.GetPositions()
        N = len(pos)
        np_pos = np.zeros( ( N, 3))
        np_pos[:N, :] = pos
        pos_nlist = nlist_model(np_pos)
        nlist = np.zeros( (MAX_ATOM_NUMBER, NEIGHBOR_NUMBER, 3) )

        
        # compute bond distances
        bonds = np.zeros( (MAX_ATOM_NUMBER,MAX_ATOM_NUMBER), dtype=np.int64)
        # need to rebuild adjacency matrix with new atom ordering
        for b in mol.GetBonds():
            bonds[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] = 1
            bonds[b.GetEndAtomIdx(), b.GetBeginAtomIdx()] = 1

        # a 0 -> non-bonded
        for index in range(N):
            for ni in range(len(pos_nlist[index])):
                if pos_nlist[index, ni, 0] >= 100: # this is a large distance sentinel indicating not part of nlist
                    continue
                j = int(pos_nlist[index, ni, 1])
                # / 10 to get to nm
                nlist[index, ni, 0] = pos_nlist[index, ni, 0] / 10
                nlist[index, ni, 1] = j
                # a 0 -> non-bonded
                if bonds[index, ni] == 0:
                    nlist[index,ni,2] = embeddings['nlist']['nonbonded']
                else:
                    # currently only single is used!
                    nlist[index,ni,2] = embeddings['nlist'][1]
        # pad out the nlist
        for index in range(N, MAX_ATOM_NUMBER):
            for ni in range(NEIGHBOR_NUMBER):
                nlist[index, ni, 0] = 0
                nlist[index, ni, 1] = 0
                nlist[index, ni, 2] = embeddings['nlist']['none']
        if False:
            # debugging
            print(nlist[:len(atoms)])
            a1, a2 = np.nonzero(A)
            for a1i, a2i in zip(a1, a2):
                print(a1i, a2i)
            exit()
        yield nlist

DATA_DIR = sys.argv[1]

embeddings = load_embeddings('embeddings.pb')

with open(DATA_DIR + 'metabolite_data.pb', 'rb') as f:
    raw_data = pickle.load(f)


# turn off GPU for more memory
config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
#config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
with tf.python_io.TFRecordWriter('train-structure-metabolite-data-{}-{}.tfrecord'.format(MAX_ATOM_NUMBER, NEIGHBOR_NUMBER),
                                 options=tf.io.TFRecordCompressionType.GZIP) as writer:
    with tf.Session(config=config) as sess:
        nm = nlist_model(NEIGHBOR_NUMBER, sess)
        successes = 0
        pbar = tqdm.tqdm(raw_data)
        for rd in pbar:
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
                    pbar.set_description('{}:{}. Successes: {}'.format(class_label,ci, successes))
                    record = make_tfrecord(atom_data, mask_data, nlist, peak_data, embeddings['class'][class_label], name_data)
                    writer.write(record.SerializeToString())
            except ValueError as e:
                continue
            successes += 1
# I like my embeddings rn, so won't overwrite
#save_embeddings(embeddings, 'embeddings.pb')
