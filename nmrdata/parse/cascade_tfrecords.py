from multiprocessing.sharedctypes import Value
import tensorflow as tf
import numpy as np
import pandas as pd
try:
    from rdkit import Chem
except ImportError:
    # TODO think smarter about this
    pass
import tqdm
from nmrdata import *


def make_nlist(pos, embeddings, neighbor_number):
    # all nlists are non-bonded here
    N = pos.shape[0]
    pos_nlist = nlist_model(pos, neighbor_number)
    nlist = np.zeros((N, neighbor_number, 3), dtype=np.float32)
    # a 0 -> non-bonded
    for index in range(N):
        for ni in range(len(pos_nlist[index])):
            # this is a large distance sentinel indicating not part of nlist
            if pos_nlist[index, ni, 0] >= 100:
                continue
            j = int(pos_nlist[index, ni, 1])
            # / 10 to get to nm
            nlist[index, ni, 0] = pos_nlist[index, ni, 0] / 10
            nlist[index, ni, 1] = j
            nlist[index, ni, 2] = embeddings['nlist']['nonbonded']
        # pad out the nlist
        for index in range(N, N):
            for ni in range(neighbor_number):
                nlist[index, ni, 0] = 0
                nlist[index, ni, 1] = 0
                nlist[index, ni, 2] = embeddings['nlist']['none']
    return nlist


def parse_mol(mol, embeddings, shifts):
    new_embeddings = False
    N = mol.GetNumAtoms()
    name = mol.GetProp('_Name')
    features = np.empty((N), dtype=np.int64)
    atom_names = np.empty((N), dtype=np.int64)
    pos = np.empty((N, 3), dtype=np.float32)
    peaks = np.zeros((N), dtype=np.float32)
    for i in range(N):
        e = mol.GetAtomWithIdx(i).GetSymbol()
        if e not in embeddings['atom']:
            embeddings['atom'][e] = len(embeddings['atom'])
        features[i] = embeddings['atom'][e]
        atom_name = 'CAS-' + e
        if atom_name not in embeddings['name']:
            print('*******Adding new atom name*********')
            print(atom_name)
            embeddings['name'][atom_name] = len(embeddings['name'])
            new_embeddings = True
        atom_names[i] = embeddings['name'][atom_name]
        pos[i, :] = [mol.GetConformer().GetAtomPosition(i).x, mol.GetConformer(
        ).GetAtomPosition(i).y, mol.GetConformer().GetAtomPosition(i).z]
        match = shifts[shifts['atom_index'] == i]
        if match.shape[0] > 0:
            peaks[i] = match['Shift'].values[0]
            # check it
            if match['atom_type'].values[0] != mol.GetAtomWithIdx(
                    i).GetAtomicNum():
                raise ValueError(
                    'Mismatch between atom type and atomic number')
    return features, atom_names, pos, peaks, new_embeddings


@click.command('cascade')
@click.argument('sdf_file', type=click.Path(exists=True))
@click.argument('csv_file', type=click.Path(exists=True))
@click.argument('output_name')
@click.option('--embeddings', default=None, help='path to custom embeddings file')
@click.option('--neighbor-number', default=16, help='The model specific size of neighbor lists')
def parse_cascade(sdf_file, csv_file, output_name, embeddings, neighbor_number):

    embeddings = load_embeddings(embeddings)
    suppl = Chem.SDMolSupplier(sdf_file, removeHs=False, sanitize=False)
    shifts = pd.read_csv(csv_file)
    with tf.io.TFRecordWriter(f'cascade-{output_name}.tfrecord',
                              options=tf.io.TFRecordOptions(compression_type='GZIP')) as writer:
        successes = 0
        pbar = tqdm.tqdm(suppl)
        for mol in pbar:
            mol.UpdatePropertyCache()
            mol_id = int(mol.GetProp('_Name'))
            mshifts = shifts[shifts['mol_id'] == mol_id]
            if len(mshifts) == 0:
                print('No shifts for mol_id: {}'.format(mol_id))
                continue
            try:
                features, atom_names, pos, labels, new_embeddings = parse_mol(
                    mol, embeddings, mshifts)
            except ValueError as e:
                print(e)
                continue
            class_label = 'CAS'
            if class_label not in embeddings['class']:
                embeddings['class'][class_label] = len(embeddings['class'])
            mask_data = (labels != 0) * 1.0
            nlist = make_nlist(pos, embeddings, neighbor_number)
            pbar.set_description('{}:{}. Successes: {}'.format(
                class_label, mol_id, successes))
            record = make_tfrecord(
                features, mask_data, nlist, pos, labels, embeddings['class'][class_label], atom_names)
            writer.write(record.SerializeToString())
            successes += 1
            if new_embeddings:
                save_embeddings(embeddings, 'new-embeddings.pb')
