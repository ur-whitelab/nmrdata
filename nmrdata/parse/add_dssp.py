import mdtraj as md
import pickle, sys
import numpy as np
import os
import math
from simtk.openmm import app
import tqdm, os

# copied here so I do not need to load TF
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
        nlist = ['none', 'nonbonded'] + list(range(1, BOND_MAX + 1))
        embedding_dicts['nlist'] = dict(zip(nlist, range(len(nlist))))
    if 'name' not in embedding_dicts:
        embedding_dicts['name'] = {'X': 0}

    return embedding_dicts



PROTEIN_DIR = sys.argv[1]

# load embedding information
embedding_dicts = load_embeddings('embeddings.pb')

# load data info
with open(PROTEIN_DIR + 'data.pb', 'rb') as f:
    protein_data = pickle.load(f)


items = list(protein_data.values())
combos = dict()
with open('dssp_info.txt', 'w') as rinfo:
    pbar = tqdm.tqdm(items)
    rinfo.write('PDB Chain Residue Model_id Frame_id Residue_id DSSP\n')
    for index, entry in enumerate(pbar):
        pdb = PROTEIN_DIR + entry['pdb_file']
        chain = entry['chain']

        # get chains - FML
        # IGNORE FOR NOW - USING SHIFT
        mmp = app.PDBFile(pdb)
        if chain == '_':
            chain = list(mmp.topology.residues())[0].chain.id[0]
        chain_names = []
        for c in mmp.topology.chains():
            chain_names.append(c.id[0])
            
        chain_idx = chain_names.index(chain)
        
        p = md.load_pdb(pdb)
        
        dssp = md.compute_dssp(p)
        residues = list(p.topology.residues)        
        for i in range(dssp.shape[0]):
            for j in range(dssp.shape[1]):
                fid = i
                rid = residues[j].resSeq
                if residues[j].chain.index == chain_idx:
                    rinfo.write(f'{entry["pdb_file"].split("/")[-1]} {chain} {residues[j].name} {index} {fid} {rid} {dssp[i, j]}\n')
                    key = residues[j].name + '-' + dssp[i,j]
                    if key not in combos:
                        combos[key] = 0
                    combos[key] += 1
        pbar.set_description(f'Processed PDB {pdb} Total Records: {index}')
        rinfo.flush()

keys = list(combos.keys())
keys.sort()
for k in keys:
    print(k, combos[k])
