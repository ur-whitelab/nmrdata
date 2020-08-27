import mdtraj as md
import pickle
import sys
import numpy as np
import os
import math
from simtk.openmm import app
import tqdm
import os
from graphnmr import *

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
                    rinfo.write(
                        f'{entry["pdb_file"].split("/")[-1]} {chain} {residues[j].name} {index} {fid} {rid} {dssp[i, j]}\n')
                    key = residues[j].name + '-' + dssp[i, j]
                    if key not in combos:
                        combos[key] = 0
                    combos[key] += 1
        pbar.set_description(f'Processed PDB {pdb} Total Records: {index}')
        rinfo.flush()

keys = list(combos.keys())
keys.sort()
for k in keys:
    print(k, combos[k])
