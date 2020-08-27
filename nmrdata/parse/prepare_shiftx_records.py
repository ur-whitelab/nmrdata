import pickle
import os
import glob

# just walks through given directory and matches-up corrs with pdbs
data = dict()
pdbs = []
for f in glob.glob('**/*.pdbH', recursive=True):
    ns = f.split(os.path.sep)[-1].split('_')
    key = ns[0]
    pdb_id = ns[1].split('.')[0]
    data[key] = {'chain': '_', 'pdb_file': f, 'pdb_id': pdb_id}

for f in glob.glob('**/*.pdbresno', recursive=True):
    key = f.split(os.path.sep)[-1].split('_')[0]
    data[key]['corr'] = f

with open('data.pb', 'wb') as f:
    pickle.dump(data, f)
