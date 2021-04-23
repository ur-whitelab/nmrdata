from nmrdata import *
import MDAnalysis as md
import numpy as np
import tensorflow as tf
import warnings
import os



def parse_universe(u, neighbor_number, embeddings, cutoff=None, pbc=False):
    '''Converts universe into atoms, edges, nlist
    '''
    N = u.atoms.positions.shape[0]
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
            warnings.warn('Guessing the system dimensions are' + str(dimensions))
    gridsearch = md.lib.nsgrid.FastNS(cutoff, u.atoms.positions, dimensions, max_gridsize=N**2 // 2, pbc=pbc)
    results = gridsearch.self_search()
    ragged_nlist = results.get_indices()
    ragged_edges = results.get_distances()
    nlist = np.zeros((N, neighbor_number), dtype=np.int32)
    edges = np.zeros((N, neighbor_number), dtype=np.float32)
    atoms = np.zeros(N, dtype=np.int32) 
    # check for elements
    try:
        elements = u.atoms.elements
    except md.exceptions.NoDataError as e:
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
            print('Unparameterized element' + elements[i] + 'will replace with unknown atom')
            atoms[i] = 1
            embeddings['name'][elements[i]] = len(embeddings['name'])
            new_embeddings = True
    edges /= 10 #angstrom to nm. TODO: need more reliable check
    # note we convert atoms to be one hot
    if new_embeddings:
        nef = 'new-embeddings.pb'
        if not os.path.exists(nef):
            print('Writing modified emebddings as new_embeddings')
            save_embeddings(embeddings, 'new-embeddings.pb')
        else:
            print('Will not write modified embeddings because file exists')            
    return tf.one_hot(atoms, len(embeddings['atom'])), edges, nlist
        
