import tensorflow as tf
import pickle, sys
import numpy as np
import os
from pdbfixer import PDBFixer
from simtk.openmm import app
import Bio.SeqUtils as seq
from Bio import pairwise2
from simtk import unit
import gsd.hoomd
import math
import tqdm, os
import random
import traceback
from graphnmr import *

MA_LOST_FRAGS = 0

def pyparse_corr(path, shiftx_style):
    with open(path, 'r') as f:
        peaks = []
        entry_lines = False
        index = 0
        last_id = -1
        for line in f.readlines():
            if '_Atom_shift_assign_ID' in line:
                entry_lines = True
                continue
            if entry_lines and 'stop_' in line:
                entry_lines = False
                break
            if entry_lines and len(line.split()) > 3:
                if shiftx_style:
                    _,pdbid, srid,rname,name,element,shift,*_ = line.split()
                else:
                    _,srid,rname,name,element,shift,*_ = line.split()
                try:
                    rid = int(srid)
                except ValueError as e:
                    print(f'Failed in on line {line}')
                    raise e
                if rid != last_id:
                    peaks.append(dict(name=rname))
                    last_id = rid
                peaks[-1][name] = shift
                peaks[-1]['index'] = srid
    return peaks

def process_corr(path, debug, shiftx_style):

    peaks = pyparse_corr(path, shiftx_style)

    if len(peaks) == 0:
        raise ValueError('Could not parse file')

    sequence_map = {}

    # sequence map -> key is residue index, output is peak index 
    # want it to start from 0, since we'll get offset with alignment
    # this is necessary in case there are gaps
    min_id = min([int(p['index']) for p in peaks])
    for i,p in enumerate(peaks):
        sequence_map[int(p['index']) - min_id] = i
        continue
        if debug:
            print(sequence)
            print([p['name'] for p in peaks])
            print(i,p, sequence[int(p['index']) - min_id], p['name'])
            if p['name'] != sequence[int(p['index']) - min_id]:
                raise ValueError()

    # key in sequence_map should be index in sequence, output is peak index
    # New approach
    sequence = []
    for i in range(max(sequence_map.keys()) + 1):
        sequence.append('XXX')
        if i in sequence_map:
            sequence[i] = peaks[sequence_map[i]]['name']

    return peaks,sequence_map,sequence


def process_corr2(path, debug, shiftx_style):
    # Maybe someday use these?
    entry15000 = pynmrstar.Entry.from_file(path, convert_data_types=True)
    peaks = {}
    for chemical_shift_loop in entry15000.get_loops_by_category("Atom_chem_shift"):
        cs_result_sets.append(chemical_shift_loop.get_tag(['Comp_index_ID', 'Comp_ID', 'Atom_ID', 'Atom_type', 'Val', 'Val_err']))
    # now parse out
    


def align(seq1, seq2, debug=False):
    flat1 = seq.seq1(''.join(seq1)).replace('X', '-')
    flat2 = seq.seq1(''.join(seq2)).replace('X', '-')
    flats = [flat1, flat2]
    # aligning 2 to 1 seems to give better results
    align = pairwise2.align.localxs(flat2, flat1, -1000, -1000, one_alignment_only=True)
    start = align[0][3]
    offset = [0,0]
    # compute how many gaps had to be inserted at beginning to align
    for i in range(2):
        assert len(align[0][0]) == len(align[0][1])
        for j in range(len(align[0][0])):
            # account for the fact that 2 and 1 are switched in alignment results
            # if there is a gap in 1
            if align[0][(i + 1) % 2][j] == '-':
                # but not the other
                if flats[i][j - offset[i]] != '-':
                    offset[i] += 1
            else:
                break
    if debug:
        print(pairwise2.format_alignment(flat2[offset[0]:], flat1[offset[1]:], 10, 0, len(flat1) - offset[1]))
    return -offset[0], -offset[1]
    #return 0, start

# NN is not NEIGHBOR_NUMBer
# reason for difference is we don't want 1,3 or 1,4, etc neighbors on the list
def process_pdb(path, corr_path, chain_id, max_atoms,
                gsd_file, embedding_dicts, NN, nlist_model,
                keep_residues=[-1, 1],
                debug=False, units = unit.nanometer, frame_number=3, model_index=0,
                log_file=None, shiftx_style = False):
    
    global MA_LOST_FRAGS
    if shiftx_style:
        frame_number = 1
    # load pdb
    pdb = app.PDBFile(path)

    # load cs sets
    peak_data, sequence_map, peak_seq = process_corr(corr_path, debug, shiftx_style)
    
    result = []
    # check for weird/null chain
    if chain_id == '_':
        chain_id = list(pdb.topology.residues())[0].chain.id[0]
    # sometimes chains have extra characters (why?) 
    residues = list(filter(lambda r: r.chain.id[0] == chain_id, pdb.topology.residues()))
    if len(residues) == 0:
        if debug:
            raise ValueError('Failed to find requested chain ',chain_id)


    pdb_offset, seq_offset = None, None

    # from pdb residue index to our aligned residue index
    residue_lookup = {}
    # bonded neighbor mask
    nlist_mask = None
    peak_count = 0
    # select a random set of frames for generating data without replacement
    frame_choices = random.sample(range(0, pdb.getNumFrames()), k=min(pdb.getNumFrames(), frame_number))
    for fi in frame_choices:
        peak_successes = set()
        # clean up individual frame
        frame = pdb.getPositions(frame=fi)
        # have to fix at each frame since inserted atoms may change
        # fix missing residues/atoms
        fixer = PDBFixer(filename=path)
        # overwrite positions with frame positions
        fixer.positions = frame
        # we want to add missing atoms,
        # but not replace missing residue. We'd
        # rather just ignore those
        fixer.findMissingResidues()
        # remove the missing residues
        fixer.missingResidues = []
        # remove water!
        fixer.removeHeterogens(False)
        if not shiftx_style:
            fixer.findMissingAtoms()
            fixer.findNonstandardResidues()
            fixer.replaceNonstandardResidues()
            fixer.addMissingAtoms()
            fixer.addMissingHydrogens(7.0)
        # get new positions
        frame = fixer.positions
        num_atoms = len(frame)
        # remake residue list each time so they have correct atom ids
        residues = list(filter(lambda r: r.chain.id[0] == chain_id, fixer.topology.residues()))
        if num_atoms > 20000:
            MA_LOST_FRAGS += len(residues)
            if debug:
                print('Exceeded number of atoms for building nlist (change this if you have big GPU memory) in frame {} in pdb {}'.format(fi, path))
            break
        # check alignment once
        if pdb_offset is None:
            # create sequence from residues
            pdb_seq = ['XXX'] * max([int(r.id) + 1 for r in residues])
            for r in residues:
                rid = int(r.id)
                if rid >= 0:
                    pdb_seq[int(r.id)] = r.name
            if debug:
                print('pdb_seq', pdb_seq)
                print('peak_seq', peak_seq)
            pdb_offset, seq_offset = align(pdb_seq, peak_seq, debug)
            #TOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOODDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDOOOOOOOOOOOOOOOOOOOOOOO?????
            # Maybe it's ok
            pdb_offset = 0
            if debug:
                print('pdb_offset', pdb_offset)
                print('seq_offset', seq_offset)
                print(sequence_map)
                # now check alignment - rarely perfect
                saw_one = False
                aligned = 0
                for i in range(len(residues)):
                    segid = int(residues[i].id) + pdb_offset
                    saw_one = pdb_seq[segid] == residues[i].name
                    if not saw_one:                        
                        print('Mismatch (A) at position {} ({}). {} != {}'.format(segid, residues[i].id, pdb_seq[segid], residues[i].name))
                        continue
                    if segid + seq_offset in sequence_map:
                        peakid = sequence_map[segid + seq_offset]
                        print(segid, segid + seq_offset, len(pdb_seq), len(peak_seq))
                        saw_one = pdb_seq[segid] == peak_seq[segid + seq_offset]
                        if not saw_one:
                            print('Mismatch (B) at position {}. pdb seq: {}, peak seq: {}'.format(segid, peak_seq[segid + seq_offset], pdb_seq[peakid]))
                            continue
                        saw_one = peak_data[peakid]['name'] == residues[i].name
                        if not saw_one:
                            print('Mismatch (C) at position {}. peak seq: {}, peak data: {}, residue: {}'.format(segid, i, peak_seq[segid + seq_offset], peak_data[peakid]['name'], residues[i].name))
                            continue
                        aligned += 1
                if aligned < 5:
                    raise ValueError('Could not find more than 5 aligned residues, very unusual')
                    
            # create resiud look-up from atom index
            for i,r in enumerate(residues):
                for a in r.atoms():
                    residue_lookup[a.index] = i
            # This alignment will be checked as we compare shifts against the pdb
        # get neighbor list for frame
        np_pos = np.array([v.value_in_unit(units) for v in frame])
        frame_nlist = nlist_model(np_pos)

        for ri in range(len(residues)):
            # we build up fragment by getting residues around us, both in chain
            # and those within a certain distance of us
            rmin = max(0,ri + keep_residues[0])
            # have to +1 here (and not in range) to get min to work :)
            rmax = min(len(residues), ri + keep_residues[1] + 1) 
            # do we have any residues to consider?
            success = rmax - rmin > 0

            consider = set(range(rmin, rmax))

            # Used to indicate an atom should be included from a different residue
            marked = [False for _ in range(len(frame))]

            # now grab spatial neighbor residues
            # NOTE: I checked this by hand a lot
            # Believe this code.
            for a in residues[ri].atoms():
                for ni in range(NN):
                    j = int(frame_nlist[a.index, ni, 1])
                    try:
                        consider.add(residue_lookup[j])
                        marked[j] = True
                    except KeyError as e:
                        success = False
                        if debug:
                            print('Neighboring residue in different chain, skipping')
                        break
            atoms = np.zeros((max_atoms), dtype=np.int64)
            # we will put dummy atom at end to keep bond counts the same by bonding to it
            # Z-DISABLED
            #atoms[-1] = embedding_dicts['atom']['Z']
            mask = np.zeros( (max_atoms), dtype=np.float)
            bonds = np.zeros( (max_atoms, max_atoms), dtype=np.int64)
            # nlist:
            # :,:,0 -> distance
            # :,:,1 -> neighbor index
            # :,:,2 -> bond count
            nlist = np.zeros( (max_atoms, NEIGHBOR_NUMBER, 3), dtype=np.float)
            positions = np.zeros( (max_atoms, 3), dtype=np.float)
            peaks = np.zeros( (max_atoms), dtype=np.float)
            names = np.zeros( (max_atoms), dtype=np.int64)
            # going from pdb atom index to index in these data structures
            rmap = dict()
            index = 0
            # check our two conditions that could have made this false: there are residues and
            # we didn't have off-chain spatial neighboring residues
            if not success:
                continue
            for rj in consider:
                residue = residues[rj]
                # use the alignment result to get offset
                segid = int(residue.id) + pdb_offset
                if segid + seq_offset not in sequence_map:
                    if debug:
                        print('Could not find residue index', rj, ': ', residue, 'in the sequence map. Its index is', segid + seq_offset, 'ri: ', ri)
                        print('We are considering', consider)
                    success = False
                    break
                peak_id = sequence_map[segid + seq_offset]
                #peak_id = segid
                if peak_id >= len(peak_data):
                    success = False
                    if debug:
                        print('peakd id is outside of peak range')
                    break
                # only check for residue we actually care about
                if ri == rj and residue.name != peak_data[peak_id]['name']:
                    if debug:
                        print('Mismatch between residue ', ri, rj, peak_id, residue, segid, peak_data[peak_id], path, corr_path, chain_id)
                    success = False
                    break
                for atom in residue.atoms():
                    # Make sure atom is in residue or neighbor of residue atom
                    if ri != rj and not marked[atom.index]:
                        continue
                    mask[index] = float(ri == rj)
                    atom_name = residue.name + '-' + atom.name
                    if atom_name not in embedding_dicts['name']:
                        embedding_dicts['name'][atom_name] = len(embedding_dicts['name'])
                    names[index] = embedding_dicts['name'][atom_name]

                    if atom.element.symbol not in embedding_dicts['atom']:
                        if debug:
                            print('Could not identify atom', atom.element.symbol)
                        success = False
                        break
                    atoms[index] = embedding_dicts['atom'][atom.element.symbol]
                    positions[index] = np_pos[atom.index, :]
                    rmap[atom.index] = index
                    peaks[index] = 0
                    if mask[index]:
                        if atom.name[:3] in peak_data[peak_id]:
                            peaks[index] = peak_data[peak_id][atom.name[:3]]
                            peak_count += 1
                            peak_successes.add(peak_id)
                        else:
                            mask[index] = 0
                    index += 1
                    # Z-DISABLED
                    # -1 for dummy atom which is stored at end
                    if index == max_atoms - 1:#2:
                        MA_LOST_FRAGS += 1
                        if debug:
                            print('Not enough space for all atoms in ri', ri)
                        success = False
                        break
                if ri == rj and sum(mask) == 0:
                    if debug:
                        print('Warning found no peaks for', ri, rj, residue, peak_data[peak_id])
                    success = False
                if not success:
                    break
            if not success:
                continue
            # do this after so our reverse mapping is complete
            for rj in consider:
                residue = residues[rj]
                for b in residue.bonds():
                    # set bonds
                    try:
                        bonds[rmap[b.atom1.index], rmap[b.atom2.index]] = 1
                        bonds[rmap[b.atom2.index], rmap[b.atom1.index]] = 1
                    except KeyError:
                        # for bonds that cross residue
                        pass
            for rj in consider:
                residue = residues[rj]
                for a in residue.atoms():
                    # Make sure atom is in residue or neighbor of residue atom
                    if ri != rj and not marked[a.index] :
                        continue
                    index = rmap[a.index]
                    # convert to local indices and filter neighbors
                    n_index = 0
                    for ni in range(NN):
                        if frame_nlist[a.index, ni,0] > 50.0:
                            # large distances are sentinels for things
                            # like self neighbors
                            continue
                        try:
                            j = rmap[int(frame_nlist[a.index, ni, 1])]
                        except KeyError:
                            # either we couldn't find a neighbor on the root residue (which is bad)
                            # or just one of the neighbors is not on a considered residue.
                            if rj == ri:
                                success = False
                                if debug:
                                    print('Could not find all neighbors', int(frame_nlist[a.index, ni, 1]), consider)
                                break
                            # Z-DISABLED
                            #j = max_atoms - 1 # point to dummy atom
                            continue
                        # mark as not a neighbor if out of molecule (only for non-subject nlists)
                        if False and j == max_atoms - 1:
                            #set index
                            nlist[index,n_index,1] = j
                            # set distance
                            nlist[index,n_index,0] = frame_nlist[a.index, ni,0]
                            #set type
                            nlist[index, n_index, 2] = embedding_dicts['nlist']['none']
                            n_index += 1
                        # a 0 -> non-bonded
                        elif bonds[index, j] == 0:
                            #set index
                            nlist[index,n_index,1] = j
                            # set distance
                            nlist[index,n_index,0] = frame_nlist[a.index, ni,0]
                            #set type
                            nlist[index,n_index,2] = embedding_dicts['nlist']['nonbonded']
                            n_index += 1
                        # single bonded
                        else:
                            #set index
                            nlist[index,n_index,1] = j
                            # set distance
                            nlist[index,n_index,0] = frame_nlist[a.index,ni,0]
                            #set type
                            nlist[index,n_index,2] = embedding_dicts['nlist'][1]
                            n_index += 1
                        if n_index == NEIGHBOR_NUMBER:
                            break                            
                    # how did we do on peaks
                    if False and (peaks[index] > 0 and peaks[index] < 25):
                        nonbonded_count =  np.sum(nlist[index, :, 2] == embedding_dicts['nlist']['nonbonded'])
                        bonded_count = np.sum(nlist[index, :, 2] == embedding_dicts['nlist'][1])
                        print('neighbor summary: non-bonded: {}, bonded: {}, total: {}'.format(nonbonded_count, bonded_count, NEIGHBOR_NUMBER))
                        print(nlist[index, :, :])
                        exit()
            if not success:
                if debug:
                    raise RuntimeError()
                continue
            if gsd_file is not None:
                snapshot = write_record_traj(positions, atoms, mask, nlist, peaks, embedding_dicts['class'][residues[ri].name], names, embedding_dicts)
                snapshot.configuration.step = len(gsd_file)
                gsd_file.append(snapshot)
            result.append(make_tfrecord(atoms, mask, nlist, peaks, embedding_dicts['class'][residues[ri].name], names, indices=np.array([model_index, fi, int(residues[ri].id)], dtype=np.int64)))
            if log_file is not None:
                log_file.write('{} {} {} {} {} {} {} {}\n'.format(path.split('/')[-1], corr_path.split('/')[-1], chain_id, len(peak_successes), len(gsd_file), model_index, fi, residues[ri].id))
    return result, len(peak_successes) / len(peak_data), len(result), peak_count


PROTEIN_DIR = sys.argv[1]
# Optional filter to only consider certain pdbs
PDB_ID_FILTER = None
if len(sys.argv) >= 3:
    PDB_ID_FILTER = []
    with open(sys.argv[2], 'r') as f:
        PDB_ID_FILTER = set([x.split()[0] for x in f.readlines()])
    print('Will filter pdbs from', sys.argv[2], flush=True) 
INVERT_SELECT = False
if len(sys.argv) >= 4:
    print('Saw arg', sys.argv[3], 'so inverting filter', flush=True)
    INVERT_SELECT = True

WRITE_FRAG_PERIOD = 25

# load embedding information
embedding_dicts = load_embeddings('embeddings.pb')

# load data info
with open(PROTEIN_DIR + 'data.pb', 'rb') as f:
    protein_data = pickle.load(f)


items = list(protein_data.values())
results = []
records = 0
peaks = 0
# turn off GPU for more memory if desired
config = tf.ConfigProto(
       # device_count = {'GPU': 0}
    )
#config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
with tf.python_io.TFRecordWriter('train-structure-protein-data-{}-{}.tfrecord'.format(MAX_ATOM_NUMBER, NEIGHBOR_NUMBER),
                                 options=tf.io.TFRecordCompressionType.GZIP) as writer:
    with tf.Session(config=config) as sess,\
        gsd.hoomd.open(name='protein_frags.gsd', mode='wb') as gsd_file,\
        open('record_info.txt', 'w') as rinfo:
        # add a bit in case we have some 1,3, or 1,4 neighbors that we don't want
        NN = NEIGHBOR_NUMBER + 4
        nm = nlist_model(NN, sess)
        pbar = tqdm.tqdm(items)
        rinfo.write('PDB Corr Chain Count GSD_id Model_id Frame_id Residue_id\n')
        for index, entry in enumerate(pbar):
            if INVERT_SELECT:
                if PDB_ID_FILTER is not None and entry['pdb_id'] not in PDB_ID_FILTER:
                    continue
            else:
                if PDB_ID_FILTER is not None and entry['pdb_id'] in PDB_ID_FILTER:
                    continue                
            try:
                result, p, n, pc = process_pdb(PROTEIN_DIR + entry['pdb_file'], PROTEIN_DIR + entry['corr'], entry['chain'],
                                        gsd_file=gsd_file,
                                        max_atoms=MAX_ATOM_NUMBER, embedding_dicts=embedding_dicts, NN=NN,
                                               nlist_model=nm, model_index=index, log_file=rinfo)
                pbar.set_description('Processed PDB {} ({}). Successes {} ({:.2}). Total Records: {}, Peaks: {}. Wrote frags: {}. Lost frags {}({})'.format(
                                   entry['pdb_id'], entry['corr'], n, p, records, peaks, index % WRITE_FRAG_PERIOD == 0, MA_LOST_FRAGS, MA_LOST_FRAGS / (MA_LOST_FRAGS + n + 1)))
                # turned off for now
                if False and len(result) == 0:
                    raise ValueError('Failed to find any records in' +  entry['pdb_id'], entry['corr'])
                for r in result:
                    writer.write(r.SerializeToString())
                records += n
                peaks += pc
                rinfo.flush()
                save_embeddings(embedding_dicts, 'embeddings.pb')
            except (ValueError, IndexError) as e:
                print(traceback.format_exc())
                print('Failed in ' +  entry['pdb_id'], entry['corr'])
                pbar.set_description('Failed in ' +  entry['pdb_id'], entry['corr'])
                #raise e
print('wrote ', records)
