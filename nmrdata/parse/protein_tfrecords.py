import tensorflow as tf
import pickle
import sys
import numpy as np
import os
from pdbfixer import PDBFixer
from simtk.openmm import app
import Bio.SeqUtils as seq
from Bio import pairwise2
from simtk import unit
import math
import tqdm
import os
import random
import traceback
import gsd.hoomd
from nmrdata import *

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
                    _, pdbid, srid, rname, name, element, shift, *_ = line.split()
                else:
                    _, srid, rname, name, element, shift, *_ = line.split()
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
    print(peaks)
    exit()

    if len(peaks) == 0:
        raise ValueError('Could not parse file')

    sequence_map = {}

    # sequence map -> key is residue index, output is peak index
    # want it to start from 0, since we'll get offset with alignment
    # this is necessary in case there are gaps
    min_id = min([int(p['index']) for p in peaks])
    for i, p in enumerate(peaks):
        sequence_map[int(p['index']) - min_id] = i

    # key in sequence_map should be index in sequence, output is peak index
    sequence = []
    for i in range(max(sequence_map.keys()) + 1):
        sequence.append('XXX')
        if i in sequence_map:
            sequence[i] = peaks[sequence_map[i]]['name']

    return peaks, sequence_map, sequence


def process_corr2(path, debug, shiftx_style):
    # Maybe someday use these?
    entry15000 = pynmrstar.Entry.from_file(path, convert_data_types=True)
    peaks = {}
    for chemical_shift_loop in entry15000.get_loops_by_category("Atom_chem_shift"):
        cs_result_sets.append(chemical_shift_loop.get_tag(
            ['Comp_index_ID', 'Comp_ID', 'Atom_ID', 'Atom_type', 'Val', 'Val_err']))
    # now parse out


def align(seq1, seq2, debug=False):
    flat1 = seq.seq1(''.join(seq1)).replace('X', '-')
    flat2 = seq.seq1(''.join(seq2)).replace('X', '-')
    flats = [flat1, flat2]
    # aligning 2 to 1 seems to give better results
    align = pairwise2.align.localxs(
        flat2, flat1, -1000, -1000, one_alignment_only=True)
    offset = [0, 0]
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
        print(pairwise2.format_alignment(
            flat2[offset[0]:], flat1[offset[1]:], 10, 0, len(flat1) - offset[1]))
    return -offset[0], -offset[1]


def process_pdb(path, corr_path, chain_id,
                gsd_file, embedding_dicts, neighbor_number, neighbor_margin=8,
                debug=False, units=unit.nanometer, frame_number=3, model_index=0,
                log_file=None, shiftx_style=False):

    global MA_LOST_FRAGS
    if shiftx_style:
        frame_number = 1
    # load pdb
    pdb = app.PDBFile(path)

    # load cs sets
    peak_data, sequence_map, peak_seq = process_corr(
        corr_path, debug, shiftx_style)

    result = []
    # check for weird/null chain
    if chain_id == '_':
        chain_id = list(pdb.topology.residues())[0].chain.id[0]
    # sometimes chains have extra characters (why?)
    residues = list(
        filter(lambda r: r.chain.id[0] == chain_id, pdb.topology.residues()))
    if len(residues) == 0:
        if debug:
            raise ValueError('Failed to find requested chain ', chain_id)

    pdb_offset, seq_offset = None, None

    # from pdb residue index to our aligned residue index
    residue_lookup = {}
    peak_count = 0
    # select a random set of frames for generating data without replacement
    frame_choices = random.sample(
        range(0, pdb.getNumFrames()), k=min(pdb.getNumFrames(), frame_number))
    for fi in frame_choices:
        success = True
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
        residues = list(
            filter(lambda r: r.chain.id[0] == chain_id, fixer.topology.residues()))
        if num_atoms > 100000:
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
            # TOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOODDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDOOOOOOOOOOOOOOOOOOOOOOO?????
            # Maybe it's ok
            pdb_offset = 0
            if debug:
                print('pdb_offset', pdb_offset)
                print('seq_offset', seq_offset)
                #print(sequence_map)
                # now check alignment - rarely perfect
                saw_one = False
                aligned = 0
                for i in range(len(residues)):
                    segid = int(residues[i].id) + pdb_offset
                    saw_one = pdb_seq[segid] == residues[i].name
                    if not saw_one:
                        print('Mismatch (A) at position {} ({}). {} != {}'.format(
                            segid, residues[i].id, pdb_seq[segid], residues[i].name))
                        continue
                    if segid + seq_offset in sequence_map:
                        peakid = sequence_map[segid + seq_offset]
                        print(segid, segid + seq_offset,
                              len(pdb_seq), len(peak_seq))
                        saw_one = pdb_seq[segid] == peak_seq[segid + seq_offset]
                        if not saw_one:
                            print('Mismatch (B) at position {}. pdb seq: {}, peak seq: {}'.format(
                                segid, peak_seq[segid + seq_offset], pdb_seq[peakid]))
                            continue
                        saw_one = peak_data[peakid]['name'] == residues[i].name
                        if not saw_one:
                            print('Mismatch (C) at position {}. peak seq: {}, peak data: {}, residue: {}'.format(
                                segid, i, peak_seq[segid + seq_offset], peak_data[peakid]['name'], residues[i].name))
                            continue
                        aligned += 1
                if aligned < 5:
                    raise ValueError(
                        'Could not find more than 5 aligned residues, very unusual')

            # create resiud look-up from atom index
            for i, r in enumerate(residues):
                for a in r.atoms():
                    residue_lookup[a.index] = i
            # This alignment will be checked as we compare shifts against the pdb
        # get neighbor list for frame
        np_pos = np.array([v.value_in_unit(units)
                           for v in frame], dtype=np.float32)
        frame_nlist = nlist_model(
            np_pos, neighbor_number + neighbor_margin, sorted=True)

        atoms = np.zeros((num_atoms), dtype=np.int64)
        mask = np.zeros((num_atoms), dtype=np.float)
        bonds = np.zeros((num_atoms, num_atoms), dtype=np.int64)
        # nlist:
        # :,:,0 -> distance
        # :,:,1 -> neighbor index
        # :,:,2 -> type
        nlist = np.zeros((num_atoms, neighbor_number, 3), dtype=np.float)
        positions = np.zeros((num_atoms, 3), dtype=np.float)
        peaks = np.zeros((num_atoms), dtype=np.float)
        names = np.zeros((num_atoms), dtype=np.int64)
        # going from pdb atom index to index in these data structures
        rmap = dict()
        index = 0
        for ri, residue in enumerate(residues):
            use_peaks = True
            # use the alignment result to get offset
            segid = int(residue.id) + pdb_offset
            if segid + seq_offset not in sequence_map:
                if debug:
                    print('Could not find residue index', residue,
                          'in the sequence map. Its index is', segid + seq_offset)
                use_peaks = False
            else:
                peak_id = sequence_map[segid + seq_offset]
                if peak_id >= len(peak_data):
                    if debug:
                        print('peakd id is outside of peak range')
                    use_peaks = False
                else:
                    # check if things are aligned
                    if residue.name != peak_data[peak_id]['name']:
                        if debug:
                            print('Mismatch between residue ', ri, peak_id, residue,
                                  segid, peak_data[peak_id], path, corr_path, chain_id)
                        use_peaks = False
            for atom in residue.atoms():
                mask[index] = 1. if use_peaks else 0.
                atom_name = residue.name + '-' + atom.name
                if atom_name not in embedding_dicts['name']:
                    embedding_dicts['name'][atom_name] = len(
                        embedding_dicts['name'])
                names[index] = embedding_dicts['name'][atom_name]

                if atom.element.symbol not in embedding_dicts['atom']:
                    if debug:
                        print('Could not identify atom',
                              atom.element.symbol)
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
        if not success:
            break
        # do this after so our reverse mapping is complete
        for residue in residues:
            for b in residue.bonds():
                # set bonds
                try:
                    bonds[rmap[b.atom1.index], rmap[b.atom2.index]] = 1
                    bonds[rmap[b.atom2.index], rmap[b.atom1.index]] = 1
                except KeyError:
                    # can be due to other chain bond or residue part of alignment.
                    pass
        for residue in residues:
            for a in residue.atoms():
                index = rmap[a.index]
                # convert to local indices and filter neighbors
                n_index = 0
                for ni in range(neighbor_number + neighbor_margin):
                    if frame_nlist[a.index, ni, 0] > 50.0:
                        # large distances are sentinels for things
                        # like self neighbors
                        continue
                    try:
                        j = rmap[int(frame_nlist[a.index, ni, 1])]
                    except KeyError:
                        continue
                    # a 0 -> non-bonded
                    if bonds[index, j] == 0:
                        # set index
                        nlist[index, n_index, 1] = j
                        # set distance
                        nlist[index, n_index,
                              0] = frame_nlist[a.index, ni, 0]
                        # set type
                        nlist[index, n_index,
                              2] = embedding_dicts['nlist']['nonbonded']
                        n_index += 1
                    # covalent bonded
                    else:
                        # set index
                        nlist[index, n_index, 1] = j
                        # set distance
                        nlist[index, n_index,
                              0] = frame_nlist[a.index, ni, 0]
                        # set type
                        # 1 -> covalent bond
                        nlist[index, n_index,
                              2] = embedding_dicts['nlist'][1]
                        n_index += 1
                    if n_index == neighbor_number:
                        break
        if not success:
            if debug:
                raise RuntimeError()
            continue
        if gsd_file is not None:
            snapshot = write_record_traj(positions, atoms, mask, nlist, peaks,
                                         embedding_dicts['class'][residues[ri].name], names, embedding_dicts)
            snapshot.configuration.step = len(gsd_file)
            gsd_file.append(snapshot)
        result.append(make_tfrecord(atoms, mask, nlist, peaks, embedding_dicts['class'][residues[ri].name], names, indices=np.array(
            [model_index, fi, int(residues[ri].id)], dtype=np.int64)))
        if log_file is not None:
            log_file.write('{} {} {} {} {} {} {} {}\n'.format(path.split('/')[-1], corr_path.split(
                '/')[-1], chain_id, len(peak_successes), len(gsd_file), model_index, fi, residues[ri].id))
    return result, len(peak_successes) / len(peak_data), len(result), peak_count


@click.command()
@click.argument('protein_dir')
@click.argument('output_name')
@click.option('--embeddings', default=None, help='path to custom embeddings file')
@click.option('--shiftx/--no-shiftx', default=False, help='Are these the cleaned shiftx files?')
@click.option('--debug/--no-debug', default=False)
@click.option('--pdb_filter', default=None, help='file containing list of pdbs to exclude')
@click.option('--invert_filter', default=False, help='Invert the pdb filter to only include the pdbs')
@click.option('--neighbor-number', default=16, help='The model specific size of neighbor lists')
@click.option('--gsd_frag_period', default=-1, help='How frequently to write GSD fragments')
def parse_refdb(protein_dir, embeddings, output_name, neighbor_number, pdb_filter, invert_filter, gsd_frag_period, shiftx, debug):
    # Optional filter to only consider certain pdbs
    pdb_filter_list = None
    if pdb_filter:
        pdb_filter_list = []
        with open(pdb_filter, 'r') as f:
            pdb_filter_list = set([x.split()[0] for x in f.readlines()])
        print('Will filter pdbs from', pdb_filter, flush=True)

    # load embedding information
    embedding_dicts = load_embeddings(embeddings)

    # load data info
    with open(os.path.join(protein_dir,'data.pb'), 'rb') as f:
        protein_data = pickle.load(f)

    items = list(protein_data.values())
    records = 0
    peaks = 0
    # turn off GPU for more memory if desired
    # config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    with tf.io.TFRecordWriter(f'structure-{output_name}-data.tfrecord',
                              options=tf.io.TFRecordOptions(compression_type='GZIP')) as writer,\
            gsd.hoomd.open(name='protein_frags.gsd', mode='wb') as gsd_file,\
            open('record_info.txt', 'w') as rinfo:
        pbar = tqdm.tqdm(items)
        rinfo.write(
            'PDB Corr Chain Count GSD_id Model_id Frame_id Residue_id\n')
        for index, entry in enumerate(pbar):
            if invert_filter:
                if pdb_filter_list is not None and entry['pdb_id'] not in pdb_filter_list:
                    continue
            else:
                if pdb_filter_list is not None and entry['pdb_id'] in pdb_filter_list:
                    continue
            try:
                result, p, n, pc = process_pdb(os.path.join(protein_dir,entry['pdb_file']), os.path.join(protein_dir,entry['corr']), entry['chain'],
                                               gsd_file=gsd_file, debug=debug,
                                               embedding_dicts=embedding_dicts, neighbor_number=neighbor_number,
                                               model_index=index, log_file=rinfo, shiftx_style=shiftx)
                pbar.set_description('Processed PDB {} ({}). Successes {} ({:.2}). Total Records: {}, Peaks: {}. Wrote frags: {}. Lost frags {}({})'.format(
                    entry['pdb_id'], entry['corr'], n, p, records, peaks, index % gsd_frag_period == 0, MA_LOST_FRAGS, MA_LOST_FRAGS / (MA_LOST_FRAGS + n + 1)))
                # turned off for now
                if False and len(result) == 0:
                    raise ValueError(
                        'Failed to find any records in' + entry['pdb_id'], entry['corr'])
                for r in result:
                    writer.write(r.SerializeToString())
                records += n
                peaks += pc
                rinfo.flush()
            except (ValueError, IndexError) as e:
                print(traceback.format_exc())
                print('Failed in ' + entry['pdb_id'], entry['corr'])
                pbar.set_description(
                    'Failed in ' + entry['pdb_id'], entry['corr'])
                # raise e
    if embeddings is not None:
        save_embeddings(embedding_dicts, 'embeddings.pb')
    print('wrote ', records)
