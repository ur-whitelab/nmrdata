from .loading import *
import numpy as np


class PeakSummary(tf.Module):
    def __init__(self):
        self.running_hist, self.running_max, self.running_min = None, None, None

    # TODO: why does this fail?
    # @tf.function
    def __call__(self, data, embeddings, nbins, hist_range, predict_atom='H'):
        '''Returns summary statistics of peaks
        '''
        mask = tf.cast(data['mask'] > 0, tf.float32)
        if predict_atom is not None:
            mask *= tf.cast(tf.math.equal(
                data['features'], embeddings['atom'][predict_atom]), tf.float32)
        peaks = data['peaks'] * mask
        hist = tf.histogram_fixed_width(
            peaks, hist_range, nbins)
        # check for nans
        tf.debugging.check_numerics(
            data['peaks'], 'peaks invalid in {}'.format(data['index']))
        # throw out zeros
        hist = hist * tf.constant([0] + [1] * (nbins - 1), dtype=tf.int32)
        if self.running_hist is None:
            self.running_hist = tf.Variable(
                tf.zeros_like(hist), trainable=False)
            self.running_max = tf.Variable(1.)
            self.running_min = tf.Variable(1.)
        self.running_hist.assign_add(hist)

        # print out range, suspicious values
        peaks_min = tf.reduce_min(peaks)
        peaks_max = tf.reduce_max(peaks)
        self.running_min.assign(tf.math.minimum(peaks_min, self.running_min))
        self.running_max.assign(tf.math.maximum(peaks_max, self.running_max))
        count = tf.reduce_sum(mask)
        return self.running_min, self.running_max, self.running_hist, count


@click.command()
@click.argument('tfrecords')
def count_records(tfrecords):
    '''Counts the number of records total in the filename
    '''
    data = load_records(tfrecords)
    count = 0
    for d in data:
        count += 1
        print('\rCounting records...{}'.format(count), end='')
    print('Records', count)
    return count


@click.command()
@click.argument('tfrecords')
@click.option('--embeddings', default=None, help='Location to custom embeddings')
@click.option('--atom_filter', default=None, help='only look at this atom')
def validate_peaks(tfrecords, embeddings, atom_filter):
    '''Checks for peaks beyond in extreme ranges and reports them
    '''
    data = load_records(tfrecords)
    embeddings = load_embeddings(embeddings)
    nbins = int(1e6)
    hist_range = [0, 1e6]
    records = 0
    i = 0
    ps = PeakSummary()
    for d in data:
        peaks_min, peaks_max, histogram, count = ps(
            d, embeddings, nbins, hist_range, atom_filter)
        if count == 0:
            print(f'Found no peaks in record {records} !!')
        i += count
        records += 1
        print('\rValidating Peaks...peaks: {} records: {} min: {} max: {}'.format(
            i, records, peaks_min.numpy(), peaks_max.numpy()), end='')
    print('\nSummary')
    histogram = histogram.numpy()
    print('N = {}, Min = {}, Max = {}, > {} = {}'.format(
        i, peaks_min.numpy(), peaks_max.numpy(), hist_range[1], histogram[-1]))
    step = nbins / (len(histogram) + 1)
    for i in range(len(histogram)):
        if atom_filter == 'H' and step * i > 20 and histogram[i] > 0:
            print('Suspicious peaks @ {} (N = {})'.format(
                step * i, histogram[i]))
    try:
        import matplotlib.pyplot as plt
        plt.plot(np.arange(0 + step, hist_range[1] - step, step), histogram)
        plt.xlim(0, 200)
        plt.savefig('peak-histogram.png', dpi=300)
    except ModuleNotFoundError:
        pass


@click.command()
@click.argument('tfrecords')
@click.option('--embeddings', default=None, help='Location to custom embeddings')
def validate_embeddings(tfrecords, embeddings):
    '''Ensures that the records do not contain records not found in embeddings
    '''
    weights = 'weighted' in tfrecords
    data = load_records(tfrecords)
    embeddings = load_embeddings(embeddings)
    i = 0
    for d in data:
        # check that the features are examples from the embedding
        tf.debugging.assert_less(tf.reduce_max(d['features']),
                                 tf.constant(max(list(embeddings['atom'].values())) + 1, dtype=tf.int64))
        tf.debugging.assert_less(tf.reduce_max(tf.cast(d['nlist'][:, :, 2], tf.int32)),
                                 tf.constant(max(list(embeddings['nlist'].values())) + 1))
        tf.debugging.assert_less(tf.reduce_max(tf.cast(d['name'], tf.int32)),
                                 tf.constant(max(list(embeddings['name'].values())) + 1))
        if weights:
            tf.debugging.assert_less(tf.reduce_max(d['mask']),
                                     tf.constant(0.1))
        i += 1
        print('\rValidating Embeddings...{}'.format(i), end='')
    print('\nValid')

@click.command()
@click.argument('tfrecords')
@click.option('--embeddings', default=None, help='Location to custom embeddings')
def validate_nlist(tfrecords, embeddings):
    '''Writes peak labels from records with embedding labels
    '''

    embeddings = load_embeddings(embeddings)

    full_data = load_records(tfrecords)
    print(f'Validating Neighborlists...', end='')

    count = 0
    for data in full_data:
        mask, peaks, name, nlist = [data['mask'].numpy(
        ), data['peaks'].numpy(), data['name'].numpy(), data['nlist'].numpy()]
        indices = np.nonzero(mask)
        assert len(indices) > 0
        for i in indices[0]:
            assert np.sum(nlist[i, :, 2] != embeddings['nlist']['none']) > 0, 'empty neighbor lists'
            assert np.sum((nlist[i, :, 2] == embeddings['nlist']['nonbonded']) * (nlist[i, :, 0] > 0.01)) > 0, 'neighbor lists distances are zero'
        count += 1
        print('\rValidating Neighborlists...{}'.format(count), end='')
    print('\nValid')


@click.command()
@click.argument('tfrecords')
@click.option('--embeddings', default=None, help='Location to custom embeddings')
def count_names(tfrecords, embeddings):
    '''Counts the number of records for each name
    '''
    data = load_records(tfrecords)
    embeddings = load_embeddings(embeddings)
    name_counts = [0 for _ in range(len(embeddings['name']))]

    count = 0
    for d in data:
        mask, names = d['mask'], d['name']
        masked = tf.reshape(tf.cast(names, tf.float32) * mask, (-1,))
        for j in masked[masked > 0]:
            name_counts[int(j)] += 1
            count += 1
        print('\rCounting names...{}'.format(count), end='')
    for k, v in embeddings['name'].items():
        print(k, name_counts[v])
    return np.array(name_counts)


@click.command()
@click.argument('tfrecords')
@click.argument('record_info')
@click.argument('output')
@click.option('--embeddings', default=None, help='Location to custom embeddings')
def write_peak_labels(tfrecords, embeddings, record_info, output):
    '''Writes peak labels from records with embedding labels
    '''

    embeddings = load_embeddings(embeddings)
    # get look-ups for pdbs
    with open(record_info, 'r') as f:
        rinfo_table = np.loadtxt(f, skiprows=1, dtype='str')
        print(rinfo_table.shape)
        # convert to dict
        # key is model_id, value is pdb id
        rinfo = {int(mid): pdb.split('.pdb')[0] for pdb, mid in zip(
            rinfo_table[:, 0], rinfo_table[:, -3])}
    # look-ups for atom and res
    namedict = {v: k for k, v in embeddings['name'].items()}

    data = load_records(tfrecords)

    # Now write out data
    with open(output, 'w') as f:
        for d in data:
            mask, peaks, name, index = [
                d['mask'], d['peaks'], d['name'], d['index']]
            indices = np.nonzero(mask)
            for b, i in zip(indices[0], indices[1]):
                p = rinfo[index[b, 0].numpy()]
                n = namedict[name[b, i].numpy()]
                f.write(
                    ' '.join([p, str(index[b, 2].numpy()), *n.split('-'), str(peaks[b, i].numpy()), '\n']))
    return


@click.command()
@click.argument('tfrecords')
@click.argument('name_i')
@click.argument('name_j')
@click.option('--embeddings', default=None, help='Location to custom embeddings')
def find_pairs(tfrecords, embeddings, name_i, name_j):
    '''Writes peak labels from records with embedding labels
    '''

    embeddings = load_embeddings(embeddings)
    pi = embeddings['name'][name_i]
    pj = embeddings['name'][name_j]

    full_data = load_records(tfrecords)
    print(f'Finding pairs between {name_i}({pi}) and {name_j}({pj})')

    result = []
    count = 0
    for data in full_data:
        mask, peaks, name, nlist = [data['mask'].numpy(
        ), data['peaks'].numpy(), data['name'].numpy(), data['nlist'].numpy()]
        indices = np.nonzero(mask)
        for i in indices[0]:
            # find particle i
            if name[i] != pi:
                continue
            p = peaks[i]
            # get names on nlist
            diff = (name[nlist[i, :, 1].astype(int)] - pj)**2
            if np.min(diff) == 0:
                j = np.argmin(diff)
                r = nlist[i, j, 0]
                if r < 0.0001:
                    continue
                result.append([r, p, peaks[j]])
                count += 1
            print(
                f'\rCounting pairs...{count} last={result[-1] if count > 0 else ...}', end='')
    print('')
    if len(result) == 0:
        print('none found')
    np.savetxt(f'{name_i}-{name_j}.txt', result,
               header='distance i-shift j-shift')


def duplicate_labels(tfrecords, embeddings, record_info, atom_filter='H'):
    '''Finds duplicate labels (same PDB) in record
    '''

    # get look-ups for pdbs
    with open(record_info, 'r') as f:
        rinfo_table = np.loadtxt(f, skiprows=1, dtype='str')
        # convert to dict
        # key is model_id, value is pdb id
        rinfo = {int(mid): pdb.split('.pdb')[0] for pdb, mid in zip(
            rinfo_table[:, 0], rinfo_table[:, -3])}
    # look-ups for atom and res
    resdict = {v: k for k, v in embeddings['class'].items()}
    namedict = {v: k for k, v in embeddings['name'].items()}

    full_data = load_records(tfrecords)
    all_labels = dict()

    # Now write out data
    for data in full_data:
        mask, peaks, name, c, index = [data['mask'].numpy(), data['peaks'].numpy(
        ), data['name'].numpy(), data['class'].numpy(), data['index'].numpy()]
        indices = np.nonzero(mask)
        for i in indices[0]:
            p = rinfo[index[0]]  # protein
            r = resdict[c[0]]  # residue
            n = namedict[name[i]]  # name
            # check if hydrogen
            if atom_filter is not None and n.split('-')[1][0] != atom_filter:
                continue
            key = '{}-{}{}-{}'.format(p, index[2], r, n)
            if key in all_labels:
                # check if it's same nmr file (index 0 matches)
                present = False
                for id, v in all_labels[key]:
                    if id == index[0]:
                        present = True
                        break
                if not present:
                    all_labels[key].append((index[0], peaks[i]))
            else:
                all_labels[key] = [(index[0], peaks[i])]
        print('\rFinding Unique Shifts: {}'.format(
            len(all_labels)), end='')
    print('\nDataset complete')
    dup_labels = dict()
    for k, v in all_labels.items():
        if len(v) > 1:
            # remove model ids when keeping
            dup_labels[k] = [p for i, p in v]
    return dup_labels
