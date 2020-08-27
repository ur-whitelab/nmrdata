from .loading import *
import matplotlib.pyplot as plt
import numpy as np


def load_records(filename, batch_size=1):
    data = tf.data.TFRecordDataset(
        filename, compression_type='GZIP').map(data_parse)
    data = data.batch(batch_size)
    iterator = tf.data.Iterator.from_structure(
        data.output_types, data.output_shapes)
    init_op = iterator.make_initializer(data)
    bond_inputs, atom_inputs, peak_inputs, mask_inputs, name_inputs, class_input, record_index = iterator.get_next()
    return init_op, {'features': atom_inputs,
                     'nlist': bond_inputs,
                     'peaks': peak_inputs,
                     'mask': mask_inputs,
                     'name': name_inputs,
                     'class': class_input,
                     'index': record_index}


def peak_summary(data, embeddings, nbins, hist_range, predict_atom='H'):
    '''Returns summary statistics of peaks
    '''
    mask = tf.cast(data['mask'] > 0, tf.float32) * tf.cast(tf.math.equal(
        data['features'], embeddings['atom'][predict_atom]), tf.float32)
    hist = tf.histogram_fixed_width(data['peaks'] * mask, hist_range, nbins)
    run_ops = []
    # check for nans
    check = tf.check_numerics(
        data['peaks'], 'peaks invalid in {}'.format(data['index']))
    run_ops.append(check)
    # throw out zeros
    hist = hist * tf.constant([0] + [1] * (nbins - 1), dtype=tf.int32)
    running_hist = tf.get_variable(
        'peak-hist', initializer=tf.zeros_like(hist), trainable=False)
    run_ops.append(running_hist.assign_add(hist))
    # print out range, suspicious values
    running_min = tf.get_variable('peak-min', initializer=tf.constant(1.))
    running_max = tf.get_variable('peak-max', initializer=tf.constant(1.))
    peaks_min = tf.reduce_min(data['peaks'])
    peaks_max = tf.reduce_max(data['peaks'])
    run_ops.append(running_min.assign(tf.math.minimum(peaks_min, running_min)))
    run_ops.append(running_max.assign(tf.math.maximum(peaks_max, running_max)))
    count = tf.reduce_sum(mask)
    return running_min, running_max, running_hist, count, run_ops


def validate_peaks(filename, embeddings, batch_size=32):
    '''Checks for peaks beyond in extreme ranges and reports them
    '''
    tf.reset_default_graph()
    init_data_op, data = load_records(filename, batch_size=batch_size)
    nbins = int(1e6)
    hist_range = [0, 1e6]
    peaks_min_op, peaks_max_op, histogram_op, count_op, run_ops = peak_summary(
        data, embeddings, nbins, hist_range)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        sess.run(init_data_op)
        try:
            i = 0
            records = 0
            while True:
                peaks_min, peaks_max, histogram, count, * \
                    _ = sess.run([peaks_min_op, peaks_max_op,
                                  histogram_op, count_op] + run_ops)
                i += count
                records += batch_size
                print('\rValidating Peaks...peaks: {} records: {} min: {} max: {}'.format(
                    i, records, peaks_min, peaks_max), end='')
        except tf.errors.OutOfRangeError:
            print('Dataset complete')
            pass
    print('\nSummary')
    print('N = {}, Min = {}, Max = {}, > {} = {}'.format(
        i, peaks_min, peaks_max, hist_range[1], histogram[-1]))
    step = nbins / (len(histogram) + 1)
    for i in range(len(histogram)):
        if step * i > 20 and histogram[i] > 0:
            print('Suspicious peaks @ {} (N = {})'.format(
                step * i, histogram[i]))
    plt.plot(np.arange(0 + step, hist_range[1] - step, step), histogram)
    plt.xlim(0, 200)
    plt.savefig('peak-histogram.png', dpi=300)


def validate_embeddings(filename, embeddings, batch_size=32):
    '''Ensures that the records do not contain records not found in embeddings
    '''
    tf.reset_default_graph()
    init_data_op, data = load_records(filename, batch_size=batch_size)
    assert_ops = []
    assert_ops.append(tf.less(tf.reduce_max(data['features']),
                              tf.constant(max(list(embeddings['atom'].values())), dtype=tf.int64)))
    assert_ops.append(tf.less(tf.reduce_max(tf.cast(data['nlist'][:, :, 2], tf.int32)),
                              tf.constant(max(list(embeddings['nlist'].values())))))
    assert_ops.append(tf.less(tf.reduce_max(data['mask']),
                              tf.constant(0.1)))
    with tf.Session() as sess:
        sess.run(init_data_op)
        try:
            i = 0
            while True:
                sess.run(assert_ops)
                i += 1
                print('\rValidating Embeddings...{}'.format(i), end='')
        except tf.errors.OutOfRangeError:
            pass
    print('\nValid')


def count_names(filename, embeddings, batch_size=32):
    '''Counts the number of records for each name
    '''
    tf.reset_default_graph()
    init_data_op, data = load_records(filename, batch_size=batch_size)
    name_counts = [0 for _ in range(len(embeddings['name']))]

    with tf.Session() as sess:
        sess.run(init_data_op)
        try:
            count = 0
            while True:
                mask, names = sess.run([data['mask'], data['name']])
                masked = (names * mask).flatten()
                for j in masked[masked > 0]:
                    name_counts[int(j)] += 1
                    count += 1
                print('\rCounting names...{}'.format(count), end='')
        except tf.errors.OutOfRangeError:
            print('Dataset complete')
            pass
    return np.array(name_counts)


def write_peak_labels(filename, embeddings, record_info, output, batch_size=32):
    '''Writes peak labels from records with embedding labels
    '''

    # get look-ups for pdbs
    with open(record_info, 'r') as f:
        rinfo_table = np.loadtxt(f, skiprows=1, dtype='str')
        print(rinfo_table.shape)
        # convert to dict
        # key is model_id, value is pdb id
        rinfo = {int(mid): pdb.split('.pdb')[0] for pdb, mid in zip(
            rinfo_table[:, 0], rinfo_table[:, -3])}
    # look-ups for atom and res
    resdict = {v: k for k, v in embeddings['class'].items()}
    namedict = {v: k for k, v in embeddings['name'].items()}

    tf.reset_default_graph()
    init_data_op, data = load_records(filename, batch_size=batch_size)

    # Now write out data
    with tf.Session() as sess, open(output, 'w') as f:
        sess.run(init_data_op)
        try:
            count = 0
            while True:
                mask, peaks, name, c, index = sess.run(
                    [data['mask'], data['peaks'], data['name'], data['class'], data['index']])
                indices = np.nonzero(mask)
                for b, i in zip(indices[0], indices[1]):
                    p = rinfo[index[b, 0]]
                    r = resdict[c[b, 0]]
                    n = namedict[name[b, i]]
                    f.write(
                        ' '.join([p, str(index[b, 2]), *n.split('-'), str(peaks[b, i]), '\n']))
        except tf.errors.OutOfRangeError:
            print('Dataset complete')
            pass
    return


def find_pairs(filename, embeddings, name_i, name_j, batch_size=32):
    '''Writes peak labels from records with embedding labels
    '''

    pi = embeddings['name'][name_i]
    pj = embeddings['name'][name_j]

    tf.reset_default_graph()
    init_data_op, data = load_records(filename, batch_size=batch_size)
    print(f'Finding pairs between {name_i}({pi}) and {name_j}({pj})')

    result = []
    with tf.Session() as sess:
        sess.run(init_data_op)
        try:
            count = 0
            while True:
                mask, peaks, name, nlist = sess.run(
                    [data['mask'], data['peaks'], data['name'], data['nlist']])
                indices = np.nonzero(mask)
                for b, i in zip(indices[0], indices[1]):
                    # find particle i
                    if name[b, i] != pi:
                        continue
                    p = peaks[b, i]
                    # get names on nlist
                    diff = (name[b, nlist[b, i, :, 1].astype(int)] - pj)**2
                    if np.min(diff) == 0:
                        j = np.argmin(diff)
                        r = nlist[b, i, j, 0]
                        if r < 0.0001:
                            continue
                        result.append([r, p, peaks[b, j]])
                        count += 1
                    print(
                        f'\rCounting pairs...{count} last={result[-1] if count > 0 else ...}', end='')
        except tf.errors.OutOfRangeError:
            print('Dataset complete')
            pass
    return result


def duplicate_labels(filename, embeddings, record_info, batch_size=32, atom_filter='H'):
    '''Finds duplicate labels (same PDB) in record
    '''

    # get look-ups for pdbs
    with open(record_info, 'r') as f:
        rinfo_table = np.loadtxt(f, skiprows=1, dtype='str')
        print(rinfo_table.shape)
        # convert to dict
        # key is model_id, value is pdb id
        rinfo = {int(mid): pdb.split('.pdb')[0] for pdb, mid in zip(
            rinfo_table[:, 0], rinfo_table[:, -3])}
    # look-ups for atom and res
    resdict = {v: k for k, v in embeddings['class'].items()}
    namedict = {v: k for k, v in embeddings['name'].items()}

    tf.reset_default_graph()
    init_data_op, data = load_records(filename, batch_size=batch_size)

    # Now write out data
    all_labels = dict()
    with tf.Session() as sess:
        sess.run(init_data_op)
        try:
            count = 0
            while True:
                mask, peaks, name, c, index = sess.run(
                    [data['mask'], data['peaks'], data['name'], data['class'], data['index']])
                indices = np.nonzero(mask)
                for b, i in zip(indices[0], indices[1]):
                    p = rinfo[index[b, 0]]  # protein
                    r = resdict[c[b, 0]]  # residue
                    n = namedict[name[b, i]]  # name
                    # check if hydrogen
                    if atom_filter is not None and n.split('-')[1][0] != atom_filter:
                        continue
                    key = '{}-{}{}-{}'.format(p, index[b, 2], r, n)
                    if key in all_labels:
                        # check if it's same nmr file (index 0 matches)
                        present = False
                        for id, v in all_labels[key]:
                            if id == index[b, 0]:
                                present = True
                                break
                        if not present:
                            all_labels[key].append((index[b, 0], peaks[b, i]))
                    else:
                        all_labels[key] = [(index[b, 0], peaks[b, i])]
                print('\rFinding Unique Shifts: {}'.format(
                    len(all_labels)), end='')
        except tf.errors.OutOfRangeError:
            print('\nDataset complete')
    dup_labels = dict()
    for k, v in all_labels.items():
        if len(v) > 1:
            # remove model ids when keeping
            dup_labels[k] = [p for i, p in v]
    return dup_labels
