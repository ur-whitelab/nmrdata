# Data for NMR GNN

This contains the parsing scripts and data used for our [GNN chemical shift predictor model](https://github.com/ur-whitelab/nmrgnn).

## Install

```sh
pip install nmrgnn-data
```

## Working in Python

Here's an example of how to load and work with data in python. The records
are loaded as a tensorflow dataset ([read more here](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)), but can be used in a for loop as shown below.

```py
import nmrdata
dataset = nmrdata.load_records('data/metabolite-records.tfrecord')
for record in dataset:
    # get single record
    break
print(record.keys())
```
output:
```
dict_keys(['natoms', 'nneigh', 'features', 'nlist', 'positions', 'peaks', 'mask', 'name', 'class', 'index'])
```

Access positions as a numpy array
```py
record['positions'].numpy()
```
output:
```
array([[ 0.83740795,  0.09760247,  0.2959486 ],
       [-0.562893  ,  0.00262405, -0.00434441],
       [-1.0725924 , -0.37873718,  0.9061929 ],
       [-0.75536764, -0.72710234, -0.8159687 ],
       [-1.0367495 ,  0.9557108 , -0.27988592],
       [ 1.2855262 , -0.8334997 ,  0.10487328],
       [ 1.3046683 ,  0.8834019 , -0.20681578]], dtype=float32)
```
Get chemical shifts
```py
record['peaks'].numpy()
```
```
array([0.  , 0.  , 2.59, 2.59, 2.59, 0.  , 0.  ], dtype=float32)
```


## Numpy Error

If you see this error:

```py
ValueError: numpy.ndarray size changed, may indicate binary incompatibility. Expected 88 from C header, got 80 from PyObject
```

Try re-install numpy
```sh
pip uninstall -y numpy && pip install numpy
```

## Parsing Scripts
To install with the parsing functionality, use this

```sh
conda install -c omnia openmm
pip install nmrgnn-data[parse]
```

## Working with Data

All commands below can have additional information printed using the `--help` argument.

### Find pairs

Find pairs of atoms with chemical shifts that are neighbors and sort them based on distance.

```sh
nmrdata find-pairs structure-test.tfrecords-data.tfrecord ALA-H ALA-N
```

### Count Names

Get class/atom name counts:

```sh
nmrdata count-names structure-test.tfrecords-data.tfrecord
```

### Validate

Check that records are consistent with embeddings

```sh
nmrdata validate-embeddings structure-test.tfrecords-data.tfrecord
```

Check that neighbor lists are consistent with embeddings

```sh
nmrdata validate-nlist structure-test.tfrecords-data.tfrecord
```

Check that peaks are reasonable (no nans, no extreme values, no bad masks)

```sh
nmrdata validate-peaks structure-test.tfrecords-data.tfrecord
```

### Output Lables

To extract labels ordered by PDB and residue:

```sh
nmrdata write-peak-labels test-structure-shift-data.tfrecord  test-structure-shift-record-info.txt labels.txt
```

## Making New Data

See commands `nmrparse shiftml`, `nmrparse metabolites`, `nmrparse shiftx` which are parsers for various databases.

### From RefDB Files

This requires a pickled python object called `data.pb` to be in the directory. It is
a list of `dict`s containing `pdb_file` (path to PDB), `pdb` (PDB ID), `corr` (path to `.corr` file), and `chain` (which chain).
`chain` can be `_` to indicate use first chain.

```sh
nmrparse parse-refdb directory name --pdb_filter exclude_ids.txt
```
## Citation

Please cite [Predicting Chemical Shifts with Graph Neural Networks](https://pubs.rsc.org/en/content/articlehtml/2021/sc/d1sc01895g)

```bibtex
@article{yang2021predicting,
  title={Predicting chemical shifts with graph neural networks},
  author={Yang, Ziyue and Chakraborty, Maghesree and White, Andrew D},
  journal={Chemical science},
  volume={12},
  number={32},
  pages={10802--10809},
  year={2021},
  publisher={Royal Society of Chemistry}
}
```
