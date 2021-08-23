# Data for NMR GNN

This contains the parsing scripts and data used for our [GNN chemical shift predictor model](https://github.com/ur-whitelab/nmrgnn).

## Install

```sh
pip install nmrgnn-data
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
conda install -c omnia -c conda-forge rdkit openmm numpy==1.18.5
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

See commands `parse-shiftml`, `parse-metabolites`, `parse-shiftx` which are parsers for various databases.

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
  title={Predicting Chemical Shifts with Graph Neural Networks},
  author={Yang, Ziyue and Chakraborty, Maghesree and White, Andrew D},
  journal={Chemical Science},
  year={2021},
  publisher={Royal Society of Chemistry}
}
```
