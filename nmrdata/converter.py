import click
import tensorflow as tf
from .loading import *
from MDAnalysis import Universe


@click.command()
@click.argument('trajectory')
@click.argument('name')
@click.argument('--topology', default=None, help='extra topology file (e.g., PDB or GRO)')
@click.option('--embeddings', default=None, help='path to custom embeddings')
@click.option('--max-atom-number', default=256, help='The model specific size of fragments')
@click.option('--neighbor-number', default=16, help='The model specific size of neighbor lists')
def convert_trajectory(trajectory, topology, name, embeddings, max_atom_number, neighbor_number):
    # TODO: This only works for systems with less than 256 atoms.
    if topology is None:
        u = Universe(trajectory)
    else:
        u = Universe(topology, trajectory)
    protein = u.select_atoms('protein')

    with tf.python_io.TFRecordWriter(f'structure-{name}-data-{max_atom_number}-{neighbor_number}.tfrecord',
                                     options=tf.io.TFRecordCompressionType.GZIP) as writer:
