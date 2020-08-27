import tensorflow as tf
import pickle
import numpy as np
import os
import sys
from nmrdata import *
from .main import *
import click


if len(sys.argv) != 3:
    print('Usage check_data.py [filename.tfrecords] [embeddings]')


@nmrdata.command()
@click.option('--embeddings', default=None, help='Location to custom embeddings')
@click.argument('tfrecords')
def check_records(tfrecords, embeddings=None):
    '''Run validation functions on given path to TFRecords'''
    embeddings = load_embeddings(embeddings)
    validate_peaks(tfrecords, embeddings)
    validate_embeddings(tfrecords, embeddings)
