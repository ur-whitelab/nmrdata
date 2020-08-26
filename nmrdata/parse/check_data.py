import tensorflow as tf
import pickle
import numpy as np
from graphnmr import *
import matplotlib.pyplot as plt
import os, sys


if len(sys.argv) != 3:
    print('Usage check_data.py [filename.tfrecords] [embeddings]')

embeddings = load_embeddings(sys.argv[2])
validate_peaks(sys.argv[1], embeddings)
validate_embeddings(sys.argv[1], embeddings)
