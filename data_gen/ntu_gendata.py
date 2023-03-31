import argparse
import pickle
from tqdm import tqdm
import sys
from numpy.lib.format import open_memmap

sys.path.extend(['../'])
from data_gen.preprocess import pre_normalization

# ntu 60
training_subjects = [
    1, 