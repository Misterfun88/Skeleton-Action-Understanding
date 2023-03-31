import argparse
import pickle
from tqdm import tqdm
import sys
from numpy.lib.format import open_memmap

sys.path.extend(['../'])
from data_gen.preprocess import pre_normalization

# ntu 60
training_subjects = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19,