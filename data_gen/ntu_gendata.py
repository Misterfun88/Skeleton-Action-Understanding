import argparse
import pickle
from tqdm import tqdm
import sys
from numpy.lib.format import open_memmap

sys.path.extend(['../'])
from data_gen.preprocess import pre_normalization

# ntu 60
training_subjects = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
]

# ntu 120
training_subjects = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38, 
    45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78,80, 81, 82, 
    83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100,