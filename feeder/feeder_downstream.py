# sys
import pickle

# torch
import torch
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
np.set_printoptions(threshold=np.inf)

try:
    from feeder import augmentations
except:
    import augmentations


class Feeder(torch.utils.data.Dataset):
    """ 
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
    """

    def __init__(self,
                 data_path,
                 label_path,
                 num_frame_path,
                 l_ratio,
                 input_size,
                 mmap=True):

        self.data_path = data_path
        self.label_path = label_path
        self.num_frame_path= num_frame_path
        self.input_size=input_size
     
        self.l_ratio = l_ratio


        self.load_data(mmap)
        self.N, self.C, self.T, self.V, self.M = self.data.shape
        self.S = self.V
        self.B = self.V
        self.Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                