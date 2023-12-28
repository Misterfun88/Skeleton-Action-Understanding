
import torch.nn.functional as F
import torch
import random
import numpy as np


def joint_courruption(input_data):                                                                     

    out = input_data.copy()

    flip_prob  = random.random()

    if flip_prob < 0.5:

        #joint_indicies = np.random.choice(25, random.randint(5, 10), replace=False)
        joint_indicies = np.random.choice(25, 15,replace=False)
        out[:,:,joint_indicies,:] = 0 
        return out