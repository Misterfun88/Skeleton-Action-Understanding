
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
    
    else:
         #joint_indicies = np.random.choice(25, random.randint(5, 10), replace=False)
         joint_indicies = np.random.choice(25, 15,replace=False)
         
         temp = out[:,:,joint_indicies,:] 
         Corruption = np.array([
                           [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)],
                           [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)],
                           [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)] ])
         temp = np.dot(temp.transpose([1, 2, 3, 0]), Corruption)
         temp = temp.transpose(3, 0, 1, 2)
         out[:,:,joint_indicies,:] = temp
         return out



def pose_augmentation(input_data):


        Shear       = np.array([
                      [1,	random.uniform(-1, 1), 	random.uniform(-1, 1)],
                      [random.uniform(-1, 1), 1, 	random.uniform(-1, 1)],
                      [random.uniform(-1, 1), 	random.uniform(-1, 1),      1]
                      ])

        temp_data = input_data.copy()
        result =  np.dot(temp_data.transpose([1, 2, 3, 0]),Shear.transpose())
        output = result.transpose(3, 0, 1, 2)

        return output

def temporal_cropresize(input_data,num_of_frames,l_ratio,output_size):


    C, T, V, M =input_data.shape