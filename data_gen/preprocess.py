
import sys

sys.path.extend(['../'])
from data_gen.rotation import *
from tqdm import tqdm


def pre_normalization(data, zaxis=[0, 1], xaxis=[8, 4]):
    N, C, T, V, M = data.shape
    s = np.transpose(data, [0, 4, 2, 3, 1])  # N, C, T, V, M  to  N, M, T, V, C

    print('pad the null frames with the previous frames')
    for i_s, skeleton in enumerate(tqdm(s)):  # pad
        if skeleton.sum() == 0:
            print(i_s, ' has no skeleton')
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            if person[0].sum() == 0:
                index = (person.sum(-1).sum(-1) != 0)
                tmp = person[index].copy()
                person *= 0
                person[:len(tmp)] = tmp
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    if person[i_f:].sum() == 0: