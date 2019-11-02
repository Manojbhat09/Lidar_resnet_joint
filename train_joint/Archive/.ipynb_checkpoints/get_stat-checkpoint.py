import numpy as np
import os
import glob

import pdb

_MAP_VERSION = 'v1.2'

def get_max_len(root):
    data_id = glob.glob(root + os.path.sep +"*")
    # print(scene_id)

    label_files = []
    for each in data_id:

        scenes = glob.glob(each + os.path.sep + "*")

        for scene in scenes:

            for pnt in glob.glob(os.path.join(scene, 'map', _MAP_VERSION, '*')):
                
                ind = pnt.split(os.path.sep)[-1].split('.')[0]

                label_files.append(os.path.join(scene, 'labels', ind + '.label'))


    max_len = 0

    for file_name in label_files:
        pnt = np.fromfile(file_name, dtype=np.int32)
        max_len = pnt.shape[0] if pnt.shape[0] > max_len else max_len

    print('Get Stat: max_len of label is ==> {}'.format(max_len))

    return max_len

root = os.path.join('/mnt/sdb1', 'shpark', 'dataset', 'argoverse', 'argoverse11', 'argoverse-forecasting-from-tracking', 'train')

if __name__ == "__main__":

    get_max_len(root)