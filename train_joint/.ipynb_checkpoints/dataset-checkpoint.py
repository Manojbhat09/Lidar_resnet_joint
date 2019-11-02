import os
import pickle
import numpy as np
from PIL import Image
import multiprocessing as mp

import torch
import torchvision
from torch.utils.data.dataset import Dataset

from pyntcloud import PyntCloud
# from open3d import io
from laserscan import LaserScan

import glob

_MAP_VERSION = 'v1.2'


# class ArgoverseDataset(Dataset):

#     def __init__(self, data_dir, min_past_obv_len=2, min_future_obv_len=10, min_future_pred_len=15, transform=None, num_workers=None):
#         """
#         Args:
#         :param data : List of [scene_id, scene_image, number_agents, past_list, future_list,
#                                encode_coordinates, decode_coordinates]
#         """
#         self.transform = transform
#         self.num_workers = num_workers

#         self.scene_id = []
#         self.scene_map_paths = []
        
#         # Extract Data:
#         self.get_data(data_dir)

#     def __getitem__(self, idx):

#         # Extract scene map image.
#         map_image = Image.open(self.scene_map_paths[idx])
#         label_ = np.fromfile(file_name, dtype=np.int32).reshape((-1))
#         pointcloud_ = 
#         assert len(label_) == len(pointcloud_)
        
#         if self.transform:
#             map_image = self.transform(map_image)
#         map_tensor = torchvision.transforms.ToTensor()(map_image)
#         return map_tensor, pointcloud_tensor

#     def __len__(self):
#         return len(self.scene_id)

#     def get_data(self, root_dir):
#         sub_directories = os.listdir(root_dir)
#         sub_directories.sort()
#         for i, sub_directory in enumerate(sub_directories):
#             sub_directory = root_dir + sub_directory + '/'
#             print(f'Extracting data from [{i}/{len(os.listdir(root_dir))}] directory:  {sub_directory}')
#             self.extract_directory(sub_directory)

#         print('Extraction Compltete!\n')

#     def data_partition(self, path_list, n):
#         chunk_size = len(path_list) // n
#         return [[path_list[i:i + chunk_size]] for i in range(0, len(path_list), chunk_size)]

#     def extract_directory(self, directory):
#         if self.num_workers:
#             num_processes = self.num_workers
#         else:
#             num_processes = mp.cpu_count()

#         scene_segments = os.listdir(directory)

#         path_list = []
#         for scene_segment in scene_segments:
#             observation_dir = directory + scene_segment + '/observation'
#             observations = os.listdir(observation_dir)
#             prediction_dir = directory + scene_segment + '/prediction'
#             predictions = os.listdir(prediction_dir)
#             print("here")

#             assert (len(predictions) == len(observations))

#             for observation in observations:
#                 path_list.append((directory, scene_segment, observation))

#         slices = self.data_partition(path_list, num_processes)
#         pool = mp.Pool(processes=num_processes)
#         results = pool.starmap(self.extract_submodule_multicore, slices)

#         for result in results:
#             self.scene_id.extend(result[0])
#             self.scene_map_paths.extend(result[1])

#     def extract_submodule_multicore(self, path_lists):

#         scene_id = []
#         future_agent_masks_list = []
#         past_agents_state_list = []
#         future_agents_state_list = []
#         encode_coordinates = []
#         decode_coordinates = []
#         scene_map_paths = []

#         for path_list in path_lists:
#             directory, scene_segment, observation = path_list
#             observation_path = directory + scene_segment + '/observation/' + observation
#             prediction_path = directory + scene_segment + '/prediction/' + observation
#             map_path = directory + scene_segment + '/map/v1/' + observation.replace('pkl', 'jpg')

#             scene_id.append(scene_segment + '/' + observation)
#             scene_map_paths.append(map_path)

#         return scene_id, scene_map_paths

#     def argoverse_collate(self, batch):
#         batch_size = len(batch)
#         batch_scene = torch.empty((0))

#         for k in range(batch_size):
#             # Scene Images
#             scene_image = (batch[k][0]).unsqueeze(0)
#             batch_scene = torch.cat((batch_scene, scene_image), dim=0)
#         return (batch_scene)


import glob

class JointDataset(Dataset):

    def __init__(self, root,  transform=None, num_workers=None):
        """
        Args:
        :param data : List of [scene_id, scene_image, number_agents, past_list, future_list,
                               encode_coordinates, decode_coordinates]
        """
        self.transform = transform
        self.num_workers = num_workers

        self.root = root
        self.scene_id = [] 
        self.scene_map_paths = []
        self.label_files = []
        self.pc_files = []
        self.index = {}
        self.current_scene = 0
        
        # Since there are limited number of map data: from 19 to some extent, We should have to limit the possible data to them.
        # Thus: 
        #   1. Load the possible data in the Map.
        #   2. Extend their names.

        self.scene_id = glob.glob(root + os.path.sep +"*")
        for each in self.scene_id:
            
            for pnt in glob.glob(os.path.join(each, 'map',  _MAP_VERSION ,'*')):
                ind = pnt.split(os.path.sep)[-1].split('.')[0]  # Following what done in the Argoverse-API. Quite messy.
                self.scene_map_paths.append(os.path.join(each, 'map', _MAP_VERSION , ind + '.png'))
                self.label_files.append(os.path.join(each, 'labels', ind + '.label'))
                self.pc_files.append(os.path.join(each, 'lidar', ind + '.ply'))

            
            assert(len(self.label_files) == len(self.scene_map_paths))
        

    def __getitem__(self, idx):

        # Extract scene map image.
        # print(np.array(self.scene_map_paths).shape)
        # print(np.array(self.scene_map_paths)[0])
        # print(self.scene_map_paths[idx])

        # Data Loading
        label_ = np.fromfile(self.label_files[idx], dtype=np.int32).reshape((-1))
        pointcloud_ply = PyntCloud.from_file(self.pc_files[idx])
        pointcloud_ = pointcloud_ply.xyz
#         pointcloud_ply = io.read_pointcloud(self.pc_files[idx])
#         pointcloud_ = pointcloud_ply.points
        assert len(label_) == len(pointcloud_)
        full_data = np.concatenate((pointcloud_,label_[:,np.newaxis]), axis=1)
#         print("full data shape ",full_data.shape)
        scanner = LaserScan(project=True)
        scanner.create_scan(full_data)
        
        main_tensor = np.empty((150000,4))
        main_tensor[:full_data.shape[0], :full_data.shape[1]] = full_data
        
        # prepare input with channel in dim=2
        complete_data = scanner.proj_xyz
        complete_data_torch = torch.from_numpy(complete_data)
        complete_data_torch = torch.transpose(complete_data_torch, 0, 2)
        complete_data_torch = torch.transpose(complete_data_torch, 1, 2)
#         print("xyz shape ",complete_data_torch.size())

        range_data = scanner.proj_range
        range_data_torch = torch.from_numpy(range_data)
        range_data_torch = torch.transpose(range_data_torch.unsqueeze(2), 0, 2)
        range_data_torch = torch.transpose(range_data_torch, 1, 2)
#         print("range shape ",range_data_torch.size())

        remission_data = scanner.proj_remission
        remission_data_torch = torch.from_numpy(remission_data)
        remission_data_torch = torch.transpose(remission_data_torch.unsqueeze(2), 0, 2)
        remission_data_torch = torch.transpose(remission_data_torch, 1, 2)
#         print("remission shape ",remission_data_torch.size())

        concat_data = torch.cat((range_data_torch, complete_data_torch, remission_data_torch), dim=0)
#         print("concat dim ", concat_data.size())
        
        map_image = Image.open(self.scene_map_paths[idx])
        if self.transform:
            map_image = self.transform(map_image)

        map_tensor = torchvision.transforms.ToTensor()(map_image)
        
        return map_tensor, concat_data
    
    def __len__(self):
        return len(self.scene_map_paths)


if __name__ == "__main__":
    print(os.path.abspath('.'))

    # temp_class = JointDataset(os.path.join('.', 'train_joint', 'data'))
    temp_class = JointDataset(os.path.join("/mnt/sdb1/shpark/dataset/argoverse/argoverse11/argoverse-forecasting-from-tracking/train/train1/"))
    temp_val = temp_class[0]

    print(temp_val)

