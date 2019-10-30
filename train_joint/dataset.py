import os
import pickle
import numpy as np
from PIL import Image
import multiprocessing as mp

import torch
import torchvision
from torch.utils.data.dataset import Dataset
import glob

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

        self.scene_id = []
        self.scene_map_paths = []
        self.label_files = []
        self.pc_files = []
        self.root = root
        
        self.scenes = glob.glob(root+"/*")
        for each in self.scenes:
            self.scene_map_paths.append(glob.glob(each+"/map/*"))
            self.label_files.append(glob.glob(each+"/label/*"))
            self.pc_files.append(glob.glob(each+"/lidar/*"))
        
        # Extract Data:
#         self.get_data(data_dir)

    def __getitem__(self, idx):

        # Extract scene map image.
        print(np.array(self.scene_map_paths).shape)
        print(np.array(self.scene_map_paths)[0])
        print(self.scene_map_paths[idx])
        map_image = Image.open(self.scene_map_paths[idx])
        label_ = np.fromfile(self.label_files[idx], dtype=np.int32).reshape((-1))
        pointcloud_ply = PyntCloud.from_file(self.pc_files[idx])
        pointcloud_ = points.xyz
        assert len(label_) == len(pointcloud_)
        
        if self.transform:
            map_image = self.transform(map_image)
        map_tensor = torchvision.transforms.ToTensor()(map_image)
        return map_tensor, pointcloud_, label_

    def __len__(self):
        return len(self.scene_id)