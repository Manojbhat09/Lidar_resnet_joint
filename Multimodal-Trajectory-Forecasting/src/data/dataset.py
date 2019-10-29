import os
import pickle
import numpy as np
from PIL import Image
import multiprocessing as mp

import torch
import torchvision
from torch.utils.data.dataset import Dataset


class ArgoverseDataset(Dataset):

    def __init__(self, data_dir, min_past_obv_len=2, min_future_obv_len=10, min_future_pred_len=15, transform=None, num_workers=None):
        """
        Args:
        :param data : List of [scene_id, scene_image, number_agents, past_list, future_list,
                               encode_coordinates, decode_coordinates]
        """
        self.min_past_obv_len = min_past_obv_len
        self.min_future_obv_len = min_future_obv_len
        self.min_future_pred_len = min_future_pred_len
        self.transform = transform
        self.num_workers = num_workers

        self.scene_id = []
        self.scene_map_paths = []
        self.future_agent_masks_list = []
        self.past_agents_state_list = []
        self.future_agents_state_list = []
        self.encode_coordinates = []
        self.decode_coordinates = []

        # Extract Data:
        self.get_data(data_dir)

    def __getitem__(self, idx):

        # Create one past list and future list with all the
        scene_agents_past_list, scene_agents_future_list = [], []
        decode_coordinates = []

        for k in range(len(self.future_agent_masks_list[idx])):
            scene_agents_past_list.append(torch.tensor(self.past_agents_state_list[idx][k]))

            if self.future_agent_masks_list[idx][k]:
                scene_agents_future_list.append(torch.tensor(self.future_agents_state_list[idx][k]))
                decode_coordinates.append(self.decode_coordinates[idx][k])

        # Extract scene map image.
        map_image = Image.open(self.scene_map_paths[idx])
        if self.transform:
            map_image = self.transform(map_image)
        map_tensor = torchvision.transforms.ToTensor()(map_image)

        scene_data = (map_tensor, self.future_agent_masks_list[idx], scene_agents_past_list,
                      scene_agents_future_list, self.encode_coordinates[idx], decode_coordinates)
        return scene_data

    def __len__(self):
        return len(self.scene_id)

    def get_data(self, root_dir):
        sub_directories = os.listdir(root_dir)
        sub_directories.sort()
        for i, sub_directory in enumerate(sub_directories):
            sub_directory = root_dir + sub_directory + '/'
            print(f'Extracting data from [{i}/{len(os.listdir(root_dir))}] directory:  {sub_directory}')
            self.extract_directory(sub_directory)

        print('Extraction Compltete!\n')

    def data_partition(self, path_list, n):
        chunk_size = len(path_list) // n
        return [[path_list[i:i + chunk_size]] for i in range(0, len(path_list), chunk_size)]

    def extract_directory(self, directory):
        if self.num_workers:
            num_processes = self.num_workers
        else:
            num_processes = mp.cpu_count()

        scene_segments = os.listdir(directory)

        path_list = []
        for scene_segment in scene_segments:
            observation_dir = directory + scene_segment + '/observation'
            observations = os.listdir(observation_dir)
            prediction_dir = directory + scene_segment + '/prediction'
            predictions = os.listdir(prediction_dir)

            assert (len(predictions) == len(observations))

            for observation in observations:
                path_list.append((directory, scene_segment, observation))

        slices = self.data_partition(path_list, num_processes)
        pool = mp.Pool(processes=num_processes)
        results = pool.starmap(self.extract_submodule_multicore, slices)

        for result in results:
            self.scene_id.extend(result[0])
            self.future_agent_masks_list.extend(result[1])
            self.past_agents_state_list.extend(result[2])
            self.future_agents_state_list.extend(result[3])
            self.encode_coordinates.extend(result[4])
            self.decode_coordinates.extend(result[5])
            self.scene_map_paths.extend(result[6])

    def extract_submodule_multicore(self, path_lists):

        scene_id = []
        future_agent_masks_list = []
        past_agents_state_list = []
        future_agents_state_list = []
        encode_coordinates = []
        decode_coordinates = []
        scene_map_paths = []

        for path_list in path_lists:
            directory, scene_segment, observation = path_list
            observation_path = directory + scene_segment + '/observation/' + observation
            prediction_path = directory + scene_segment + '/prediction/' + observation
            map_path = directory + scene_segment + '/map/v1/' + observation.replace('pkl', 'jpg')

            with open(observation_path, 'rb') as f:
                observation_df = pickle.load(f)
            with open(prediction_path, 'rb') as f:
                prediction_df = pickle.load(f)

            past_agent_ids, future_agent_ids_mask = self.get_agent_ids(observation_df, self.min_past_obv_len,
                                                                       self.min_future_obv_len, self.min_future_pred_len)
            past_trajs, future_trajs, encode_coords, decode_coords = self.extract_trajectory_info(observation_df, prediction_df,
                                                                                                  past_agent_ids, future_agent_ids_mask)

            scene_id.append(scene_segment + '/' + observation)
            future_agent_masks_list.append(future_agent_ids_mask)
            past_agents_state_list.append(past_trajs)
            future_agents_state_list.append(future_trajs)
            encode_coordinates.append(encode_coords)
            decode_coordinates.append(decode_coords)
            scene_map_paths.append(map_path)

        return scene_id, future_agent_masks_list, past_agents_state_list, future_agents_state_list, encode_coordinates, decode_coordinates, scene_map_paths

    def get_agent_ids(self, dataframe, min_past_obv_len, min_future_obv_len, min_future_pred_len):
        """
        Returns:
                List of past agent ids: List of agent ids that are to be considered for the ecoding phase.
                Future agent ids mask: A mask which dentoes if an agent in past agent ids list is to be considered
                                       during decoding phase.
        """
        # Select past agent ids for the encoding phase.
        past_df = dataframe.loc[((dataframe['class'] == 'VEHICLE') | (dataframe['class'] == 'LARGE_VEHICLE') | (
                    dataframe['class'] == 'EMERGENCY_VEHICLE') | (dataframe['class'] == 'TRAILER'))
                                & (dataframe['observation_length'] >= min_past_obv_len)]
        past_agent_ids = np.unique(past_df['track_id'].values)

        # Select future agent ids for the decoding phase.
        future_df = dataframe.loc[((dataframe['class'] == 'VEHICLE') | (dataframe['class'] == 'LARGE_VEHICLE') | (
                    dataframe['class'] == 'EMERGENCY_VEHICLE') | (dataframe['class'] == 'TRAILER'))
                                  & (dataframe['observation_length'] >= min_future_obv_len) & (
                                              dataframe['prediction_length'] >= min_future_pred_len)]
        future_agent_ids = np.unique(future_df['track_id'].values)

        # Create a mask corresponding to the past_agent_ids list where the value '1' in mask denotes
        # that agent is to be considered while decoding and 0 denotes otherwise.
        future_agent_ids_mask = []
        for agent_id in past_agent_ids:
            if agent_id in future_agent_ids:
                future_agent_ids_mask.append(1)
            else:
                future_agent_ids_mask.append(0)

        return past_agent_ids, future_agent_ids_mask

    def extract_trajectory_info(self, obv_df, pred_df, past_agent_ids, future_agent_ids_mask):
        """
        Extracts the past and future trajectories of the agents as well as the encode and decode
        coordinates.
        """
        past_traj_list = []
        future_traj_list = []
        encode_coordinate_list = []
        decode_coordinate_list = []
        for k, agent_id in enumerate(past_agent_ids):
            x_cords, y_cords = obv_df['X'][obv_df['track_id'] == agent_id], obv_df['Y'][obv_df['track_id'] == agent_id]
            past_agent_traj = [(x, y) for x, y in zip(x_cords, y_cords)]
            past_traj_list.append(past_agent_traj)
            encode_coordinate_list.append(past_agent_traj[-1])
            if future_agent_ids_mask[k]:
                decode_coordinate_list.append(past_agent_traj[-1])
                x_cords, y_cords = pred_df['X'][pred_df['track_id'] == agent_id], pred_df['Y'][pred_df['track_id'] == agent_id]
                future_agent_traj = [(x, y) for x, y in zip(x_cords, y_cords)]
                future_traj_list.append(future_agent_traj)
            else:
                decode_coordinate_list.append(None)
                future_traj_list.append(None)
        return past_traj_list, future_traj_list, encode_coordinate_list, decode_coordinate_list
