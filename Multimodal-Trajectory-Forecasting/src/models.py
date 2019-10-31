# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from utils import conv2DBatchNormRelu, deconv2DBatchNormRelu


class MATF(nn.Module):
    """
    A standard Encoder-Decoder architecture.
    """

    def __init__(self, scene_encoder, agent_lstm, spatial_pool_agent, agent_map_fusion, spatial_fetch_agent, agent_decoder_lstm, device,
                 noise_dim=16, resample=None, generator=None):
        super(MATF, self).__init__()

        self.scene_encoder = scene_encoder
        self.agent_lstm = agent_lstm
        self.spatial_pool_agent = spatial_pool_agent
        self.agent_map_fusion = agent_map_fusion
        self.spatial_fetch_agent = spatial_fetch_agent
        self.agent_decoder_lstm = agent_decoder_lstm
        self.device = device
        self.generator = generator
        self.noise_dim = noise_dim
        self.resample = resample

    def forward(self, scene_images, agent_masks, past_trajs, src_lens, sorted_agent_idxs, encode_coords, decode_coords, num_agents):
        past_trajs = past_trajs.permute(1, 0, 2).to(self.device)

        fused_agent_encodings = self.encode(scene_images, agent_masks, past_trajs, src_lens, sorted_agent_idxs, encode_coords,
                                            decode_coords, num_agents)

        return self.decode(fused_agent_encodings, agent_masks, past_trajs, decode_coords)

    def encode(self, scene_images, agent_masks, past_trajs, src_lens, sorted_agent_idxs, encode_coords, decode_coords, num_agents):

        # Encode Scene and Past Agent Paths
        scene_encodings = self.scene_encoder(scene_images)
        agent_lstm_encodings = self.agent_lstm(past_trajs, src_lens).squeeze(0)  # [Total agents in batch X 32]

        # Reorder the sorted agent_lstm_encodings back to original unsorted order:
        reordered_encodings = torch.zeros_like(agent_lstm_encodings)
        for k, agent_idx in enumerate(sorted_agent_idxs):
            reordered_encodings[agent_idx, :] = agent_lstm_encodings[k, :]
        agent_lstm_encodings = reordered_encodings

        # Pool the agent encodings on a zero initalized map tensor:
        pooled_agents_map_batch = torch.zeros_like(scene_encodings)  # [B X 32 X 30 X 30]
        pooled_agents_map_batch = self.spatial_pool_agent(pooled_agents_map_batch, agent_lstm_encodings,
                                                          encode_coords, num_agents)

        # Concat pooled agents map and embed scene image to get a fused grid:
        fused_grid_batch = self.agent_map_fusion(input_agent=pooled_agents_map_batch, input_map=scene_encodings)

        # Fetch fused agents states back w.r.t. coordinates from fused map grid:
        fused_agent_encodings = self.spatial_fetch_agent(fused_grid_batch, agent_lstm_encodings,
                                                         decode_coords, agent_masks, num_agents)
        return fused_agent_encodings

    def decode(self, fused_agent_encodings, agent_masks, past_trajs, decode_coords):
        if self.resample is None:
            # concat with noise
            noise = torch.zeros(self.noise_dim, device=self.device)
            noise = noise.repeat(fused_agent_encodings.shape[0], 1)

            fused_noise_encodings = torch.cat((fused_agent_encodings, noise), dim=1)

            decoder_h = fused_noise_encodings.unsqueeze(0)
            decoder_c = torch.zeros_like(decoder_h)
            state_tuple = (decoder_h, decoder_c)

            # relative position of the last time stamp in past
            all_agents_last_rel = []
            for k in range(past_trajs.shape[1]):
                if agent_masks[k]:
                    all_agents_last_rel.append(past_trajs[-1, k, :] - past_trajs[-2, k, :])

            all_agents_last_rel = torch.stack(all_agents_last_rel, dim=0)  # [Num agents X 2]

            predicted_trajs, final_decoder_h = self.agent_decoder_lstm(last_pos_rel=all_agents_last_rel,
                                                                       state_tuple=state_tuple,
                                                                       start_pos=decode_coords,
                                                                       start_vel=None
                                                                       # all_agents_past_batch[-1, :, :] - all_agents_past_batch[-2, :, :]
                                                                       )
            predicted_trajs = predicted_trajs.permute(1, 0, 2)  # [B X L X 2]

            return predicted_trajs


class AgentEncoderLSTM(nn.Module):

    def __init__(self, device, input_dim=2, embedding_dim=32, h_dim=32, mlp_dim=512, num_layers=1, dropout=0.3):
        super(AgentEncoderLSTM, self).__init__()

        self.device = device
        self._mlp_dim = mlp_dim
        self._h_dim = h_dim
        self._embedding_dim = embedding_dim
        self._num_layers = num_layers
        self._input_dim = input_dim

        self._encoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)
        self._spatial_embedding = nn.Linear(input_dim, embedding_dim)

    def init_hidden(self, batch_size):
        # h_0, c_0 of shape (num_layers * num_directions, batch, hidden_size)
        # batch size should be number of agents in the whole batch, instead of number of scenes
        return (
            torch.zeros(self._num_layers, batch_size, self._h_dim).to(self.device),
            torch.zeros(self._num_layers, batch_size, self._h_dim).to(self.device)
        )

    def forward(self, obs_traj, src_lens):
        # reset the LSTM hidden state. Must be done before you run a new batch. Otherwise the LSTM will treat
        # a new batch as a continuation of a sequence
        batch_size = obs_traj.size(1)
        self.hidden = self.init_hidden(batch_size)

        # convert to relative, as Social GAN do
        rel_curr_ped_seq = torch.zeros_like(obs_traj).to(self.device)
        rel_curr_ped_seq[1:, :, :] = obs_traj[1:, :, :] - obs_traj[:-1, :, :]

        # Encode observed Trajectory
        batch = obs_traj.size(1)
        obs_traj_embedding = self._spatial_embedding(rel_curr_ped_seq.view(-1, self._input_dim))
        obs_traj_embedding = obs_traj_embedding.view(-1, batch, self._embedding_dim)

        obs_traj_embedding = nn.utils.rnn.pack_padded_sequence(obs_traj_embedding, src_lens)
        output, (hidden_final, cell_final) = self._encoder(obs_traj_embedding, self.hidden)

        return hidden_final


class AgentDecoderLSTM(nn.Module):
    '''
    This part of the code is revised from Social GAN's paper for fair comparison
    '''

    def __init__(self, seq_len, device, output_dim=2, embedding_dim=32, h_dim=32, num_layers=1, dropout=0.0):
        super(AgentDecoderLSTM, self).__init__()

        self._seq_len = seq_len
        self.device = device
        self._h_dim = h_dim
        self._embedding_dim = embedding_dim

        self._decoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)
        self._spatial_embedding = nn.Linear(output_dim, embedding_dim)
        self._hidden2pos = nn.Linear(h_dim, output_dim)

    def relative_to_abs(self, rel_traj, start_pos=None):
        """
        Inputs:
        - rel_traj: pytorch tensor of shape (seq_len, batch, 2)
        - start_pos: pytorch tensor of shape (batch, 2)
        Outputs:
        - abs_traj: pytorch tensor of shape (seq_len, batch, 2)
        """
        # in our case, start pos is always 0
        if start_pos is None:
            start_pos = torch.zeros_like(rel_traj[0]).to(self.device)

        rel_traj = rel_traj.permute(1, 0, 2)
        displacement = torch.cumsum(rel_traj, dim=1)
        start_pos = torch.unsqueeze(start_pos, dim=1)
        abs_traj = displacement + start_pos

        return abs_traj.permute(1, 0, 2)

    def forward(self, last_pos_rel, state_tuple, start_pos=None, start_vel=None):
        """
        Inputs:
        - last_pos_rel: Tensor of shape (batch, 2)
        - state_tuple: (hh, ch) each tensor of shape (num_layers, batch, h_dim)
        Output:
        - pred_traj: tensor of shape (self.seq_len, batch, 2)
        """
        batch = last_pos_rel.size(0)
        pred_traj_fake_rel = []
        decoder_input = self._spatial_embedding(last_pos_rel)
        decoder_input = decoder_input.view(1, batch, self._embedding_dim)

        for _ in range(self._seq_len):
            output, state_tuple = self._decoder(decoder_input, state_tuple)
            predicted_rel_pos = self._hidden2pos(output.view(-1, self._h_dim))
            pred_traj_fake_rel.append(predicted_rel_pos.view(batch, -1))  # [B X 2]

            # For next decode step:
            decoder_input = self._spatial_embedding(predicted_rel_pos)
            decoder_input = decoder_input.view(1, batch, self._embedding_dim)

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)  # [L X B X 2]

        return self.relative_to_abs(pred_traj_fake_rel, start_pos), state_tuple[0]


class AgentsMapFusion(nn.Module):

    def __init__(self, in_channels=32 + 32, out_channels=32):
        super(AgentsMapFusion, self).__init__()

        self._conv1 = conv2DBatchNormRelu(in_channels=in_channels, n_filters=out_channels,
                                          k_size=3, stride=1, padding=1, dilation=1)
        self._pool1 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self._conv2 = conv2DBatchNormRelu(in_channels=out_channels, n_filters=out_channels,
                                          k_size=3, stride=1, padding=1, dilation=1)
        self._pool2 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self._conv3 = conv2DBatchNormRelu(in_channels=out_channels, n_filters=out_channels,
                                          k_size=4, stride=1, padding=1, dilation=1)

        self._deconv2 = deconv2DBatchNormRelu(in_channels=out_channels, n_filters=out_channels,
                                              k_size=4, stride=2, padding=1)

    def forward(self, input_agent, input_map):
        cat = torch.cat((input_map, input_agent), 1)

        conv1 = self._conv1.forward(cat)
        conv2 = self._conv2.forward(self._pool1.forward(conv1))
        conv3 = self._conv3.forward(self._pool2.forward(conv2))

        up2 = self._deconv2.forward(conv2)
        up3 = F.interpolate(conv3, scale_factor=5)

        features = conv1 + up2 + up3
        return features


class SpatialPoolAgent(nn.Module):

    def __init__(self, encoding_dim=32):
        super(SpatialPoolAgent, self).__init__()
        self._encoding_dim = encoding_dim

    def transform_coordinates(self, coordinates):
        # TODO: Transform the orignal coords to a 30x30 grid.
        return [0, 0]

    def forward(self, input_grid, agent_encodings, encode_coordinates, num_agents):
        self.input_grid = input_grid

        scene_idx = 0
        agent_in_scene_idx = 0

        for k in range(len(agent_encodings)):
            self.pool_single_agent(agent_state=agent_encodings[k, :].view(1, self._encoding_dim, 1),
                                   agent_coordinates=encode_coordinates[k],
                                   batch_idx=scene_idx)
            agent_in_scene_idx += 1

            if agent_in_scene_idx >= num_agents[scene_idx]:
                scene_idx += 1  # Move to next scene
                agent_in_scene_idx = 0

        return self.input_grid

    def pool_single_agent(self, agent_state, agent_coordinates, batch_idx):

        # Accessing the i_th batch_idx (scene) at the agent's transformed coordinates.
        agent_coordinates = self.transform_coordinates(agent_coordinates)
        ori_state = self.input_grid[batch_idx, :, agent_coordinates[0], agent_coordinates[1]].clone()

        # Pool the states:
        pooled_state = torch.max(ori_state, agent_state[0, :, 0])

        # Update the input grid
        self.input_grid[batch_idx, :, agent_coordinates[0], agent_coordinates[1]] = pooled_state
        return


class SpatialFetchAgent(nn.Module):

    def __init__(self, encoding_dim=32):
        super(SpatialFetchAgent, self).__init__()
        self._encoding_dim = encoding_dim

    def transform_coordinates(self, coordinates):
        # TODO: Transform the orignal coords to a 30x30 grid.
        return [0, 0]

    def forward(self, fused_scene, agent_encodings, decode_coordinates, agent_masks, num_agents, pretrain=False):
        self.fused_scene = fused_scene

        fused_agent_encodings = torch.empty((0))
        scene_idx, agent_in_scene_idx = 0, 0
        future_agent_idx = 0

        for k in range(len(agent_encodings)):
            if agent_masks[k]:
                single_fused_agent = self.fetch_single_agent(agent_state=agent_encodings[k, :].view(1, self._encoding_dim, 1),
                                                             agent_coordinates=decode_coordinates[future_agent_idx],
                                                             batch_idx=scene_idx,
                                                             pretrain=pretrain)

                single_fused_agent = single_fused_agent.view(1, self._encoding_dim)
                fused_agent_encodings = torch.cat((fused_agent_encodings, single_fused_agent), dim=0)
                future_agent_idx += 1

            agent_in_scene_idx += 1
            if agent_in_scene_idx >= num_agents[scene_idx]:
                scene_idx += 1  # Move to next scene
                agent_in_scene_idx = 0

        return fused_agent_encodings

    def fetch_single_agent(self, agent_state, agent_coordinates, batch_idx, pretrain):
        # TODO Warning : Check the correctness of the following function!!!
        # pretrain == True for pre-training, no residual from fusion for details of other params, refer to SpatialPoolAgent

        coordinates = self.transform_coordinates(agent_coordinates)
        if pretrain:
            output = agent_state[0, :, :]
        else:
            fused_state = self.fused_scene[batch_idx, :, coordinates[0], coordinates[1]]  # Vector [32]
            fused_state_dim2 = fused_state.view(self._encoding_dim, 1)  # [32 X 1]
            output = fused_state_dim2 + agent_state[0, :, :]

        return output


class ResnetShallow(nn.Module):

    def __init__(self):  # Output Size: 30 * 30
        super(ResnetShallow, self).__init__()

        self.trunk = torchvision.models.resnet18(pretrained=True)

        self.upscale3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), )

        self.upscale4 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 7, stride=4, padding=3, output_padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), )

        self.shrink = conv2DBatchNormRelu(in_channels=384, n_filters=32,
                                          k_size=1, stride=1, padding=0)

    def forward(self, image):
        x = self.trunk.conv1(image)
        x = self.trunk.bn1(x)
        x = self.trunk.relu(x)
        x = self.trunk.maxpool(x)

        x = self.trunk.layer1(x)
        x2 = self.trunk.layer2(x)  # /8 the size
        x3 = self.trunk.layer3(x2)  # 16
        x4 = self.trunk.layer4(x3)  # 32

        x3u = self.upscale3(x3.detach())
        x4u = self.upscale4(x4.detach())

        xall = torch.cat((x2.detach(), x3u, x4u), dim=1)
        xall = F.interpolate(xall, size=(30, 30))
        output = self.shrink(xall)

        return output
