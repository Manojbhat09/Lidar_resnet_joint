# Imports
import torch
import torch.nn as nn
import numpy as np


class MultiAgentScene(nn.Module):
    '''
    MATF model
    '''

    def __init__(self, image_channels=3, agent_indim=2,
                 agent_outdim=2, npast=8, nfuture=12, embed_dim_image=32,
                 embed_dim_agent=32, embed_image_h=30, embed_image_w=30,
                 spatial_embedding_linear_hidden_dim=512, LSTM_layers=1, dropout=0.3,
                 classifier_hidden=512, noise_dim=16):

        super(MultiAgentScene, self).__init__()

        LSTM_layers = 1
        print('LSTM layer set to 1.')

        self._embed_dim_agent = embed_dim_agent
        self._embed_dim_image = embed_dim_image
        self._embed_image_h = embed_image_h
        self._embed_image_w = embed_image_w
        self._noise_dim = noise_dim  # for GAN
        self._agent_indim = agent_indim  # (x,y) coordinates
        self._LSTM_layers = LSTM_layers

        self._semantic_image_encoder = SemanticImageEncoder(
            in_channels=image_channels, out_channels=embed_dim_image)
        self._spatial_pool_agent = SpatialPoolAgent()
        self._spatial_fetch_agent = SpatialFetchAgent(encoding_dim=embed_dim_agent)
        self._agent_map_fusion = AgentsMapFusion(
            in_channels=embed_dim_image + embed_dim_agent, out_channels=embed_dim_agent)
        self._agent_encoder_lstm = AgentEncoderLSTM(input_dim=agent_indim, embedding_dim=embed_dim_agent, h_dim=embed_dim_agent,
                                                    mlp_dim=spatial_embedding_linear_hidden_dim, num_layers=LSTM_layers, dropout=dropout)
        self._agent_decoder_lstm = AgentDecoderLSTM(seq_len=nfuture, output_dim=agent_outdim, embedding_dim=embed_dim_agent + noise_dim,
                                                    h_dim=embed_dim_agent + noise_dim, num_layers=LSTM_layers, dropout=dropout)
        self._classifier = Classifier(embed_dim_agent=embed_dim_agent,
                                      classifier_hidden=classifier_hidden, dropout=dropout)
        self._resnet = resnetShallow()

        print('Multi agent scene model initiated.')

    def list2batch(self, seq):
        # assemble a list of elements to batch, batch_idx: 0th dimension
        stacked = torch.tensor(seq[0]).unsqueeze(0)  # 1 X len(past_list) X 2
        i = 1
        l = len(seq)  # total number of all agents
        while i < l:
            stacked = torch.cat((stacked, torch.tensor(seq[i]).unsqueeze(0)), 0)
            i += 1

        # stacked - dims => # Total_num_all_agents X len(past_list) X 2 | Example: Total_num_all_agents X 20 X 2
        return stacked

    def batch2list(self, batch):
        # dis-assemble batch (index 0) to a list of elements
        unstacked = torch.unbind(batch, 0)
        return unstacked

    def load_from_pretrained_deterministic(self, path='outputs/stanford/state_dict.pt'):
        print('Multi Agent Scene Warning: You are trying to load state dicts. Please make sure that'
              + ' current file corresponds the file you intend to load:  ', path)

        pretrained_dict = torch.load(path)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        difference = []
        for k in model_dict:
            if k not in pretrained_dict:
                difference.append(k)
        print(difference)
        print('Reload difference shown in G.')

        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        print('Generator parameters loaded from pretrained model.')

    def forward(self, config, num_scenes, input_list, resample=0, std=1, use_resnet=0):
        # config should be the architecture intended to run,
        # which should be in ['baseline', 'single_agent_scene', 'multi_agent', 'multi_agent_scene', 'GAN_D', 'GAN_G', 'jointGAN_D', 'jointGAN_G']
        # input_list should be a list, whose each element comes from a scene:
        # each element is a list of [scene_id, agent_id, scene_image, number_agents, past_list, future_list,
                                        # weights_list, coordinates_list, lanes, absolute_coordinate_list]

        # scene_id: str
                # agent_id: List[agent_id],  [agent1, agent2, ...]
        # scene_image: tensor   | Only a Single image maybe?
        # number_agents: scalar | Agents in scene
        # past_list: List[trajectory]
        # future_list: List[trajectory]
        # weights_list: WTF?
        # coordinates_list: TODO
        # lanes: TODO
        # absolute_coordinate_list: TODO

        # format input
        num_agents_list = []
        input_image_list = []
        past_agents_state_list = []
        future_agents_state_list = []
        input_coordinate_list = []

        for i in range(num_scenes):
            num_agents_list.append(input_list[i][3])  # number_agents
            input_image_list.append(input_list[i][2][0])  # scene_image
            past_agents_state_list.append(input_list[i][4])
            future_agents_state_list.append(input_list[i][5])
            input_coordinate_list.append(input_list[i][7])  # coordinates_list

        num_agents = sum(num_agents_list)

        #############################################################################################
        ###################################### Encode Agents ########################################

        # Process the agents input traj information:
        all_agents_past_list = []
        for i in range(num_scenes):
            for j in range(num_agents_list[i]):
                all_agents_past_list.append(past_agents_state_list[i][j])  # scene i, agent j
        # all_agents_past_batch => List: [[(x0,y0), (x1,y1),... (x20,y20)], [(x0,y0), (x1,y1),... (x20,y20)], ......] | list of trajectories
        # dims => # Total_num_all_agents X len(past_list) X 2 | Example: Total_num_all_agents X 20 X 2
        all_agents_past_batch = self.list2batch(all_agents_past_list)

        all_agents_future_list = []
        for i in range(num_scenes):
            for j in range(num_agents_list[i]):
                all_agents_future_list.append(future_agents_state_list[i][j])
        all_agents_future_batch = self.list2batch(all_agents_future_list)

        if config in ['jointGAN_D', 'GAN_D']:
            all_agents_batch = torch.cat((all_agents_past_batch, all_agents_future_batch), 2)
        else:
            # ['baseline', 'single_agent_scene', 'multi_agent', 'multi_agent_scene', 'GAN_G', 'jointGAN_G']
            all_agents_batch = all_agents_past_batch

        # Batch first = False | 2 X Total_num_all_agents X 20
        all_agents_batch = all_agents_batch.permute(2, 0, 1)
        # permute for LSTM input order

        # Encode the agent traj with LSTM
        encoder_final_h_batch = self._agent_encoder_lstm(all_agents_batch.cuda())  # [batch, 32]
        agents_indiv_batch = encoder_final_h_batch.view(
            num_agents, self._embed_dim_agent)  # [num_agents, 32]
        agents_indiv_list = self.batch2list(agents_indiv_batch)
        # agents_indiv_list: [List [lists]]. Where the total length is num_agents and each element is the positional
        # embedding (extracted from LSTM) of dimension 32 for each one of those overall agents.

        #############################################################################################
        ###################################### Encode Image ########################################

        if config in ['single_agent_scene', 'multi_agent_scene', 'GAN_G', 'jointGAN_D', 'jointGAN_G']:
            input_image_batch = self.list2batch(input_image_list)
            if use_resnet == 0:
                embed_image_batch = self._semantic_image_encoder(input_image_batch.cuda())
            else:
                embed_image_batch = self._resnet(input_image_batch.cuda())

        elif config in ['multi_agent']:
            embed_image_batch = torch.tensor(np.zeros((num_scenes, self._embed_dim_image,
                                                       self._embed_image_h, self._embed_image_w), np.float32))

        # spatial inference module
        if config in ['single_agent_scene', 'multi_agent', 'multi_agent_scene', 'GAN_G', 'jointGAN_D', 'jointGAN_G']:

            # place and pool agents into a spatial map, which is inited as 0; iter on agents
            pooled_agents_map_batch = torch.tensor(np.zeros((num_scenes, self._embed_dim_agent,
                                                             self._embed_image_h, self._embed_image_w), np.float32))  # [num_scenes X 32 X 30 X 30]
            scene_idx = 0
            agent_in_scene_idx = 0

            for agent_indiv in agents_indiv_list:  # list of encoded trajectories from LSTM
                pooled_agents_map_batch = self._spatial_pool_agent(input_grid=pooled_agents_map_batch,
                                                                   input_state=agent_indiv.view(
                                                                       1, self._embed_dim_agent, 1),
                                                                   coordinate=input_coordinate_list[scene_idx][agent_in_scene_idx], batch_idx=scene_idx)

                agent_in_scene_idx += 1
                if agent_in_scene_idx >= num_agents_list[scene_idx]:
                    # move on to next scene
                    scene_idx += 1
                    agent_in_scene_idx = 0

            # concat pooled agents map and embed image, reason on the joint grid
            fused_grid_batch = self._agent_map_fusion(input_agent=pooled_agents_map_batch.cuda(),
                                                      input_map=embed_image_batch.cuda())

            # fetch fused agents states back w.r.t. coordinates from fused map
            agents_fused_list = []
            agent_idx = 0

            for i in range(num_scenes):
                for j in range(num_agents_list[i]):
                    individual_state = agents_indiv_list[agent_idx].view(
                        1, self._embed_dim_agent, 1)  # [1 X 32 X 1]
                    agent_fused = self._spatial_fetch_agent(fused_scene=fused_grid_batch, individual_state=individual_state,
                                                            coordinate=input_coordinate_list[i][j], batch_idx=i, pretrain=False)
                    agent_idx += 1
                    agents_fused_list.append(agent_fused.view(self._embed_dim_agent))

        # final agent encodings, shape, a list of (self._embed_dim_agent,)
        if config in ['single_agent_scene', 'multi_agent', 'multi_agent_scene', 'GAN_G', 'jointGAN_D', 'jointGAN_G']:
            final_agents_encoding_list = agents_fused_list
        else:
            final_agents_encoding_list = agents_indiv_list

        # classification for D of GANs, a list of scores
        if config in ['GAN_D', 'jointGAN_D']:
            final_agents_encoding_batch = self.list2batch(final_agents_encoding_list)
            classified = self._classifier(final_agents_encoding_batch.cuda())
            return classified, all_agents_future_batch, num_agents

        #################################################################
        ############################ DECODER ############################
        # prediction of future trajectories, using decoder
        else:

            if resample == 0:

                # concat with noise
                noise = get_noise(shape=(self._noise_dim,), noise_type='gaussian')
                if config not in ['GAN_G', 'jointGAN_G']:
                    noise = 0.0 * noise

                # relative position of the last time stamp in past
                all_agents_last_rel = (
                    all_agents_past_batch[:, :, -1] - all_agents_past_batch[:, :, -2])
                all_agents_last_rel = all_agents_last_rel.view(num_agents, self._agent_indim)

                noised_list = []
                for agent in final_agents_encoding_list:
                    noised_agent = torch.cat((agent, noise), 0)
                    noised_list.append(noised_agent)
                decoder_h = self.list2batch(noised_list).view(
                    1, num_agents, self._embed_dim_agent + self._noise_dim).cuda()

                decoder_c = torch.zeros(
                    1, num_agents, self._embed_dim_agent + self._noise_dim).cuda()
                state_tuple = (decoder_h, decoder_c)

                # decode
                decoded, final_decoder_h = self._agent_decoder_lstm(last_pos_rel=all_agents_last_rel.cuda(),
                                                                    state_tuple=state_tuple,
                                                                    start_pos=all_agents_past_batch[:, :, -1],
                                                                    start_vel=all_agents_past_batch[:, :, -1] -
                                                                    all_agents_past_batch[:, :, -2])
                decoded = decoded.permute(1, 2, 0)
                return decoded, all_agents_future_batch, num_agents

            # resample for validation evaluation for GANs
            else:

                outputs_samples = []
                for resample_it in range(resample):

                    # concat with noise
                    noise = std * get_noise(shape=(self._noise_dim,), noise_type='gaussian')

                    all_agents_last_rel = (all_agents_past_batch[:, :, -1]
                                           - all_agents_past_batch[:, :, -2])\
                        .view(num_agents, self._agent_indim)
                    # relative position of the last time stamp in past

                    noised_list = []
                    for agent in final_agents_encoding_list:
                        noised_agent = torch.cat((agent, noise), 0)
                        noised_list.append(noised_agent)
                    decoder_h = self.list2batch(noised_list).view(
                        1, num_agents, self._embed_dim_agent + self._noise_dim).cuda()

                    decoder_c = torch.zeros(
                        1, num_agents, self._embed_dim_agent + self._noise_dim).cuda()
                    state_tuple = (decoder_h.detach(), decoder_c.detach())
                    # no BP

                    # decode
                    decoded, final_decoder_h = self._agent_decoder_lstm(last_pos_rel=all_agents_last_rel.cuda(),
                                                                        state_tuple=state_tuple,
                                                                        start_pos=all_agents_past_batch[:, :, -1],
                                                                        start_vel=all_agents_past_batch[:, :, -1] -
                                                                        all_agents_past_batch[:, :, -2])
                    decoded = decoded.permute(1, 2, 0)

                    outputs_samples.append(decoded.detach())
                return outputs_samples, all_agents_future_batch, num_agents
