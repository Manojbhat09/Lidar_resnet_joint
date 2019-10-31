import torch


def argoverse_collate(batch):
    batch_size = len(batch)

    batch_scene = torch.empty((0))
    batch_agent_masks = []
    num_agents = []
    batch_past_list, batch_future_list = [], []
    batch_encode_coordinates, batch_decode_coordinates = [], []

    for k in range(batch_size):
        # Scene Images
        scene_image = (batch[k][0]).unsqueeze(0)
        batch_scene = torch.cat((batch_scene, scene_image), dim=0)

        # Masks
        batch_agent_masks.extend(batch[k][1])
        num_agents.append(len(batch[k][1]))

        # Past and Future Trajectories:
        batch_past_list.extend(batch[k][2])
        batch_future_list.extend(batch[k][3])

        # Batch coordinate lists:
        batch_encode_coordinates.extend(batch[k][4])
        batch_decode_coordinates.extend(batch[k][5])

    # Pad Past Trajectories:
    # Sort batch_past_list in the order of decreasing lengths so as to use pack_padded_seq later.
    sorted_agent_idxs = sorted(range(len(batch_past_list)), key=lambda k: len(batch_past_list[k]), reverse=True)
    batch_past_list = sorted(batch_past_list, key=lambda k: len(k), reverse=True)

    embed_dim = batch_past_list[0].shape[1]
    src_lens = [len(path) for path in batch_past_list]
    padded_batch_past_list = torch.zeros((len(batch_past_list), max(src_lens), embed_dim))  # [N X 20 X 2]

    for i, src_len in enumerate(src_lens):
        padded_batch_past_list[i, :src_len, :] = batch_past_list[i]

    # Pad Future Trajectories:
    tgt_lens = [len(path) for path in batch_future_list]
    padded_batch_future_list = torch.zeros((len(batch_future_list), max(tgt_lens), embed_dim))  # [N X 30 X 2]

    for i, tgt_len in enumerate(tgt_lens):
        padded_batch_future_list[i, :tgt_len, :] = batch_future_list[i]

    padded_batch_past_list = torch.FloatTensor(padded_batch_past_list)
    padded_batch_future_list = torch.FloatTensor(padded_batch_future_list)
    batch_encode_coordinates = torch.FloatTensor(batch_encode_coordinates)
    batch_decode_coordinates = torch.FloatTensor(batch_decode_coordinates)

    return (batch_scene, batch_agent_masks, padded_batch_past_list, src_lens, sorted_agent_idxs,
            padded_batch_future_list, tgt_lens, batch_encode_coordinates, batch_decode_coordinates, num_agents)
