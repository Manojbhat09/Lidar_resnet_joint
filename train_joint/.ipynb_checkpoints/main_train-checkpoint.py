
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import yaml
import argparse 
import datetime
import time 
import tensorflow as tf
# import tensorflow as tf
from dataset import JointDataset
# from squeezesegMOD import Backbone
from model import Encoder, Decoder

# TODO use this loss incorporating the ResNet Loss here
from ShallowResNet import ResNet18Loss as ResNetTensor

# TODO Each paths are not well defined. Set the default values as where it really are.

parser = argparse.ArgumentParser("./main_train.py")
parser.add_argument(
  '--dataset', '-d',
  type=str,
  default= 'data/',
  help='Dataset to train with',
)
parser.add_argument(
  '--arch_cfg', '-ac',
  type=str,
  default='config/arch/squeezesegV2.yaml',
  help='Architecture yaml cfg file. See /config/arch for sample. No default!',
)
parser.add_argument(
  '--data_cfg', '-dc',
  type=str,
  default='config/labels/argoverse.yaml',
  help='Classification yaml cfg file. See /config/labels for sample. No default!',
)
parser.add_argument(
  '--log', '-l',
  type=str,
  default="log_"+ str(time.time()),
  # default='logs/'+ datetime.datetime.now().strftime("%Y-%-m-%d-%H:%M") + '/',
  help='Directory to put the log data. Default: ~/logs/date+time'
)
parser.add_argument(
  '--pretrained', '-p',
  type=str,
  default=None,
  help='Directory to get the pretrained model. If not passed, do from scratch!'
)
FLAGS, unparsed = parser.parse_known_args()

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def get_batches(data_pcl):
    print(data_pcl.shape)
    xyz =data_pcl[:,0:3]
    xyz_min = np.amin(xyz, axis=0, keepdims=True)
    xyz_max = np.amax(xyz, axis=0, keepdims=True)
    block_size = (2 * (xyz_max[0, 0] - xyz_min[0, 0]), 2 * (xyz_max[0, 1] - xyz_min[0, 1]) ,  2 * (xyz_max[0, -1] - xyz_min[0, -1]))
    
    xyz_blocks = np.floor((xyz - xyz_min) / block_size).astype(np.int)

    #print('{}-Collecting points belong to each block...'.format(datetime.now(), xyzrcof.shape[0]))
    blocks, point_block_indices, block_point_counts = np.unique(xyz_blocks, return_inverse=True,
                                                                return_counts=True, axis=0)
    block_point_indices = np.split(np.argsort(point_block_indices), np.cumsum(block_point_counts[:-1]))
    #print('{}-{} is split into {} blocks.'.format(datetime.now(), dataset, blocks.shape[0]))

    block_to_block_idx_map = dict()
    for block_idx in range(blocks.shape[0]):
        block = (blocks[block_idx][0], blocks[block_idx][1])
        block_to_block_idx_map[(block[0], block[1])] = block_idx

    # merge small blocks into one of their big neighbors
    block_point_count_threshold = max_point_num / 3
    #print("block_point_count_threshold",block_point_count_threshold)
    nbr_block_offsets = [(0, 1), (1, 0), (0, -1), (-1, 0), (-1, 1), (1, 1), (1, -1), (-1, -1)]
    block_merge_count = 0
    for block_idx in range(blocks.shape[0]):
        if block_point_counts[block_idx] >= block_point_count_threshold:
            #print(block_idx, block_point_counts[block_idx])

            continue


        block = (blocks[block_idx][0], blocks[block_idx][1])
        for x, y in nbr_block_offsets:
            nbr_block = (block[0] + x, block[1] + y)
            if nbr_block not in block_to_block_idx_map:
                continue

            nbr_block_idx = block_to_block_idx_map[nbr_block]
            if block_point_counts[nbr_block_idx] < block_point_count_threshold:
                continue


            #print(block_idx, nbr_block_idx, block_point_counts[nbr_block_idx])

            block_point_indices[nbr_block_idx] = np.concatenate(
                [block_point_indices[nbr_block_idx], block_point_indices[block_idx]], axis=-1)
            block_point_indices[block_idx] = np.array([], dtype=np.int)
            block_merge_count = block_merge_count + 1
            break
    #print('{}-{} of {} blocks are merged.'.format(datetime.now(), block_merge_count, blocks.shape[0]))

    idx_last_non_empty_block = 0
    for block_idx in reversed(range(blocks.shape[0])):
        if block_point_indices[block_idx].shape[0] != 0:
            idx_last_non_empty_block = block_idx
            break

    # uniformly sample each block
    for block_idx in range(idx_last_non_empty_block + 1):
        point_indices = block_point_indices[block_idx]
        if point_indices.shape[0] == 0:
            continue

        #print(block_idx, point_indices.shape)
        block_points = xyz[point_indices]
        block_min = np.amin(block_points, axis=0, keepdims=True)
        xyz_grids = np.floor((block_points - block_min) / grid_size).astype(np.int)
        grids, point_grid_indices, grid_point_counts = np.unique(xyz_grids, return_inverse=True,
                                                                 return_counts=True, axis=0)
        grid_point_indices = np.split(np.argsort(point_grid_indices), np.cumsum(grid_point_counts[:-1]))
        grid_point_count_avg = int(np.average(grid_point_counts))
        point_indices_repeated = []
        for grid_idx in range(grids.shape[0]):
            point_indices_in_block = grid_point_indices[grid_idx]
            repeat_num = math.ceil(grid_point_count_avg / point_indices_in_block.shape[0])
            if repeat_num > 1:
                point_indices_in_block = np.repeat(point_indices_in_block, repeat_num)
                np.random.shuffle(point_indices_in_block)
                point_indices_in_block = point_indices_in_block[:grid_point_count_avg]
            point_indices_repeated.extend(list(point_indices[point_indices_in_block]))
        block_point_indices[block_idx] = np.array(point_indices_repeated)
        block_point_counts[block_idx] = len(point_indices_repeated)

    idx = 0
    for block_idx in range(idx_last_non_empty_block + 1):
        point_indices = block_point_indices[block_idx]
        if point_indices.shape[0] == 0:
            continue

        block_point_num = point_indices.shape[0]
        block_split_num = int(math.ceil(block_point_num * 1.0 / max_point_num))
        point_num_avg = int(math.ceil(block_point_num * 1.0 / block_split_num))
        point_nums = [point_num_avg] * block_split_num
        point_nums[-1] = block_point_num - (point_num_avg * (block_split_num - 1))
        starts = [0] + list(np.cumsum(point_nums))

        np.random.shuffle(point_indices)
        block_points = xyz[point_indices]


        block_min = np.amin(block_points, axis=0, keepdims=True)
        block_max = np.amax(block_points, axis=0, keepdims=True)
        #block_center = (block_min + block_max) / 2
        #block_center[0][-1] = block_min[0][-1]
        #block_points = block_points - block_center  # align to block bottom center
        x, y, z = np.split(block_points, (1, 2), axis=-1)

        block_xzyrgbi = np.concatenate([x, z, y, i[point_indices]], axis=-1)

        for block_split_idx in range(block_split_num):
            start = starts[block_split_idx]
            point_num = point_nums[block_split_idx]
            #print(block_split_num, block_split_idx, point_num )



            end = start + point_num
            idx_in_batch = idx % batch_size
            data[idx_in_batch, 0:point_num, ...] = block_xzyrgbi[start:end, :]
            data_num[idx_in_batch] = point_num
            indices_split_to_full[idx_in_batch, 0:point_num] = point_indices[start:end]

            #print("indices_split_to_full", idx_in_batch, point_num, indices_split_to_full)

            if  (block_idx == idx_last_non_empty_block and block_split_idx == block_split_num - 1): #Last iteration

                item_num = idx_in_batch + 1
                
            idx = idx + 1
    return data, data_num, indices_split_to_full, item_num, all_label_pred, indices_for_prediction


# TODO defined for the main-limitation.
if __name__ == "__main__":
	
    # TODO I think some of the settings are not governed through this setting. You should have to specify whether those are OK or not.

    # Model parameters
    # Hyper parameters 
    loss = "xentropy"       # must be either xentropy or iou
    max_epochs = 150
    lr = 0.001              # sgd learning rate
    wup_epochs = 0.01       # warmup during first XX epochs (can be float)
    momentum = 0.9          # sgd momentum
    lr_decay = 0.99         # learning rate decay per epoch after initial cycle (from min lr)
    w_decay = 0.0001        # weight decay
    batch_size = 8          # batch size
    report_batch = 1        # every x batches, report loss
    report_epoch = 1        # every x epochs, report validation set
    epsilon_w = 0.001       # class weight w = 1 / (content + epsilon_w)
    workers = 0            # number of threads to get data
    dropout = 0.01
    OS = 16 # output stride (only horizontally)
    bn_d = 0.01
    train = True # train backbone?
    extra = False


    # TODO This JointDataset does not seperates Valid data. It should be more specified for the training purpose.
    # Question: how could we make this valid not incorporated in the training process?
    train_dataset = JointDataset("/mnt/sdb1/shpark/dataset/argoverse/argoverse11/argoverse-forecasting-from-tracking/train/train1/")
#   print("Data Done")

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=5, shuffle=True)
#   print(next(iter(trainloader)))
#   assert len(trainloader) > 0
    	# Load labels
#   	break
#   quit()

    # ### Load the model 
    #   # Load encoder only
    try:
        with open(FLAGS.arch_cfg, 'r') as pnt:
            ARCH = yaml.safe_load(pnt)

    except Exception as e:
        print(e)

    # TODO Consider the inputs of the Backbone Module.
#     model = Backbone(ARCH["backbone"])
    encoder_model = Encoder(ARCH["backbone"])
    decoder_model = Decoder(ARCH["backbone"])

    ### Training function 
    # empty the cache to train now
    torch.cuda.empty_cache()
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu') # TODO Specify the GPU number, if possible. We aren't using the our own private server.
    current_time = str(datetime.datetime.now().timestamp())
    train_log_dir = 'logs/log_with_only_dec_drop_wshuffle_full_real_' + current_time
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    numEpochs = 500
    
    # Comment this out if using gpu
    device = torch.device('cpu')
    # ResNet for the Map-to-Feature map
    map_transform = ResNetTensor()
    map_transform.to(device)
    # TODO Consider to freeze the parameters or not.

    # switch to train mode
    encoder_model.train()
    decoder_model.train()
    encoder_model.apply(init_weights)
    decoder_model.apply(init_weights)
    encoder_model.to(device)
    decoder_model.to(device)
    # model.load_state_dict(torch.load("saved/iter_600_ep_0.pth"))

    # loss function has a input vector, use CosineEmbeddingLoss for 2d loss
    loss_function = nn.CosineEmbeddingLoss(margin=0.)
    optimizer = torch.optim.SGD(
        list(encoder_model.parameters())+ list(decoder_model.parameters()),
        lr=0.01,weight_decay=w_decay,momentum=0.9
    )


    ### Training loop
    for epoch in range(numEpochs):	
        avg_loss = 0
        iters = 0
        for batch_num, (data_image, data_pcl) in enumerate(trainloader):  # TODO Setup to feed the Labels from train_loader
            optimizer.zero_grad()
#             print(data_image.shape)
#             print("pcl shape ",data_pcl.shape)
#             print("Fix Jimins code shape",data_image.shape)
            
            Input_pcl = data_pcl.to(device)
            Input_image = data_image.to(device)
            
            # Lidar, run through Squeezeseg
            endoded, e_skips = encoder_model(Input_pcl)
            output_pcl = decoder_model(endoded, e_skips)
#             print("lidar ", output_pcl.shape)
            
            # Image, run through Resnet block
            output_image = map_transform(Input_image)
#             print("img ", output_image.shape)
            
            target = torch.from_numpy(np.ones((output_image.shape[-2], output_image.shape[-1])))
            target = target.unsqueeze(dim=0).unsqueeze(dim=0)
#             print("target ", target.shape)
            # Loss function
            # loss = 1-(loss_function(output_pcl, output_image)) # This was used when we use nn.Cosine similarity
            loss = loss_function(output_pcl,output_image, target) # Here y = 1 as the input embedding should always be similar, use -1 if dissimilar
            avg_loss += loss.item()
    
            loss.backward()
            optimizer.step()
            
            if batch_num % 1 ==0:
                with train_summary_writer.as_default():
                    tf.summary.scalar('Iter_avg_loss', avg_loss, step=(epoch+1)*(batch_num+1))
            if batch_num % 1 == 0:
            	print('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}'.format(epoch+1, batch_num+1, avg_loss)) 
            if batch_num % 100 ==0:
                torch.save(encoder_model.state_dict(), "saved_ckpt/encoder_full_real_iter_{}_ep_{}.pth".format(batch_num, epoch))
                torch.save(decoder_model.state_dict(), "saved_ckpt/decoder_full_real_iter_{}_ep_{}.pth".format(batch_num, epoch))
                print(loss.item())
                print('='*20)
            

            torch.cuda.empty_cache()
            iters += 1