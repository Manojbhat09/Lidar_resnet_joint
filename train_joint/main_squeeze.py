
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import yaml
import argparse 
import datetime
import time 
# import tensorflow as tf
from model import Encoder, Decoder

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
                self.label_files.append(os.path.join(each, 'labels', ind + '.label'))
                self.pc_files.append(os.path.join(each, 'lidar', ind + '.ply'))

            
            assert(len(self.label_files) == len(self.pc_files))
        

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

        main_data = torchvision.transforms.ToTensor()(pointcloud_)
        target_data = torchvision.transforms.ToTensor()(label_)
        return  main_data, target_data
    
    def __len__(self):
        return len(self.label_files)
















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
    train_log_dir = 'logs/log_with_only_dec_drop_wshuffle_full_' + current_time
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    numEpochs = 500
    
    # Comment this out if using gpu
    device = torch.device('cpu')
    # ResNet for the Map-to-Feature map
    map_transform = ResnetShallow()
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
#     loss_function = nn.CosineEmbeddingLoss(margin=0.)
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
            
            # Lidar, run through Squeezeseg
            endoded, e_skips = encoder_model(Input_pcl)
            output_pcl = decoder_model(endoded, e_skips)
#             print("lidar ", output_pcl.shape)
            
#             print("target ", target.shape)
            # Loss function
            # loss = 1-(loss_function(output_pcl, output_image)) # This was used when we use nn.Cosine similarity
            loss = loss_function(output_pcl, output_image, target) # Here y = 1 as the input embedding should always be similar, use -1 if dissimilar
            avg_loss += loss.item()
    
            loss.backward()
            optimizer.step()
            
            if batch_num % 1 ==0:
                pass
#                 with train_summary_writer.as_default():
#                     tf.summary.scalar('Iter_avg_loss', avg_loss, step=(epoch+1)*(batch_num+1))
            if batch_num % 1 == 0:
            	print('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}'.format(epoch+1, batch_num+1, avg_loss)) 
            if batch_num % 100 ==0:
                torch.save(encoder_model.state_dict(), "saved_ckpt/encoder_full_iter_{}_ep_{}.pth".format(batch_num, epoch))
                torch.save(decoder_model.state_dict(), "saved_ckpt/decoder_full_iter_{}_ep_{}.pth".format(batch_num, epoch))
                print(loss.item())
                print('='*20)
            

            torch.cuda.empty_cache()
            iters += 1