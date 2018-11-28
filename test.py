import os
import argparse
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models import *
from utils import *

import functools
import torch.nn as nn

### Training parament setting
parser = argparse.ArgumentParser(description = '.. implementation')
parser.add_argument('--train_data', required = True, help = 'CUHKStudent')
parser.add_argument('--test_data', required = True, help = 'CUHKStudent')
parser.add_argument('--cuda', action = 'store_true', help = 'use cuda?')
parser.add_argument('--threads', type = int, default = 4, help = 'number of threads for data loader to use')
parser.add_argument('--G1_model', type=str, required=True, help='model file to use')
parser.add_argument('--G2_model', type=str, required=True, help='model file to use')
parser.add_argument('--my_layer_model', type=str, required=True, help='model file to use')
parser.add_argument('--ngf', type = int, default = 64, help = 'generator filters in first conv layer')
opt = parser.parse_args()
print(opt)

### batch size
batch_size = 1
### RGB chanels
chanels = 3
### image size
image_sz = 256

### cuda setting
if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

### uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms
cudnn.benchmark = True

G1_model_dir = "checkpoint/{}/{}".format(opt.train_data, opt.G1_model)
G_1_state_dict = torch.load(G1_model_dir)
G_1 = ResnetGenerator(chanels, chanels, opt.ngf, norm_layer=functools.partial(nn.BatchNorm2d, affine=True), use_dropout=True)
G_1.load_state_dict(G_1_state_dict)

G2_model_dir = "checkpoint/{}/{}".format(opt.train_data, opt.G2_model)
G_2_state_dict = torch.load(G2_model_dir)
G_2 = G(chanels, chanels, opt.ngf)
G_2.load_state_dict(G_2_state_dict)

my_layer_model_dir = "checkpoint/{}/{}".format(opt.train_data, opt.my_layer_model)
my_layer_state_dict = torch.load(my_layer_model_dir)
my_layer = MyLayer()
my_layer.load_state_dict(my_layer_state_dict)

root_path = "dataset/"
test_set = getTestData(root_path + opt.test_data)
testing_data_loader = DataLoader(dataset = test_set, num_workers = opt.threads, batch_size = batch_size, shuffle = False)

if not os.path.exists("result"):
        os.mkdir("result")
if not os.path.exists(os.path.join("result", opt.train_data)):
    os.mkdir(os.path.join("result", opt.train_data))
if not os.path.exists(os.path.join("result", opt.train_data, opt.test_data)):
    os.mkdir(os.path.join("result", opt.train_data, opt.test_data))

for batch in testing_data_loader:
    real_photo, mask, mean_mask, face, image_name = Variable(batch[0]), Variable(batch[2]), \
                                                 Variable(batch[3]), Variable(batch[4]), batch[5]

    mask = (mask > 0)
    mean_mask = (mean_mask > 0)

    mask_sum = mask + mean_mask
    mask = mask_sum.eq(1) + mask_sum.eq(2)
    mask = (mask == 0).float()

    face = (face > 0).float()

    if opt.cuda:
        G_1 = G_1.cuda()
        G_2 = G_2.cuda()
        real_photo = real_photo.cuda()
        face = face.cuda()
        mask = mask.cuda()
        my_layer = my_layer.cuda()

    mask_wp = real_photo * mask
    norm_wp = real_photo - mask_wp + mask
    real_photo = norm_wp

    wp = G_1(real_photo)
    wp = my_layer(wp, mask, face)
    out = G_2(wp)
    out = out.cpu()
    out_img = out.data[0]

    saveImage(out_img, "result/{}/{}/{}".format(opt.train_data, opt.test_data, image_name[0]))
