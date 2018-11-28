import os
import argparse
from math import log10
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models import *
from utils import getTrainData, getTestData
import functools

### Training parament setting
parser = argparse.ArgumentParser(description = '.. implementation')
parser.add_argument('--dataset', required = True, help = 'CUHKStudent')
parser.add_argument('--nEpochs', type = int, default = 200, help = 'number of epochs to train the model')
parser.add_argument('--ngf', type = int, default = 64, help = 'generator filters in first conv layer')
parser.add_argument('--ndf', type = int, default = 64, help = 'discriminator filters in first conv layer')
parser.add_argument('--lr', type = float, default = 0.0002, help = 'Learning Rate. Default = 0.002')
parser.add_argument('--beta1', type = float, default = 0.5, help = 'beta1 for adam. default = 0.5')
parser.add_argument('--cuda', action = 'store_true', help = 'use cuda?')
parser.add_argument('--threads', type = int, default = 4, help = 'number of threads for data loader to use')
parser.add_argument('--seed', type = int, default = 233, help = 'random seed to use. Default=233')
parser.add_argument('--lamb', type=int, default=100, help='weight on L1 term in objective')

opt = parser.parse_args()
print(opt)
### batch size
batch_size = 1
### RGB chanels
chanels = 3
### image h x w
image_sz = 256

### cuda setting
if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")
### uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms
cudnn.benchmark = True

### random seed
torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

### Load data
print('----> Loading data......')
root_path = "dataset/"
train_set = getTrainData(root_path + opt.dataset)
test_set = getTestData(root_path + opt.dataset)
training_data_loader = DataLoader(dataset = train_set, num_workers = opt.threads, batch_size = batch_size, shuffle = True)
testing_data_loader = DataLoader(dataset = test_set, num_workers = opt.threads, batch_size = batch_size, shuffle = False)

### Init models
print('----> Initialize models......')

G_1 = ResnetGenerator(chanels, chanels, opt.ngf, norm_layer=functools.partial(nn.BatchNorm2d, affine=True), use_dropout=True)
G_1.apply(weightsInit)

G_2 = G(chanels, chanels, opt.ngf)
G_2.apply(weightsInit)

sketch_D = D(chanels, chanels, opt.ngf)
sketch_D.apply(weightsInit)

w_D = D(chanels, chanels, opt.ngf)
w_D.apply(weightsInit)

my_layer = MyLayer()

### Init setting
loss_BCE = nn.BCELoss()
loss_MSE = nn.MSELoss()
loss_L1 = nn.L1Loss()

real_photo = Variable(torch.FloatTensor(batch_size, chanels, image_sz, image_sz))
real_sketch = Variable(torch.FloatTensor(batch_size, chanels, image_sz, image_sz))
mask = Variable(torch.FloatTensor(batch_size, chanels, image_sz, image_sz))
face = Variable(torch.FloatTensor(batch_size, chanels, image_sz, image_sz))
photo_label = Variable(torch.FloatTensor(batch_size))
sketch_label = Variable(torch.FloatTensor(batch_size))
real_label = True
fake_label = False

if opt.cuda:
    G_1 = G_1.cuda()
    G_2 = G_2.cuda()
    sketch_D = sketch_D.cuda()
    w_D = w_D.cuda()

    my_layer = my_layer.cuda()

    loss_BCE = loss_BCE.cuda()
    loss_MSE = loss_MSE.cuda()
    loss_L1 = loss_L1.cuda()
    real_photo = real_photo.cuda()
    real_sketch = real_sketch.cuda()
    mask = mask.cuda()
    face = face.cuda()
    photo_label = photo_label.cuda()
    sketch_label = sketch_label.cuda()

### optimizer
optimizer_G_1 = optim.Adam(G_1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_G_2 = optim.Adam(G_2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_D  = optim.Adam(sketch_D.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
# optimizer_w_D  = optim.Adam(w_D.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_my_layer = optim.Adam(my_layer.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

def train(epoch):
    for iteration, batch in enumerate(training_data_loader, 1):

        real_photo_cpu, real_sketch_cpu, mask_cpu, mean_mask_cpu, face_cpu = batch[0], batch[1], batch[2], batch[3], batch[4]
        real_photo.data.resize_(real_photo_cpu.size()).copy_(real_photo_cpu)
        real_sketch.data.resize_(real_sketch_cpu.size()).copy_(real_sketch_cpu)

        mask_cpu = (mask_cpu > 0)
        mean_mask_cpu = (mean_mask_cpu > 0)

        mask_sum = mask_cpu + mean_mask_cpu
        mask_cpu = mask_sum.eq(1) + mask_sum.eq(2)
        mask_cpu = (mask_cpu == 0)
        mask.data.resize_(mask_cpu.size()).copy_(mask_cpu)

        face_cpu = (face_cpu > 0)
        face.data.resize_(face_cpu.size()).copy_(face_cpu)
        
        mask_photo = real_photo * mask
        norm_photo = real_photo - mask_photo + mask
        real_photo.data.copy_(norm_photo.data)

        # print(mask)
        ############################
        # (1) Update sD network: maximize log(sD(y)) + log(1 - sD(G(y)))
        # (1) Update pD network: maximize log(sD(x)) + log(1 - sD(G(x)))
        ###########################
        sketch_D.volatile = False
        sketch_D.zero_grad()

        wp = G_1(real_photo)

        ####
        wp = my_layer(wp, mask, face)

        p_fake_sketch = G_2(wp)

        # train with fake
        p_fake_D_out = sketch_D(p_fake_sketch.detach())
        sketch_label.data.resize_(p_fake_D_out.size()).fill_(fake_label)

        err_d_fake = loss_BCE(p_fake_D_out, sketch_label)
        err_d_fake.backward()

        # train with real
        s_real_D_out = sketch_D(real_sketch)
        sketch_label.data.resize_(s_real_D_out.size()).fill_(real_label)

        err_d_fake = loss_BCE(s_real_D_out, sketch_label)
        err_d_fake.backward()

        optimizer_D.step()
        ############################
        # (2) Update sG network: maximize log(D(sG(y))) + L1(y,sG(y))
        # (2) Update pG network: maximize log(D(pG(x))) + L1(x,pG(x))
        ###########################
        sketch_D.volatile = True
        G_1.zero_grad()
        G_2.zero_grad()

        wp = G_1(real_photo)

        #####
        wp = my_layer(wp, mask, face)

        p_fake_sketch = G_2(wp)
        p_fake_D_out = sketch_D(p_fake_sketch)
        sketch_label.data.resize_(p_fake_D_out.size()).fill_(real_label)

        err_g = loss_BCE(p_fake_D_out, sketch_label) \
                + opt.lamb * loss_L1(p_fake_sketch, real_sketch)
        err_g.backward()

        out1 = p_fake_D_out.data.mean()

        optimizer_G_1.step()
        optimizer_G_2.step()
        
        optimizer_my_layer.step()

        ############################
        # (3) Update sG1 network: minimize L1(sG1(x),pG1(y))
        # (3) Update pG1 network: minimize L1(sG1(x),pG1(y))
        ###########################
        w_D.volatile = False
        w_D.zero_grad()
        
        wp = G_1(real_photo)
        
        # train with fake
        wp_fake_D_out = sketch_D(wp.detach())
        photo_label.data.resize_(wp_fake_D_out.size()).fill_(fake_label)
        
        err_d_fake = loss_BCE(wp_fake_D_out, photo_label)
        err_d_fake.backward()
        
        # train with real
        w_real_D_out = sketch_D(real_sketch.detach())
        sketch_label.data.resize_(w_real_D_out.size()).fill_(real_label)
        
        err_d_fake = loss_BCE(w_real_D_out, sketch_label)
        err_d_fake.backward()
        
        optimizer_D.step()
        
        ###
        sketch_D.volatile = True
        G_2.volatile = True
        G_1.zero_grad()
        
        wp = G_1(real_photo)

        wp_real_D_out = sketch_D(wp)
        sketch_label.data.resize_(wp_real_D_out.size()).fill_(real_label)

        err_g = loss_BCE(wp_real_D_out, sketch_label) \
                 + opt.lamb * loss_L1(wp, real_sketch.detach())
        err_g.backward()
        
        optimizer_G_1.step()

        print("===> Epoch[{}]({}/{}), DL1:[{:.4f}]".format(epoch, iteration, len(training_data_loader), out1))


def testPSNR():
    avg_psnr = 0
    for batch in testing_data_loader:
        input, target = Variable(batch[0]), Variable(batch[1])
        if opt.cuda:
            input = input.cuda()
            target = target.cuda()

        wp = G_1(input)
        prediction = G_2(wp)

        mse = loss_MSE(prediction, target)
        psnr = 10 * log10(1 / mse.data[0])
        avg_psnr += psnr
    ret = avg_psnr / len(testing_data_loader)
    print("----> Avg. PSNR: {:.4f} dB".format(ret))
    return ret

def checkpoint(epoch):
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    if not os.path.exists(os.path.join("checkpoint", opt.dataset)):
        os.mkdir(os.path.join("checkpoint", opt.dataset))

    net_G1_model_out_path = "checkpoint/{}/G_1_model_epoch_{}.pth".format(opt.dataset, epoch)
    net_G2_model_out_path = "checkpoint/{}/G_2_model_epoch_{}.pth".format(opt.dataset, epoch)

    net_my_layer_model_out_path = "checkpoint/{}/my_layer_model_epoch_{}.pth".format(opt.dataset, epoch)

    torch.save(G_1.state_dict(), net_G1_model_out_path)
    torch.save(G_2.state_dict(), net_G2_model_out_path)
    torch.save(my_layer.state_dict(), net_my_layer_model_out_path)
    print("Checkpoint saved to {}".format("checkpoint" + opt.dataset))

# max_psnr = 0
if __name__ == "__main__":
    for epoch in range(1, opt.nEpochs + 1):
        train(epoch)
        # tmp_psnr = testPSNR()
        # if max_psnr < tmp_psnr:
        #     checkpoint(epoch)
        #     max_psnr = tmp_psnr
        if epoch % 10 == 0:
            checkpoint(epoch)
