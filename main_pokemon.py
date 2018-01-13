from __future__ import print_function
import os
import matplotlib as mpl
import tarfile
import matplotlib.image as mpimg
from matplotlib import pyplot as plt

import mxnet as mx
from mxnet import gluon
from mxnet import ndarray as nd
from mxnet.gluon import nn, utils
from mxnet import autograd
import numpy as np

import msvcrt
import shutil

epochs = 30 # Set low by default for tests, set higher when you actually run this code.
batch_size = 10
latent_z_size = 100
filename_G = "netG.params"
filename_D = "netD.params"
param_datapath = "checkpoints_pokemon"

use_gpu = True
ctx = mx.gpu() if use_gpu else mx.cpu()

lr = 0.0002
beta1 = 0.5

data_path = 'pokemon_dataset'

if not os.path.exists(data_path):
    print("Dataset not found! (folder: {})".format(data_path))
    exit()

# Reshape dataset
target_wd = 64
target_ht = 64
img_list = []

def transform(data, target_wd, target_ht):
    # resize to target_wd * target_ht
    data = data[:,:,:3]
    data = mx.image.imresize(data, target_wd, target_ht)
    # transpose from (target_wd, target_ht, 3)
    # to (3, target_wd, target_ht)
    data = nd.transpose(data, (2,0,1))
    # normalize to [-1, 1]
    data = data.astype(np.float32)/127.5 - 1
    # if image is greyscale, repeat 3 times to get RGB image.
    if data.shape[0] == 1:
        data = nd.tile(data, (3, 1, 1))
    return data.reshape((1,) + data.shape)

# Loading dataset
for dirpaths, dirnames, fnames in os.walk(data_path):
    for fname in fnames:
        if not fname.endswith('.png'):
            continue
        img = os.path.join(data_path, fname)
        img_arr = mx.image.imread(img)
        img_arr = transform(img_arr, target_wd, target_ht)
        img_list.append(img_arr)
train_data = mx.io.NDArrayIter(data=nd.concatenate(img_list), batch_size=batch_size)

def visualize(img_arr):
    plt.imshow(((img_arr.asnumpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
    plt.axis('off')

for i in range(4):
    plt.subplot(1,4,i+1)
    visualize(img_list[i + 10][0])
plt.show()

# build the generator
nc = 3
generator_frames = 64
netG = nn.Sequential()
with netG.name_scope():
    # input is Z, going into a deconvolution
    # state size. (generator_frames*n) x h x w
    # n get smaller to compensate increasing input size on next layer
    # size -> 4 x 4
    netG.add(nn.Conv2DTranspose(generator_frames * 8, 4, 1, 0, use_bias=False))
    netG.add(nn.BatchNorm())
    netG.add(nn.Activation('relu'))
    # size -> 8 x 8
    netG.add(nn.Conv2DTranspose(generator_frames * 4, 4, 2, 1, use_bias=False))
    netG.add(nn.BatchNorm())
    netG.add(nn.Activation('relu'))
    # size -> 16 x 16
    netG.add(nn.Conv2DTranspose(generator_frames * 2, 4, 2, 1, use_bias=False))
    netG.add(nn.BatchNorm())
    netG.add(nn.Activation('relu'))
    # size -> 32 x 32
    netG.add(nn.Conv2DTranspose(generator_frames, 4, 2, 1, use_bias=False))
    netG.add(nn.BatchNorm())
    netG.add(nn.Activation('relu'))
    # state size. (nc -> colors) x 64 x 64
    netG.add(nn.Conv2DTranspose(nc, 4, 2, 1, use_bias=False))
    netG.add(nn.Activation('tanh'))

# build the discriminator
discriminator_frames = 64
netD = nn.Sequential()
with netD.name_scope():
    # input is (nc) x 64 x 64
    # state size. (generator_frames*n) x h x w
    # n get smaller to compensate increasing input size on next layer
    netD.add(nn.Conv2D(discriminator_frames, 4, 2, 1, use_bias=False))
    netD.add(nn.LeakyReLU(0.2))
    # size -> 32 x 32
    netD.add(nn.Conv2D(discriminator_frames * 2, 4, 2, 1, use_bias=False))
    netD.add(nn.BatchNorm())
    netD.add(nn.LeakyReLU(0.2))
    # size -> 16 x 16
    netD.add(nn.Conv2D(discriminator_frames * 4, 4, 2, 1, use_bias=False))
    netD.add(nn.BatchNorm())
    netD.add(nn.LeakyReLU(0.2))
    # size -> 8 x 8
    netD.add(nn.Conv2D(discriminator_frames * 8, 4, 2, 1, use_bias=False))
    netD.add(nn.BatchNorm())
    netD.add(nn.LeakyReLU(0.2))
    # size -> 4 x 4
    netD.add(nn.Conv2D(1, 4, 1, 0, use_bias=False))

# loss
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()

# initialize the generator and the discriminator
netG.initialize(mx.init.Normal(0.02), ctx=ctx)
netD.initialize(mx.init.Normal(0.02), ctx=ctx)

# trainer for the generator and the discriminator
trainerG = gluon.Trainer(netG.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1})
trainerD = gluon.Trainer(netD.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1})


real_label = nd.ones((batch_size,), ctx=ctx)
fake_label = nd.zeros((batch_size,),ctx=ctx)

def facc(label, pred):
    pred = pred.ravel()
    label = label.ravel()
    return ((pred > 0.5) == label).mean()
metric = mx.metric.CustomMetric(facc)

epoch_iter = 0

# Load latest existing checkpoint
if os.path.exists(param_datapath):
    f = open(param_datapath+"/"+"epoch_iter.txt", mode='r')
    content = f.read()
    epoch_iter = int(content)
    f.close()
    D_path = param_datapath+"/"+content+"/"+filename_D
    G_path = param_datapath+"/"+content+"/"+filename_G
    netD.load_params(D_path, ctx = ctx)
    netG.load_params(G_path, ctx = ctx)

# for epoch in range(epochs):
while True:
    train_data.reset()
    iter = 0
    epoch_iter += 1
    for batch in train_data:
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        data = batch.data[0].as_in_context(ctx)
        latent_z = mx.nd.random_normal(0, 1, shape=(batch_size, latent_z_size, 1, 1), ctx=ctx)
        with autograd.record():
            # train with real image
            output = netD(data).reshape((-1, 1))
            errD_real = loss(output, real_label)
            metric.update([real_label,], [output,])

            # train with fake image
            fake = netG(latent_z)
            output = netD(fake.detach()).reshape((-1, 1))
            errD_fake = loss(output, fake_label)
            errD = errD_real + errD_fake
            errD.backward()
            metric.update([fake_label,], [output,])

        trainerD.step(batch.data[0].shape[0])

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        with autograd.record():
            fake = netG(latent_z)
            output = netD(fake).reshape((-1, 1))
            errG = loss(output, real_label)
            errG.backward()

        trainerG.step(batch.data[0].shape[0])

        # Print log infomation every ten batches
        if iter % 50 == 0:
            name, acc = metric.get()
            print("Iter: {} || Epoch: {}".format(iter, epoch_iter))
            print("Discriminator Loss = {}\nGenerator Loss = {}\nBinary Training Accuracy = {}".format(nd.mean(errD).asscalar(), nd.mean(errG).asscalar(), acc))
        iter = iter + 1

        # Visualize one generated image on keypress
        if msvcrt.kbhit():
            key = msvcrt.getch()
            # Press s to save current parameters
            if(key == b's'):
                print("Saving Current Params...")
                if not os.path.exists(param_datapath):
                    os.makedirs(param_datapath)
                os.makedirs(param_datapath+"/"+str(epoch_iter))
                netD.save_params(param_datapath+"/"+str(epoch_iter)+"/"+filename_D)
                netG.save_params(param_datapath+"/"+str(epoch_iter)+"/"+filename_G)
                f = open(param_datapath+"/"+"epoch_iter.txt", mode='w+')
                f.write(str(epoch_iter))
                f.close()
            latent_z = mx.nd.random_normal(0, 1, shape=(2, latent_z_size, 1, 1), ctx=ctx)
            fake_img = netG(latent_z)
            visualize(fake_img[1])
            plt.show()

    name, acc = metric.get()
    metric.reset()

    # Save checkpoint every 100 epochs
    if epoch_iter % 100 == 0:
        # f = open(filename_D, mode='w+')
        # f.close()
        # f = open(filename_G, mode='w+')
        # f.close()
        if not os.path.exists(param_datapath):
            os.makedirs(param_datapath)
        # Delete older checkpoints to conserve space
        if os.path.exists(param_datapath+"/"+str(epoch_iter - 1000)):
            shutil.rmtree(param_datapath+"/"+str(epoch_iter - 1000))
        os.makedirs(param_datapath+"/"+str(epoch_iter))
        netD.save_params(param_datapath+"/"+str(epoch_iter)+"/"+filename_D)
        netG.save_params(param_datapath+"/"+str(epoch_iter)+"/"+filename_G)
        f = open(param_datapath+"/"+"epoch_iter.txt", mode='w+')
        f.write(str(epoch_iter))
        f.close()
