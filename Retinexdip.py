from collections import namedtuple
from net import *
from net.downsampler import *
from net.losses import StdLoss, GradientLoss, ExtendedL1Loss, GrayLoss
from net.losses import ExclusionLoss, TVLoss
from net.noise import get_noise
import matplotlib.pyplot as plt
from PIL import Image
from skimage import exposure,color
import numpy as np
import math
import torch
import torchvision
import cv2
from scipy import misc
from torchvision import transforms
from utils.vggloss import VGG16
from utils.sgld import SGLD
import argparse
from glob import glob
import os
import time


parser = argparse.ArgumentParser()
parser.add_argument("--input", "-i", type=str, default='data/Test', help='test image folder')
parser.add_argument("--result", "-r", type=str, default='./result', help='result folder')
arg = parser.parse_args()


EnhancementResult = namedtuple("EnhancementResult", ['reflection', 'illumination'])

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

class Enhancement(object):
    def __init__(self, image_name, image, plot_during_training=True, show_every=10, num_iter=300):
        self.image = image
        self.img = image
        self.size = image.size
        self.image_np = None
        self.images_torch = None
        self.plot_during_training = plot_during_training
        # self.ratio = ratio
        self.psnrs = []
        self.show_every = show_every
        self.image_name = image_name
        self.num_iter = num_iter
        self.loss_function = None
        # self.ratio_net = None
        self.parameters = None
        self.learning_rate = 0.01
        self.input_depth = 3  # This value could affect the performance. 3 is ok for natural image, if your
                            #images are extremely dark, you may consider 8 for the value.
        self.data_type = torch.cuda.FloatTensor
        # self.data_type = torch.FloatTensor
        self.reflection_net_inputs = None
        self.illumination_net_inputs = None
        self.original_illumination = None
        self.original_reflection = None
        self.reflection_net = None
        self.illumination_net = None
        self.total_loss = None
        self.reflection_out = None
        self.illumination_out = None
        self.current_result = None
        self.best_result = None
        self._init_all()

    def _init_all(self):
        self._init_images()
        self._init_decomposition()
        self._init_nets()
        self._init_inputs()
        self._init_parameters()
        self._init_losses()


    def _maxRGB(self):
        '''
        self.image: pil image, input low-light image
        :return: np, initial illumnation
        '''
        (R, G, B) = self.image.split()
        I_0 = np.array(np.maximum(np.maximum(R, G), B))
        return I_0

    def _init_decomposition(self):
        temp = self._maxRGB() # numpy
        # get initial illumination map
        self.original_illumination = np.clip(np.asarray([temp for _ in range(3)]),1,255)/255
        # self.original_illumination = np.clip(temp,1, 255) / 255
        # get initial reflection
        self.original_reflection = self.image_np / self.original_illumination

        self.original_illumination = np_to_torch(self.original_illumination).type(self.data_type)
        self.original_reflection = np_to_torch(np.asarray(self.original_reflection)).type(self.data_type)
        # print(self.original_reflection.shape)namedtuple
        # print(self.original_illumination.shape)




    def _init_images(self):
        #self.images = create_augmentations(self.image)
        # self.images_torch = [np_to_torch(image).type(torch.cuda.FloatTensor) for image in self.images]
        self.image  =transforms.Resize((512,512))(self.image)
        self.image_np = pil_to_np(self.image)  # pil image to numpy
        self.image_torch = np_to_torch(self.image_np).type(self.data_type)
        # print(self.size)

        # print((self.image_torch.shape[2],self.image_torch.shape[3]))

    def _init_inputs(self):
        if self.image_torch is not None:
            size = (self.image_torch.shape[2], self.image_torch.shape[3])
            # print(size)
        input_type = 'noise'
        # input_type = 'meshgrid'
        self.reflection_net_inputs = get_noise(self.input_depth,
                                                  input_type, size).type(self.data_type).detach()
        # misc.imsave('out/input_illumination.png',
        #                         misc.imresize(torch_to_np(self.reflection_net_inputs).transpose(1, 2, 0),(self.size[1],self.size[0])))

        self.illumination_net_inputs = get_noise(self.input_depth,
                                             input_type, size).type(self.data_type).detach()


    def _init_parameters(self):
        self.parameters = [p for p in self.reflection_net.parameters()] + \
                          [p for p in self.illumination_net.parameters()]

    def weight_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0.0, 0.5 * math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif classname.find('Linear') != -1:
            n = m.weight.size(1)
            m.weight.data.normal_(0, 0.01)
            m.bias.data = torch.ones(m.bias.data.size())

    def _init_nets(self):
        pad = 'zero'
        self.reflection_net = skip(self.input_depth, 3,
               num_channels_down = [8, 16, 32, 64,128],
               num_channels_up   = [8, 16, 32, 64,128],
               num_channels_skip = [0, 0, 0, 0, 0],
               filter_size_down = 3, filter_size_up = 3, filter_skip_size=1,
               upsample_mode='bilinear',
               downsample_mode='avg',
               need_sigmoid=True, need_bias=True, pad=pad)
        self.reflection_net.apply(self.weight_init).type(self.data_type)


        self.illumination_net = skip(self.input_depth, 3,
               num_channels_down = [8, 16, 32, 64],
               num_channels_up   = [8, 16, 32, 64],
               num_channels_skip = [0, 0, 0, 0],
               filter_size_down = 3, filter_size_up = 3, filter_skip_size=1,
               upsample_mode='bilinear',
               downsample_mode='avg',
               need_sigmoid=True, need_bias=True, pad=pad)
        self.illumination_net.apply(self.weight_init).type(self.data_type)



    def _init_losses(self):
        self.l1_loss = nn.SmoothL1Loss().type(self.data_type) # for illumination
        self.mse_loss = nn.MSELoss().type(self.data_type)     # for reflection and reconstruction
        self.exclusion_loss =  ExclusionLoss().type(self.data_type)
        self.tv_loss = TVLoss().type(self.data_type)
        self.gradient_loss = GradientLoss().type(self.data_type)



    def optimize(self):
        # torch.backends.cudnn.enabled = True
        # torch.backends.cudnn.benchmark = True
        # optimizer = SGLD(self.parameters, lr=self.learning_rate)
        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        print("Processing: {}".format(self.image_name.split("/")[-1]))
        start = time.time()
        for j in range(self.num_iter):
            optimizer.zero_grad()
            self._optimization_closure(500,499)
            if j==499:
                self._obtain_current_result(499)
            if self.plot_during_training:
                self._plot_closure(j)
            optimizer.step()
        end = time.time()
        print("time:%.4f"%(end-start))
        cv2.imwrite(self.image_name, self.best_result)

    def _get_augmentation(self, iteration):
        if iteration % 2 == 1:
            return 0
        # return 0
        iteration //= 2
        return iteration % 8

    def _optimization_closure(self, num_iter, step):

        reg_noise_std = 1 / 10000.
        aug = self._get_augmentation(step)
        if step == num_iter - 1:
            aug = 0

        illumination_net_input = self.illumination_net_inputs + \
                                 (self.illumination_net_inputs.clone().normal_() * reg_noise_std)
        reflection_net_input = self.reflection_net_inputs + \
                               (self.reflection_net_inputs.clone().normal_() * reg_noise_std)


        self.illumination_out = self.illumination_net(illumination_net_input)
        self.reflection_out = self.reflection_net(reflection_net_input)

        # weighted with the gradient of latent reflectance
        self.total_loss = 0.5*self.tv_loss(self.illumination_out, self.reflection_out)
        self.total_loss += 0.0001*self.tv_loss(self.reflection_out)
        self.total_loss += self.l1_loss(self.illumination_out, self.original_illumination)
        self.total_loss += self.mse_loss(self.illumination_out*self.reflection_out, self.image_torch)
        self.total_loss.backward()


    def _obtain_current_result(self, step):
        """
        puts in self.current result the current result.
        also updates the best result
        :return:
        """
        if step == self.num_iter - 1 or step % 8 == 0:
            reflection_out_np = np.clip(torch_to_np(self.reflection_out),0,1)
            illumination_out_np = np.clip(torch_to_np(self.illumination_out),0,1)
            # psnr = compare_psnr(np.clip(self.image_np,0,1),  reflection_out_np * illumination_out_np)
            # self.psnrs.append(psnr)

            self.current_result = EnhancementResult(reflection=reflection_out_np, illumination=illumination_out_np)
            # if self.best_result is None or self.best_result.psnr < self.current_result.psnr:
            #     self.best_result = self.current_result

    def _plot_closure(self, step):
        print('Iteration {:5d}    Loss {:5f}'.format(step,self.total_loss.item()))
        if step % self.show_every == self.show_every - 1:
            # plot_image_grid("left_right_{}".format(step),
            #                 [self.current_result.reflection, self.current_result.illumination])
            # misc.imsave('out/illumination.png',
            #             misc.imresize(torch_to_np(self.illumination_out).transpose(1, 2, 0),(self.size[1],self.size[0])))

            misc.imsave('output/reflection/reflection-{}.png'.format(step),
                        misc.imresize(torch_to_np(self.reflection_out).transpose(1, 2, 0), (self.size[1],self.size[0])))
            self.get_enhanced(step)

    def gamma_trans(self, img, gamma):
        gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
        gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
        return cv2.LUT(img, gamma_table)

    def adjust_gammma(self,img_gray):
        # mean = np.mean(img_gray)
        # gamma_val = math.log10(0.5) / math.log10(mean / 255)
        # print(gamma_val)
        image_gamma_correct = self.gamma_trans(img_gray, 0.5)
        return image_gamma_correct

    def get_enhanced(self, step, flag=False):
        (R, G, B) = self.img.split()
        ini_illumination = torch_to_np(self.illumination_out).transpose(1, 2, 0)
        ini_illumination = misc.imresize(ini_illumination, (self.size[1], self.size[0]))
        # print(ini_illumination.shape)
        ini_illumination = np.max(ini_illumination, axis=2)
        cv2.imwrite('output/illumination/illumination-{}.png'.format(step), ini_illumination)
        # If the input image is extremely dark, setting the flag as True can produce promising result.
        if flag==True:
            ini_illumination = np.clip(np.max(ini_illumination, axis=2), 0.0000002, 255)
        else:
            ini_illumination = np.clip(self.adjust_gammma(ini_illumination), 0.0000002, 255)
        R = R / ini_illumination
        G = G / ini_illumination
        B = B / ini_illumination
        self.best_result = np.clip(cv2.merge([B, G, R])*255, 0.02, 255).astype(np.uint8)
        cv2.imwrite('output/result-{}.png'.format(step), self.best_result)




def lowlight_enhancer(image_name, image):
    s = Enhancement(image_name, image)
    s.optimize()



if __name__ == "__main__":
    input_root = arg.input
    output_root = arg.result

    datasets = ['DICM', 'ExDark', 'Fusion', 'LIME', 'NPEA', 'Nasa', 'VV']
    # datasets = ['images-for-computing-time']
    for dataset in datasets:
        input_folder = os.path.join(input_root, dataset)
        output_folder = os.path.join(output_root, dataset)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        # print(output_folder)
        path = glob(input_folder + '/*.*')
        path.sort()
        for i in range(len(path)):
            filename = os.path.basename(path[i])
            img_path = os.path.join(input_folder, filename)
            img_path_out = os.path.join(output_folder, filename)
            img = Image.open(img_path).convert('RGB') #LOLdataset/eval15/low/1.png
            lowlight_enhancer(img_path_out, img)

    # input_folder = 'data/images-for-computing-time'
    # output_folder = './result'
    # filename = "BladeImg048_LT.BMP"
    # img_path = os.path.join(input_folder, filename)
    # img_path_out = os.path.join(output_folder, filename)
    # img = Image.open(img_path).convert('RGB')  # LOLdataset/eval15/low/1.png
    # lowlight_enhancer(img_path_out, img)

