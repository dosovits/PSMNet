from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import skimage
import skimage.io
import skimage.transform
import numpy as np
import time
import math
from utils import preprocess
from models import *

import shutil
import imageio

# 2012 data /media/jiaren/ImageNet/data_scene_flow_2012/testing/

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--loadmodel', default=None,
                    help='loading model')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

model = nn.DataParallel(model, device_ids=[0])
model.cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

def test(imgL,imgR):
        model.eval()

        if args.cuda:
           imgL = torch.FloatTensor(imgL).cuda()
           imgR = torch.FloatTensor(imgR).cuda()

        imgL, imgR= Variable(imgL), Variable(imgR)

        with torch.no_grad():
            output = model(imgL,imgR)
        output = torch.squeeze(output)
        pred_disp = output.data.cpu().numpy()

        return pred_disp

def save_image(filename, img_np):
    print(filename, img_np.shape)
    # img_np = img_th.cpu().detach().numpy()
    img_np = np.clip(img_np * 255., 0., 255.).astype(np.uint8)
    if img_np.shape[0] == 1 or img_np.shape[0] == 3:
        imageio.imwrite(filename, np.transpose(img_np, (1,2,0)))
    elif len(img_np.shape) == 3 and (img_np.shape[2] == 1 or img_np.shape[2] == 3):
        imageio.imwrite(filename, img_np)
    else:
        raise Exception("Cannot write image of size", img_np.shape)

def get_images(dataset, scene_id=0, sample_id=0):
    if dataset == "mp3d":
        file_template = "/home/adosovit/work/toolboxes/2019/Revisiting_Single_Depth_Estimation/data/mp3d/stereo/train/{:05}/{:03}_{}"
    elif dataset == "suncg_f":
        file_template = "/home/adosovit/work/toolboxes/2019/Revisiting_Single_Depth_Estimation/data/suncg/stereo/furnished/train1/{:05}/{:03}_{}"
    elif dataset == "suncg_e":
        file_template = "/home/adosovit/work/toolboxes/2019/Revisiting_Single_Depth_Estimation/data/suncg/stereo/empty/train1/{:05}/{:03}_{}"
    else:
        raise Exception("Unknown dataset", dataset)

    left_file = file_template.format(scene_id, sample_id, "color_l.jpg")
    right_file = file_template.format(scene_id, sample_id, "color_r.jpg")
    depth_file = file_template.format(scene_id, sample_id, "depth.png")

    return left_file, right_file, depth_file

def main():
   processed = preprocess.get_transform(augment=False)

   out_template = "results/{}_{:05}_{:03}_{{}}"
   dataset = "mp3d" # "mp3d" "suncg_f" "suncg_e"

   for scene_id in range(1):
       print("=Scene {}".format(scene_id))
       for sample_id in range(1):
           print("   Sample {}".format(sample_id))
           left_file, right_file, depth_file = get_images(dataset, scene_id, sample_id)

           imgL_o = (skimage.io.imread(left_file).astype('float32'))
           imgR_o = (skimage.io.imread(right_file).astype('float32'))
           # print(imgL_o[::200,::200,:])
           imgL_r = skimage.transform.resize(imgL_o, (240,320), anti_aliasing=True)
           imgR_r = skimage.transform.resize(imgR_o, (240,320), anti_aliasing=True)
           imgL = processed(imgL_r).numpy()[None,:,:,:]
           imgR = processed(imgR_r).numpy()[None,:,:,:]
           # print(imgL.shape)
           # imgL = np.reshape(imgL,[1,3,imgL.shape[1],imgL.shape[2]])
           # imgR = np.reshape(imgR,[1,3,imgR.shape[1],imgR.shape[2]])
           out_template_curr =  out_template.format(dataset, scene_id, sample_id)
           shutil.copyfile(depth_file, out_template_curr.format("depth_gt.png"))

           # pad to (384, 1248)
           top_pad = 384-imgL.shape[2]
           left_pad = 1248-imgL.shape[3]
           imgL = np.lib.pad(imgL,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
           imgR = np.lib.pad(imgR,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
           save_image(out_template_curr.format("right_padded.png"), imgR[0]/255.)

           start_time = time.time()
           pred_disp = test(imgL,imgR)
           print('time = %.2f' %(time.time() - start_time))

           top_pad   = 384-imgL_r.shape[0]
           left_pad  = 1248-imgL_r.shape[1]
           pred_disp = pred_disp[top_pad:,:-left_pad,None]
           depth_pred = 5. / np.clip(pred_disp, 5., 100.)
           # skimage.io.imsave(test_left_img[inx].split('/')[-1],(img*256).astype('uint16'))
           print(pred_disp[::50,::50,:])
           save_image(out_template_curr.format("right.png"), imgR_o/255.)
           save_image(out_template_curr.format("left.png"), imgL_o/255.)
           save_image(out_template_curr.format("disparity.png"), pred_disp / 255.)
           save_image(out_template_curr.format("depth_pred.png"), depth_pred)

if __name__ == '__main__':
   main()
