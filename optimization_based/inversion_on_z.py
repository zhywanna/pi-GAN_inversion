from __future__ import print_function
import math
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
import numpy as np
import skvideo.io
import curriculums
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def render_freeview_video(opt, render_options, z, generator, video_type):
    trajectory = [] 
    for t in np.linspace(0, 1, 24):
        pitch = 0.2 * t
        yaw = 0
        trajectory.append((pitch, yaw))
    for t in np.linspace(0, 1, 72):
        pitch = 0
        yaw = 0.6 * np.sin(t * 2 * math.pi)
        trajectory.append((pitch, yaw))
    # for t in np.linspace(0, 1, opt.num_frames):
    #     pitch = 0.2 * np.cos(t * 2 * math.pi)
    #     yaw = 0.4 * np.sin(t * 2 * math.pi)
    #     trajectory.append((pitch, yaw))
            
    output_name = f'{video_type}.mp4'
    # yuv444p
    writer = skvideo.io.FFmpegWriter(os.path.join('logs', opt.logfile, output_name), outputdict={'-pix_fmt': 'yuv420p', '-crf': '21'})

    print(f'start rendering {video_type}.mp4')
    frames = []
    with torch.no_grad():
        for pitch, yaw in tqdm(trajectory):
            render_options['h_mean'] = yaw + 3.14/2
            render_options['v_mean'] = pitch + 3.14/2

            frame, _ = generator(z, lock_view_dependence=True, **render_options)
            frames.append(tensor_to_PIL(frame))

    for frame in frames:
        writer.writeFrame(np.array(frame))
    writer.close()

def tensor_to_PIL(img):
    img = img.squeeze() * 0.5 + 0.5
    return Image.fromarray(img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())

def reverse_z(generator, g_z, z, opt):
    """
    Estimate z_approx given G and G(z).
    Args:
        generator: nn.Module, generator network.
        g_z: Variable, G(z).
        z: Variable, the ground truth z, ref only here, not used in recovery.
    Returns:
        z_approx, Variable, the estimated z value.
    """

    # loss metrics
    mse_loss = nn.MSELoss().to(device)
    mse_loss_ = nn.MSELoss().to(device)

    # init tensor
    z_approx = torch.randn((1, 256), device=device)
    

    # convert to variable
    z_approx = Variable(z_approx)
    z_approx.requires_grad = True

    # optimizer
    optimizer_approx = optim.Adam([z_approx], lr=opt.lr,
                                  betas=(opt.beta1, 0.999))

    # tensorboard
    writer = SummaryWriter(os.path.join('logs', opt.logfile))
    writer.add_images('image_z', g_z)
    print('start training')

    # train, use tqdm as a counter
    for i in tqdm(range(opt.niter)):
        g_z_approx, _ = generator(z_approx, **options)
        mse_g_z = mse_loss(g_z_approx, g_z)
        mse_z = mse_loss_(z_approx, z)

        # logs
        writer.add_scalar('image_z_approx_loss', mse_g_z.data, i)
        writer.add_scalar('z_approx_loss', mse_z.data, i)
        if i % 100 == 0:
            # save image
            writer.add_images('image_z_approx', g_z_approx, i)

        # backprop
        optimizer_approx.zero_grad()
        mse_g_z.backward(retain_graph=True)
        optimizer_approx.step()

        # clipping or not
        if opt.clip == 'standard':
            z_approx.data[z_approx.data > 1] = 1
            z_approx.data[z_approx.data < -1] = -1
        if opt.clip == 'stochastic':
            z_approx.data[z_approx.data > 1] = random.uniform(-1, 1)
            z_approx.data[z_approx.data < -1] = random.uniform(-1, 1)
        
        # counter
        # print('training{}/{}'.format(i, opt.niter))

    # save g(z_approx) image
    save_image(g_z_approx.data, os.path.join('logs', opt.logfile, 'g_z_approx.png'), normalize=True)

    return z_approx


def reverse_gan(opt):
    # load generator and fix its weights
    generator = torch.load(opt.generator_path, map_location=torch.device(device))
    ema_file = opt.generator_path.split('generator')[0] + 'ema.pth'
    ema = torch.load(ema_file, map_location=device)
    ema.copy_to(generator.parameters())
    generator.set_device(device)
    generator.eval()
    print('successfully load generator!')

    # init ground truth z
    z = torch.randn((1, 256), device=device)

    # generate g_z in (-1,1)
    g_z, _ = generator(z, **options) 

    save_image(g_z, os.path.join('logs', opt.logfile, 'g_z.png'), normalize=True)

    # recover z_approx
    z_approx = reverse_z(generator, g_z, z, opt)
    # print(z_approx.cpu().data.numpy().squeeze())

    # render free view video
    render_freeview_video(opt, render_options, z_approx, generator, video_type='z_approx')
    render_freeview_video(opt, render_options, z, generator, video_type='z')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generator_path', type=str, default='ckpt/CelebA/generator.pth')
    # parser.add_argument('--image_path', type=str, default='data/gt128.png')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--image_size', type=int, default=64)#128
    parser.add_argument('--num_frames', type=int, default=64)
    parser.add_argument('--max_batch_size', type=int, default=2400000)
    parser.add_argument('--niter', type=int, default=5000)
    parser.add_argument('--logfile', type=str, default='',
                        help='--logfile=free_view/large_image/without_clip')
    parser.add_argument('--clip', default='disabled',
                        help='disabled|standard|stochastic')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='beta1 for adam. default=0.5')
    opt = parser.parse_args()
    print(opt)

    options = {
    'img_size': opt.image_size,
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'num_steps': 22,#24
    'h_stddev': 0,
    'v_stddev': 0,
    'h_mean': torch.tensor(math.pi/2).to(device),
    'v_mean': torch.tensor(math.pi/2).to(device),
    'hierarchical_sample': False,
    'sample_dist': None,
    'clamp_mode': 'relu',
    'nerf_noise': 0,
    }

    render_options = {
    'img_size': 64,
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'num_steps': 22,
    'h_stddev': 0,
    'v_stddev': 0,
    'v_mean': math.pi/2,
    'hierarchical_sample': True,
    'sample_dist': None,
    'clamp_mode': 'relu',
    'nerf_noise': 0,
    'last_back': True,
    }

    if opt.seed is None:
        opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    # if opt.cuda:
    #     torch.cuda.manual_seed_all(opt.manualSeed)
    #     cudnn.benchmark = True  # turn on the cudnn autotuner
    #     # torch.cuda.set_device(1)

    reverse_gan(opt)
    print('----all done!----')