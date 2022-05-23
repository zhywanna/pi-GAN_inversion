from __future__ import print_function
from cv2 import normalize
import torch, itertools, os, shutil, PIL, argparse, numpy
from torch.nn.functional import mse_loss
from torchvision.utils import save_image
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import math
import random
import torch.nn.parallel
import torch.utils.data
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
import numpy as np
import skvideo.io
import curriculums
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, help='Learning rate', default=0.01)
parser.add_argument('--model', type=str, help='', default='train_single_image_encoder')
parser.add_argument('--expname', type=str, help='', default='debug')
parser.add_argument('--out_dir', type=str, help='', default='result')
parser.add_argument('--image_size', type=int, default=64)#128
parser.add_argument('--generator_path', type=str, default='ckpt/CelebA/generator.pth')
parser.add_argument('--num_epochs', type=int, default=10000)
parser.add_argument('--batchsize', type=int, default=1, help='training batchsize for generate random z')
parser.add_argument('--logfile', type=str, default='log', help='to save tensorboard log file')
opt = parser.parse_args()


expdir = os.path.join(opt.out_dir, opt.model, opt.expname)
os.makedirs(expdir, exist_ok=True)
os.makedirs(os.path.join(expdir, 'test'), exist_ok=True)
os.makedirs(os.path.join(expdir, 'train'), exist_ok=True)
os.makedirs(opt.logfile, exist_ok=True)

####### Some module used in Encoder #######
class AdapterBlock(nn.Module):
    def __init__(self, output_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, output_channels, 1, padding=0),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, input):
        return self.model(input)
class AddCoords(nn.Module):
    """
    Source: https://github.com/mkocabas/CoordConv-pytorch/blob/master/CoordConv.py
    """

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret
def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1 or classname.find('Conv2d') != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
class ResidualCCBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3):
        super().__init__()
        p = kernel_size//2
        self.network = nn.Sequential(
            CoordConv(inplanes, planes, kernel_size=kernel_size, padding=p),
            nn.LeakyReLU(0.2, inplace=True),
            CoordConv(planes, planes, kernel_size=kernel_size, stride=2, padding=p),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.network.apply(kaiming_leaky_init)

        self.proj = nn.Conv2d(inplanes, planes, 1, stride=2)

    def forward(self, input):
        y = self.network(input)

        identity = self.proj(input)

        y = (y + identity)/math.sqrt(2)
        return y
class CoordConv(nn.Module):
    """
    Source: https://github.com/mkocabas/CoordConv-pytorch/blob/master/CoordConv.py
    """
    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels+2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret
###########################################

# Encoder in this method. Same as discriminator in pi-GAN 's CelebA
class Encoder(nn.Module):
    def __init__(self, **kwargs): # from 4 * 2^0 to 4 * 2^7 4 -> 512
        super().__init__()
        self.epoch = 0
        self.step = 0
        self.layers = nn.ModuleList(
        [
            ResidualCCBlock(32, 64), # 6 256x256 -> 128x128
            ResidualCCBlock(64, 128), # 5 128x128 -> 64x64
            ResidualCCBlock(128, 256), # 4 64x64 -> 32x32
            ResidualCCBlock(256, 400), # 3 32x32 -> 16x16
            ResidualCCBlock(400, 400), # 2 16x16 -> 8x8
            ResidualCCBlock(400, 400), # 1 8x8 -> 4x4
            ResidualCCBlock(400, 400), # 7 4x4 -> 2x2
        ])
        self.fromRGB = nn.ModuleList(
        [
            AdapterBlock(32),
            AdapterBlock(64),
            AdapterBlock(128),
            AdapterBlock(256),
            AdapterBlock(400),
            AdapterBlock(400),
            AdapterBlock(400),
            AdapterBlock(400)
        ])
        self.final_layer = nn.Conv2d(400, 1 + 256 + 2, 2)  # 1 + 256 + 2 : prediction(1) + latent_code(256) + position(2)
        self.img_size_to_layer = {2:7, 4:6, 8:5, 16:4, 32:3, 64:2, 128:1, 256:0}

    def forward(self, input, alpha):
        start = self.img_size_to_layer[input.shape[-1]]
        x = self.fromRGB[start](input)
        
        for i, layer in enumerate(self.layers[start:]):
            if i == 1 and alpha < 1:
                x = alpha * x + (1 - alpha) * self.fromRGB[start+1](F.interpolate(input, scale_factor=0.5, mode='nearest'))

            x = layer(x)

        x = self.final_layer(x).reshape(x.shape[0], -1)
    
        prediction = x[..., 0:1]
        latent = x[..., 1:257]
        position = x[..., 257:259]

        # return prediction, latent, position
        return latent


def main():
    print('Training %s' % expdir)

    # tensorboard
    writer = SummaryWriter(opt.logfile)

    # load generator and fix its weights
    generator = torch.load(opt.generator_path, map_location=torch.device(device))
    ema_file = opt.generator_path.split('generator')[0] + 'ema.pth'
    ema = torch.load(ema_file, map_location=device)
    ema.copy_to(generator.parameters())
    generator.set_device(device)
    generator.eval()
    print('successfully load generator!')
    # generator = torch.nn.DataParallel(generator)

    # Make an encoder model.
    encoder = Encoder().to(device)

    # training batchsize for generate random z
    batch_size = opt.batchsize

    # Set up optimizer
    learning_rate = opt.lr
    optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)

    num_epochs = opt.num_epochs
    for epoch in tqdm(range(num_epochs)):
        # Batch training loop
        z_batch = torch.randn((batch_size, 256), device=device)
        img_batch, _ = generator(z_batch, **options)

        # An iterative parameter in Encoder
        alpha = min(1, encoder.step / 10000)

        z_batch_approx = encoder(img_batch, alpha)
        optimizer.zero_grad()
        loss = encoder_generator_loss(z_batch_approx, generator, img_batch)
        loss.backward()
        optimizer.step()

        # tensorboard
        writer.add_scalar('Encoder_batch_training_img_loss', loss, epoch)

        if epoch % 100 == 0:
            img_batch_approx, _ = generator(z_batch_approx, **options)
            imgs_cat = torch.cat((img_batch, img_batch_approx))
            save_image(imgs_cat, os.path.join(expdir, 'train', f'{epoch}.png'), normalize=True)

        # Single-image testing loop
        with torch.no_grad():
            z = torch.randn((1, 256), device=device)
            img, _ = generator(z, **options)
            z_approx = encoder(img, alpha)
            img_approx, _ = generator(z_approx, **options)
            writer.add_scalar('Encoder_single_image_testing_loss', encoder_generator_loss(z_approx, generator, img), epoch)
            if epoch % 100 == 0:
                save_image([img.squeeze(),img_approx.squeeze()], os.path.join(expdir, 'test', f'{epoch}.png'), normalize=True)
            

        encoder.step += 1

    torch.save(encoder, os.path.join('ckpt','CelebA', 'my_E.pth'))
    # save_checkpoint(epoch=epoch,
    #                 state_dict=encoder.state_dict(),
    #                 loss=loss.item(),
    #                 lr=learning_rate,
    #                 optimizer=optimizer.state_dict())


def save_checkpoint(**kwargs):
    dirname = os.path.join(expdir, 'snapshots')
    os.makedirs(dirname, exist_ok=True)
    filename = 'epoch_%d.pth.tar' % kwargs['epoch']
    torch.save(kwargs, os.path.join(dirname, filename))
    
def encoder_generator_loss(z_batch_approx, generator, img_batch):
    img_batch_approx, _ = generator(z_batch_approx, **options)
    return mse_loss(img_batch_approx, img_batch)


if __name__ == '__main__':
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

    main()
    print('------all done !-----------')

