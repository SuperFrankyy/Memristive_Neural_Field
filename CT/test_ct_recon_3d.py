import os
import argparse
import shutil

import torch
import torch.nn as nn
import torchvision
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
import tensorboardX

import numpy as np

from networks import Positional_Encoder, FFN, SIREN
from utils import get_config, prepare_sub_folder, get_data_loader, save_image_3d
from ct_geometry_projector import ConeBeam3DProjector
# from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity as compare_ssim
from backend.haq import haq_simu


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument('--pretrain', action='store_true', help="load pretrained model weights")
parser.add_argument('--iter', type=int, help="load model weights from iter")

# Load experiment setting
opts = parser.parse_args()
config = get_config(opts.config)
max_iter = config['max_iter']

cudnn.benchmark = True

# Setup output folder
output_folder = os.path.splitext(os.path.basename(opts.config))[0]
if opts.pretrain: 
    output_subfolder = config['data'] + '_pretrain'
else:
    output_subfolder = config['data']
model_name = os.path.join(output_folder, output_subfolder + '/img{}_proj{}_{}_{}_{}_{}_{}_lr{:.2g}_encoder_{}' \
    .format(config['img_size'], config['num_proj'], config['model'], \
        config['net']['network_input_size'], config['net']['network_width'], \
        config['net']['network_depth'], config['loss'], config['lr'], config['encoder']['embedding']))
if not(config['encoder']['embedding'] == 'none'):
    model_name += '_scale{}_size{}'.format(config['encoder']['scale'], config['encoder']['embedding_size'])
print(model_name)

output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)

# Setup input encoder:
encoder = Positional_Encoder(config['encoder'])

# Setup model
if config['model'] == 'SIREN':
    model = SIREN(config['net'])
elif config['model'] == 'FFN':
    model = FFN(config['net'])
else:
    raise NotImplementedError
model.cuda()
model.eval()

# Load pretrain model
model_path = os.path.join(checkpoint_directory, "model_{:06d}.pt".format(opts.iter))

state_dict = torch.load(model_path)
model.load_state_dict(state_dict['net'])
encoder.B = state_dict['enc']
print('Load pretrain model: {}'.format(model_path))


#  遍历模型的所有线性层，对其权重进行haq量化
with torch.no_grad():
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            weight = module.weight.data
            # 这里n、ratio、std等参数可以根据实际需求调整
            decimal_weight, weight_diff, relative_error, resistance = haq_simu(weight, n=12, ratio=1.5, std=0.15, coarse=True)
            module.weight.data.copy_(decimal_weight)
            # print(f"已对{name}的权重进行haq量化，均方误差: {weight_diff:.4e}, 相对误差: {relative_error:.4e}")

# Setup loss function
if config['loss'] == 'L2':
    loss_fn = torch.nn.MSELoss()
elif config['loss'] == 'L1':
    loss_fn = torch.nn.L1Loss()
else:
    NotImplementedError

# Setup data loader
print('Load image: {}'.format(config['img_path']))
data_loader = get_data_loader(config['data'], config['img_path'], config['img_size'], img_slice=None, train=True, batch_size=config['batch_size'])

config['img_size'] = (config['img_size'], config['img_size'], config['img_size']) if type(config['img_size']) == int else tuple(config['img_size'])
slice_idx = list(range(0, config['img_size'][0], int(config['img_size'][0]/config['display_image_num'])))
if config['num_proj'] > config['display_image_num']:
    proj_idx = list(range(0, config['num_proj'], int(config['num_proj']/config['display_image_num'])))
else:
    proj_idx = list(range(0, config['num_proj']))
    

for it, (grid, image) in enumerate(data_loader):
    # Input coordinates (x,y) grid and target image
    grid = grid.cuda()  # [bs, z, x, y, 3], [0, 1]
    image = image.cuda()  # [bs, z, x, y, 1], [0, 1]
    print(grid.shape, image.shape)

    # Data loading
    test_data = (grid, image)  # [bs, z, x, y, 1]

    # Compute testing psnr
    with torch.no_grad():
        test_embedding = encoder.embedding(test_data[0])
        test_output = model(test_embedding)

        # ############## include read noise data sampled from device data ##############
        # # test_embedding的维度是[1, 6, 128, 128, 131]
        # # 建立一个三层循环，第一层循环是6，第二层循环是128，第三层循环是128，过model
        # test_output = torch.zeros(1, 40, 128, 128, 1)
        # iter_count = 0
        # for i in range(40):
        #     for j in range(128):
        #         for k in range(128):
        #             print(iter_count)
        #             # encoding
        #             test_embedding = encoder.embedding(test_data[0], iter_count)

        #             model_dict = model.state_dict()
        #             # 对weight_prepared[0]加高斯噪声，std为0.01
        #             # std = 0.01
        #             # print(torch.mean(torch.abs(weight1_prepared[0])))
        #             # weight1_new = weight1_prepared[0] + torch.randn(weight1_prepared[0].shape) * weight1_prepared[0] * std
        #             # weight2_new = weight2_prepared[0] + torch.randn(weight2_prepared[0].shape) * weight2_prepared[0] * std
        #             # weight3_new = weight3_prepared[0] + torch.randn(weight3_prepared[0].shape) * weight3_prepared[0] * std
        #             # weight4_new = weight4_prepared[0] + torch.randn(weight4_prepared[0].shape) * weight4_prepared[0] * std
                    
        #             model_dict['model.0.linear.weight'] = weight1_prepared[iter_count]
        #             model_dict['model.1.linear.weight'] = weight2_prepared[iter_count]
        #             model_dict['model.2.linear.weight'] = weight3_prepared[iter_count]
        #             model_dict['model.3.linear.weight'] = weight4_prepared[iter_count]
        #             model.load_state_dict(model_dict)
                    
        #             input_temp = test_embedding[:, i, j, k, :]
        #             test_output_temp = model(input_temp)
        #             test_output[:, i, j, k, :] = test_output_temp
        #             iter_count += 1
        #             if iter_count == weight1_prepared.shape[0]:
        #                 iter_count = 0
        # test_output = test_output.cuda()
        # #########################################################

        test_loss = 0.5 * loss_fn(test_output, test_data[1])
        test_psnr = - 10 * torch.log10(2 * test_loss).item()
        test_loss = test_loss.item()

        test_ssim = compare_ssim(test_output.transpose(1,4).squeeze().cpu().numpy(), test_data[1].transpose(1,4).squeeze().cpu().numpy(), multichannel=True)  # [x, y, z] # treat the last dimension of the array as channels

    save_image_3d(test_output, slice_idx, os.path.join(image_directory, "recon_{}_{:.4g}dB_ssim{:.4g}.png".format(opts.iter, test_psnr, test_ssim)))
    print("[Testing Iteration: {}] Test loss: {:.4g} | Test psnr: {:.4g} | Test ssim: {:.4g}".format(opts.iter, test_loss, test_psnr, test_ssim))
    




