# TODO: reproduce Table 1 for MLP (23KB)+enc

# let's try one image C:

import torch
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('agg')

from matplotlib import pyplot as plt
import numpy as np
import cv2

from model import GamutMLP
import utils
import load

import time
from metrics import RMSE, PSNR


# flag for gpu
device = 'cpu'
if (torch.cuda.is_available):
    device = torch.device('cuda')

# load in one image
img_prophoto_gamma = cv2.imread('datasets/train/vivid_SamsungNX2000_0160.png')
# cv2.imshow('ORIGINAL', img_prophoto_gamma) 
I_PP = img_prophoto_gamma.reshape(-1, 3).T # got this from gamut github code
I_PP, I_ClippedPP, OG_mask = load.gamut_reduction(I_PP)
display_OG_mask = (OG_mask).astype(np.float32) # convert to BW image, recall OG_mask is a boolean array
display_I_PP = I_PP.T.reshape(512,512,3) # ASSUMPTION: we tried to "undo" line 112
display_I_ClippedPP = I_ClippedPP.T.reshape(512,512,3)
display_OG_mask = display_OG_mask.T.reshape(512,512,3) # A white pixel represents OG pixel (value = 1/True), black represents IG (value = 0/False)

print('I_PP_Clipped', I_ClippedPP)
print('shape of I_PP_Clipped', I_ClippedPP.shape)

# training_input: the 120D-feature vector mapped(x, y, R', G', B') values
# ground_truth: the original PP values (x, y, R, G, B)
# I_ClippedPP_5d_coords: the clipped PP values (x, y, R', G', B')
training_input, ground_truth, I_ClippedPP_5d_coords = utils.generatingTrainingInputAndGroundTruth(display_I_PP, display_I_ClippedPP, display_OG_mask)

# build network
net = GamutMLP().to(device)

# L2 (MSE) Loss Function described in Eq. 5
# https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
loss_fn =  torch.nn.MSELoss()

# build optimizer, and set learning rate
# https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
learning_rate = 0.001
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

if __name__ == '__main__':
    # ==================================================================
    # Train loop
    # ==================================================================

    # this stuff here is to track statistics
    rmse_loss = []
    rmse_og_loss = []
    psnr_loss = []
    psnr_og_loss = []
    optim_time = []
    
    start_time = time.time() # optimization start time
    for iter in range(9000): # num of optimization iterations
        net.train() # good practice, some networks have training dependent behaviour
        optimizer.zero_grad()
        x = torch.tensor(training_input).float().to(device)
        pred_residuals = net(x)
        
        # Equation 3, the recovered/predicted wide-gamut PP image
        # convert to torch tensor
        sampled_pixels_restored_values = torch.tensor(I_ClippedPP_5d_coords[:, -3:]).float().to(device) + pred_residuals
       
        # Equation 5, use L2 loss
        # convert to torch tensor
        ground_truth_RGB = torch.tensor(ground_truth[:, -3:]).float().to(device)
        loss = loss_fn(sampled_pixels_restored_values, ground_truth_RGB)

        loss.backward()
        optimizer.step()

        # print out stats every 10 its
        if iter % 1000 == 0:
            print(f'Iteration: {iter} | Loss: {loss.item()}')
    end_time = time.time() # optimization start time

    print('optimization time: ', end_time - start_time)

    # Now, let's see how well GamutMLP restores the clipped PP image
    encoded_5d_coords, I_Clipped_PP_5d_coords = utils.gamut_expansion(display_I_ClippedPP)
    with torch.no_grad():
        net.eval()
        x = torch.tensor(encoded_5d_coords).float().to(device)
        pred_residuals = net(x).to('cpu')

    # Equation 3, the recovered/predicted wide-gamut PP image
    # convert to torch tensor
    all_pixels_restored_values = torch.tensor(I_Clipped_PP_5d_coords[:, -3:]).float() + pred_residuals

    # Restore the image
    restored_I_PP = torch.zeros((512, 512, 3))
    for x in range(512):
        for y in range(512):
            idx = x * 512 + y
            restored_I_PP[x, y, 0] = all_pixels_restored_values[idx, 0]
            restored_I_PP[x, y, 1] = all_pixels_restored_values[idx, 1]
            restored_I_PP[x, y, 2] = all_pixels_restored_values[idx, 2]
    
    restored_I_PP = restored_I_PP.detach().cpu().numpy()

    # calculate the metrics
    print('I_PP shape: ', I_PP.shape)
    print('OG mask shape: ', display_OG_mask.shape)
    print('Display I_PP shape:', display_I_PP.shape)
    print('Restored I_PP shape: ', restored_I_PP.shape)
    
    # calculate metrics for OG
    # convert OG (np array to tensor)
    tensor_display_OG_mask = torch.from_numpy(display_OG_mask)
    tensor_restored_I_PP = torch.from_numpy(restored_I_PP)
    tensor_display_I_PP = torch.from_numpy(display_I_PP)

    # masked_restored_I_PP = tensor_display_OG_mask * tensor_restored_I_PP
    # masked_I_PP = tensor_display_OG_mask * tensor_display_I_PP

    # tensor([og_mask]==1)
    OG_pixels_restored_I_PP = tensor_restored_I_PP[tensor_display_OG_mask == 1]
    OG_pixels_display_I_PP = tensor_display_I_PP[tensor_display_OG_mask == 1]
    print('NUM 1 in OG mask: ', torch.sum(tensor_display_OG_mask).item())
    print('OG_pixels_restored_I_PP', OG_pixels_restored_I_PP.shape)
    print('OG_pixels_display_I_PP', OG_pixels_display_I_PP.shape)

    rmse = RMSE(tensor_display_I_PP, tensor_restored_I_PP)
    print('RMSE: ', rmse)
    psnr = PSNR(tensor_display_I_PP, tensor_restored_I_PP)
    print('PSNR: ', psnr)

    rmse_og = RMSE(OG_pixels_display_I_PP, OG_pixels_restored_I_PP)
    print('RMSE for OG: ', rmse_og)
    psnr_og = PSNR(OG_pixels_display_I_PP, OG_pixels_restored_I_PP)
    print('PSNR for OG: ', psnr_og)

    # cv2.imshow('Displaying Original I_PP image', display_I_PP)
    # cv2.imshow('Displaying Clipped I_PP image', display_I_ClippedPP)
    # cv2.imshow('Displaying Restored I_ClippedPP image', restored_I_PP) 
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()