'''
Handles the visualization of the error map between the original and restored images.
'''
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os
import cv2
import torch
import utils

# To generate the RMSE map
def RMSE_per_pixel(y_true, y_pred):
    mse = torch.mean((y_true - y_pred)**2, dim=2)
    rmse = torch.sqrt(mse)
    return rmse

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # just selecting an image for visualization
    # img_path = 'datasets/test/color_r9e70c57ft.png'           # the car img
    img_path = 'datasets/test/color_FujifilmXM1_0124.png'       # the backpack img

    # =========================================================================
    # CLIPPED PP IMAGE
    # =========================================================================
    img_prophoto_gamma = cv2.imread(img_path)
    height, width = img_prophoto_gamma.shape[0], img_prophoto_gamma.shape[1]
    I_PP = img_prophoto_gamma.reshape(-1, 3).T
    I_PP = torch.tensor(I_PP, device=device)        # move to GPU

    # Gamut reduction step
    I_PP, I_ClippedPP, OG_mask = utils.gamut_reduction(I_PP, device)
    display_I_PP = I_PP.T.reshape(height,width,3)
    display_I_ClippedPP = I_ClippedPP.T.reshape(height,width,3)

    rmse_map = RMSE_per_pixel(display_I_PP, display_I_ClippedPP) # compute RMSE
    rmse_map = rmse_map.cpu().numpy()

    plt.imshow(rmse_map, cmap='viridis', vmin=0, vmax=0.05)
    plt.colorbar(label='RMSE')
    plt.title('Per-Pixel RMSE Error Map for Clipped Image')
    plt.show()
    

    # =========================================================================
    # RESTORED PP IMAGE
    # =========================================================================
    restored_folder = 'datasets/restored/'
    restored_img_path = os.path.join(restored_folder, 'restored-backpack.png')
    restored_img = cv2.imread(restored_img_path)
    restored_img = torch.tensor(restored_img, device=device)
    restored_img = restored_img / 255.0
    
    rmse_map = RMSE_per_pixel(display_I_PP, restored_img) # compute RMSE
    rmse_map = rmse_map.cpu().numpy()

    plt.imshow(rmse_map, cmap='viridis', vmin=0, vmax=0.05)
    plt.colorbar(label='RMSE')
    plt.title('Per-Pixel RMSE Error Map for Restored Image')
    plt.show()
    

    