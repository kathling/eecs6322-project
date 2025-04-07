'''
Handles restoring the clipped PP image and evaluating the GamutMLP model on RMSE and PSNR.
'''
import torch
from metrics import RMSE, PSNR

def restore_img(GamutMLP, encoded_pixels, I_Clipped_PP_5d_coords, 
         height, width, device):
    '''
    Handles restoring the clipped PP image using the GamutMLP model.
    Input:
        GamutMLP: GamutMLP model
        encoded_pixels: 120D coordinates of the clipped PP image to restore
        I_Clipped_PP_5d_coords: 5D coordinates of the clipped PP image
        height: height of the image
        width: width of the image
        device: 'cuda' (GPU), 'cpu' (CPU)
    '''
    GamutMLP.eval()         # set to eval mode to get final results
    with torch.no_grad():
        pred_residuals = GamutMLP(encoded_pixels)
    
    restored_pixels = I_Clipped_PP_5d_coords[:, -3:] + pred_residuals
    restored_img = restored_pixels.view(height, width, 3)
    # restored_img = restored_img.detach().cpu().numpy()

    return restored_img

def calc_metrics(restored_I_PP,  display_I_PP, display_OG_mask):
    '''
    Handles calculating the RMSE and PSNR metrics for the restored image.
    Input:
        restored_I_PP: restored image (5D coordinates of the PP image)
        display_I_PP: original image (5D coordinates of the PP image)
        display_OG_mask: mask for the original image (boolean array)
    Output:
        rmse: RMSE for all pixels in restored image
        psnr: PSNR for all pixels in restored image
        rmse_og: RMSE for the out-of-gamut pixels
        psnr_og: PSNR for the out-of-gamut pixels
    '''
    # calculate metrics for entire image
    rmse = RMSE(display_I_PP, restored_I_PP)
    psnr = PSNR(display_I_PP, restored_I_PP)

    # calculate metrics for OG pixels
    OG_pixels_restored_I_PP = restored_I_PP[display_OG_mask == 1]
    OG_pixels_display_I_PP = display_I_PP[display_OG_mask == 1]
    rmse_og = RMSE(OG_pixels_display_I_PP, OG_pixels_restored_I_PP)
    psnr_og = PSNR(OG_pixels_display_I_PP, OG_pixels_restored_I_PP)

    return rmse, psnr, rmse_og, psnr_og