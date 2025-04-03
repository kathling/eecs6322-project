'''
-------------------------------------------------------------------------------
Testing out how to load and view the dataset
-------------------------------------------------------------------------------
Dataset retrieved from: https://github.com/hminle/gamut-mlp#dataset
Train Data: 2000 images
- we could use the zipped or unzipped folder
- the code below uses the unzipped folder
Test Data: 200 images
- split into 8 different zip files (I've just downloaded one of them for now)
'''

# import cv2
import numpy as np
# import colour
# import matplotlib.pyplot as plt
# import utils
import torch

'''
Clipped ProPhoto image is obtained through gamut reduction (absolute colorimetric intent)
From equation 1: sRGB = g(clip(MI, min=0, max=1))
    - M: matrix that maps ProPhoto to the unclipped sRGB (provided in Appendix A)
    - I: original ProPhoto image
    - clip: clipping operation
    - g: gamma encoding 
'''

def gamma_encoding_for_sRGB(linear_img):
    # https://en.wikipedia.org/wiki/SRGB#Transfer_function_(%22gamma%22) 
    return torch.where(linear_img <= 0.0031308, 12.92 * linear_img, 1.055 * torch.pow(linear_img, 1/2.4) - 0.055)

    # https://colour.readthedocs.io/en/latest/generated/colour.cctf_encoding.html
    # return colour.cctf_encoding(linear_img, function='sRGB')

def gamma_decoding_for_sRGB(gamma_img):
    # https://en.wikipedia.org/wiki/SRGB#Transfer_function_(%22gamma%22) 
    return torch.where(gamma_img <= 0.04045, gamma_img / 12.92, ((gamma_img + 0.055) / 1.055) ** 2.4)

    # https://colour.readthedocs.io/en/latest/generated/colour.cctf_encoding.html
    # return colour.cctf_decoding(gamma_img, function='sRGB')

def gamut_reduction(I_PP, device):
    # define M as described in the paper
    I_PP = I_PP / 255.0  # ASSUMPTION: must normalize to floating pt format before computations (because we tried it without normalization and almost all values were clipped because they ranged from 0-255)
    
    # from page 10 of paper
    M = torch.tensor([[2.0365, -0.7376, -0.2993],
                      [-0.2257, 1.2232, 0.0027],
                      [-0.0105, -0.1349, 1.1452]]).to(device)
    unclipped_sRGB = torch.matmul(M, I_PP)

    # transform unclipped sRGB such that in-gamut sRGB values are within range [0,1]
    # locate out-of-gamut (OG) sRGB values, row-wise
    # True = OG, False = in-gamut (IG)
    R_mask = (unclipped_sRGB[0] < 0) | (unclipped_sRGB[0] > 1)
    G_mask = (unclipped_sRGB[1] < 0) | (unclipped_sRGB[1] > 1)
    B_mask = (unclipped_sRGB[2] < 0) | (unclipped_sRGB[2] > 1)
    
    # ASSUMPTION: a pixel is considered OG if any one of its colour values is out of [0,1] range
    # we assume this is the correct assumption because Figures 2 and 3 show "clipped values" as (R', G', B') --> signifies all 3 color values of a pixel are clipped
    # True = Pixel is OG, False = Pixel is in-gamut (IG)
    OG_mask = torch.logical_or(R_mask, torch.logical_or(G_mask, B_mask))
    OG_mask = torch.stack((OG_mask, OG_mask, OG_mask))

    # clip, no values outside of [0,1] range
    clipped_sRGB = torch.clamp(unclipped_sRGB, min=0, max=1)
    
    # gamma encoding
    I_sRGB = gamma_encoding_for_sRGB(clipped_sRGB)
    
    # Equation (2)
    M_inverse = torch.linalg.inv(M)
    de_gamma_I_sRGB = gamma_decoding_for_sRGB(I_sRGB)

    I_ClippedPP = torch.matmul(M_inverse, de_gamma_I_sRGB)
    
    return I_PP, I_ClippedPP, OG_mask