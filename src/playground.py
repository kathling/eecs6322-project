# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
import torch
print('pytorch version: ', torch.__version__)
print('is_cuda_avail: ', torch.cuda.is_available())

# import cv2
# import numpy as np
# import load 
# import utils
# # load in one image
# img_prophoto_gamma = cv2.imread('datasets/train/standard_r9e70c57ft.png')
# # cv2.imshow('ORIGINAL', img_prophoto_gamma) 
# I_PP = img_prophoto_gamma.reshape(-1, 3).T # got this from gamut github code
# I_PP, I_ClippedPP, OG_mask = load.gamut_reduction(I_PP)
# display_OG_mask = (OG_mask).astype(np.float32) # convert to BW image, recall OG_mask is a boolean array
# display_I_PP = I_PP.T.reshape(512,512,3) # ASSUMPTION: we tried to "undo" line 112
# display_I_ClippedPP = I_ClippedPP.T.reshape(512,512,3)
# display_OG_mask = display_OG_mask.T.reshape(512,512,3) # A white pixel represents OG pixel (value = 1/True), black represents IG (value = 0/False)

# print('I_ClippedPP', I_ClippedPP)
# print('shape of I_ClippedPP', I_ClippedPP.shape)

# print('display_I_ClippedPP', display_I_ClippedPP)
# print('shape of display_I_ClippedPP', display_I_ClippedPP.shape)

# utils.gamut_expansion(display_I_ClippedPP)
