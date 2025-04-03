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

import cv2
import numpy as np
import colour
import matplotlib.pyplot as plt
import utils

# let's take a look at the first image
# img_prophoto_gamma = cv2.imread('datasets/train/vivid_SamsungNX2000_0160.png')
# img_prophoto_gamma = cv2.imread('datasets/test/vivid_re757c387t.png')
# print('Type: ', type(img_prophoto_gamma))
# print('Datatype:', img_prophoto_gamma.dtype, '\nDimensions:', img_prophoto_gamma.shape) #unit8 datatype means values range from 0-255 = GAMMA

# *IMP* check if linear or gamma encoding
# is_linear = "Linear" if cv2.imread("datasets/train/standard_r9e70c57ft.png", cv2.IMREAD_ANYDEPTH).dtype == "float32" else "Gamma"
# print(is_linear)

# *IMP* display image
# cv2.imshow('ORIGINAL', img_prophoto_gamma) 
# cv2.waitKey(0)

# TODO: create a DataLoader 
# TODO: retrieve the Clipped ProPhoto and Out-of-Gamut (OG) photos

# Clipped ProPhoto image is obtained through gamut reduction (absolute colorimetric intent)
# From equation 1: sRGB = g(clip(MI, min=0, max=1))
# - M: matrix that maps ProPhoto to the unclipped sRGB (provided in Appendix A)
# - I: original ProPhoto image
# - clip: clipping operation
# - g: gamma encoding 


def gamma_encoding_for_sRGB(linear_img):
    # https://en.wikipedia.org/wiki/SRGB#Transfer_function_(%22gamma%22) 
    return np.where(linear_img <= 0.0031308, 12.92 * linear_img, 1.055 * np.power(linear_img, 1/2.4) - 0.055)

    # https://colour.readthedocs.io/en/latest/generated/colour.cctf_encoding.html
    # return colour.cctf_encoding(linear_img, function='sRGB')

def gamma_decoding_for_sRGB(gamma_img):
    # https://en.wikipedia.org/wiki/SRGB#Transfer_function_(%22gamma%22) 
    return np.where(gamma_img <= 0.04045, gamma_img / 12.92, ((gamma_img + 0.055) / 1.055) ** 2.4)

    # https://colour.readthedocs.io/en/latest/generated/colour.cctf_encoding.html
    # return colour.cctf_decoding(gamma_img, function='sRGB')

def gamut_reduction(I_PP):
    # define M as described in the paper
    # print ('img', I_PP)
    I_PP = I_PP / 255.0  # ASSUMPTION: must normalize to floating pt format before computations (because we tried it without normalization and almost all values were clipped because they ranged from 0-255)
    # print(I_PP.dtype) # ASSUMPTION: working with float64, no values outside of [0,1] range
    # print(I_PP.min(), I_PP.max())
    # print('img normalized', I_PP) 

    # from page 10 of paper
    M = np.array([[2.0365, -0.7376, -0.2993],
                [-0.2257, 1.2232, 0.0027],
                [-0.0105, -0.1349, 1.1452]])
    
    # unclipped sRGB, M(I_PP), some resulting values fall outside the [0,1] range
    unclipped_sRGB = np.dot(M, I_PP)
    # transform unclipped sRGB such that in-gamut sRGB values are within range [0,1]
    # print('unclipped_sRGB', unclipped_sRGB)

    # locate out-of-gamut (OG) sRGB values, row-wise
    # True = OG, False = in-gamut (IG)
    R_mask = (unclipped_sRGB[0] < 0) | (unclipped_sRGB[0] > 1)
    G_mask = (unclipped_sRGB[1] < 0) | (unclipped_sRGB[1] > 1)
    B_mask = (unclipped_sRGB[2] < 0) | (unclipped_sRGB[2] > 1)
    
    # ASSUMPTION: a pixel is considered OG if any one of its colour values is out of [0,1] range
    # we assume this is the correct assumption because Figures 2 and 3 show "clipped values" as (R', G', B') --> signifies all 3 color values of a pixel are clipped
    # True = Pixel is OG, False = Pixel is in-gamut (IG)
    OG_mask = np.logical_or(R_mask, G_mask, B_mask)
    # print(OG_mask.shape)
    OG_mask = np.stack((OG_mask, OG_mask, OG_mask))
    # print(OG_mask.shape)
    # print(OG_mask)
    # print("====================================")

    # clip, no values outside of [0,1] range
    clipped_sRGB = np.clip(unclipped_sRGB, 0, 1)
    # print('clipped_sRGB', clipped_sRGB)
    # print("====================================")

    # gamma encoding
    I_sRGB = gamma_encoding_for_sRGB(clipped_sRGB)
    # nonlinear_srgb = gamma_encoding(clipped_sRGB)
    # print('I_sRGB', I_sRGB)
    # print("=====WORKS UNTIL HERE, done equation 1===============================")

    # Equation (2)
    M_inverse = np.linalg.inv(M)
    de_gamma_I_sRGB = gamma_decoding_for_sRGB(I_sRGB)

    I_ClippedPP = np.dot(M_inverse, de_gamma_I_sRGB)
    # print(I_ClippedPP.min(), I_ClippedPP.max())
    # print("=====WORKS UNTIL HERE, done equation 2===============================")
    
    return I_PP, I_ClippedPP, OG_mask

# # reshape the image to be 3 x N (where N is the number of pixels)
# # the original image is 512 x 512 x 3
# # img_prophoto_gamma.show()
# I_PP = img_prophoto_gamma.reshape(-1, 3).T # got this from gamut github code

# I_PP, I_ClippedPP, OG_mask = gamut_reduction(I_PP)
# display_OG_mask = (OG_mask).astype(np.float32) # convert to BW image, recall OG_mask is a boolean array

# # ASSUMPTION: cv2 can handle normalized FLOAT version of pixels without multiplying by 255
# # change shape for display
# display_I_PP = I_PP.T.reshape(512,512,3) # ASSUMPTION: we tried to "undo" line 112
# display_I_ClippedPP = I_ClippedPP.T.reshape(512,512,3)
# display_OG_mask = display_OG_mask.T.reshape(512,512,3) # A white pixel represents OG pixel (value = 1/True), black represents IG (value = 0/False)
# # cv2.imshow('Displaying I_PP image using OpenCV', display_I_PP) 
# # cv2.imshow('Displaying I_ClippedPP image using OpenCV', display_I_ClippedPP) 
# # cv2.imshow('Displaying OG_mask image using OpenCV', display_OG_mask)

# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# # SECTION 3.2: GamutMLP model and optimization
# utils.generatingTrainingInputAndGroundTruth(display_I_PP, display_I_ClippedPP, display_OG_mask)