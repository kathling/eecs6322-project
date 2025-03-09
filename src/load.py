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

from PIL import Image
import numpy as np
import colour

# let's take a look at the first image
img = Image.open('datasets/train/standard_r9e70c57ft.png')
# print(img.info.get('icc_profile'))
# img.show()

# TODO: create a DataLoader 
# TODO: retrieve the Clipped ProPhoto and Out-of-Gamut (OG) photos

# Clipped ProPhoto image is obtained through gamut reduction (absolute colorimetric intent)
# From equation 1: sRGB = g(clip(MI, min=0, max=1))
# - M: matrix that maps ProPhoto to the unclipped sRGB (provided in Appendix A)
# - I: original ProPhoto image
# - clip: clipping operation
# - g: gamma encoding 


def gamma_encoding(linear_img):
    # https://en.wikipedia.org/wiki/SRGB#Transfer_function_(%22gamma%22) 
    # return np.where(linear_img <= 0.0031308, 12.92 * linear_img, 1.055 * np.power(linear_img, 1/2.4) - 0.055)

    # https://colour.readthedocs.io/en/latest/generated/colour.cctf_encoding.html
    return colour.cctf_encoding(linear_img, function='sRGB')

def gamut_reduction(img):
    # define M as described in the paper
    print ('img', img)
    print(img.max())
    print(img.dtype)
    img = img / 255.0           # must normalize to floating pt format before computations
    print('img normalized', img)
    print("====================================")

    M = np.array([[2.0365, -0.7376, -0.2993],
                [-0.2257, 1.2232, 0.0027],
                [-0.0105, -0.1349, 1.1452]])
    
    # unclipped sRGB, M(I_PP)
    unclipped_sRGB = np.dot(M, img)
    # # transform unclipped sRGB such that in-gamut sRGB values are within range [0,1]
    # unclipped_sRGB = unclipped_sRGB/ 255.0
    print('unclipped_sRGB', unclipped_sRGB)


    print("====================================")

    # clip
    clipped_sRGB = np.clip(unclipped_sRGB, 0, 1)
    print('clipped_sRGB', clipped_sRGB)
    print("====================================")

    # # obtain the out of gamut (OG) pixels
    # OG = np.where((unclipped_sRGB < 0) | (unclipped_sRGB > 1))

    # gamma encoding
    nonlinear_srgb = gamma_encoding(unclipped_sRGB)
    # nonlinear_srgb = gamma_encoding(clipped_sRGB)
    print('nonlinear_srgb', nonlinear_srgb)
    print("====================================")

    return nonlinear_srgb

# reshape the image to be 3 x N (where N is the number of pixels)
# the original image is 512 x 512 x 3
img.show()
temp = np.reshape(img, (3, 512*512))

res = gamut_reduction(temp)

# print('OG shape', OG.shape)
# OG_image = OG.reshape(3, 512, 512).transpose(1, 2, 0)
image = res.reshape(3, 512, 512).transpose(1, 2, 0)
image = (image * 255).astype(np.uint8)
print(image)
print(image.shape)
Image.fromarray(image).show()
# Image.fromarray(OG_image).show()