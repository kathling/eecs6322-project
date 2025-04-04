'''
This file handles loading the dataset and generating the training data for the GamutMLP model.
The main function is generate_normalized_training_data which samples pixels from the original ProPhoto image
and applies normalization and encoding to retrieve a 120D feature vector for each pixel.
The training data is then used to train the GamutMLP model.
'''

import torch
import numpy as np

# flag for gpu
device = 'cpu'
if (torch.cuda.is_available()):
    device = torch.device('cuda')

# =================================================================================================
# Functions for Gamut Reduction
# =================================================================================================
def gamma_encoding_for_sRGB(linear_img):
    # https://en.wikipedia.org/wiki/SRGB#Transfer_function_(%22gamma%22) 
    return torch.where(linear_img <= 0.0031308, 12.92 * linear_img, 1.055 * torch.pow(linear_img, 1/2.4) - 0.055)

    # Alternate: https://colour.readthedocs.io/en/latest/generated/colour.cctf_encoding.html
    # return colour.cctf_encoding(linear_img, function='sRGB')

def gamma_decoding_for_sRGB(gamma_img):
    # https://en.wikipedia.org/wiki/SRGB#Transfer_function_(%22gamma%22) 
    return torch.where(gamma_img <= 0.04045, gamma_img / 12.92, ((gamma_img + 0.055) / 1.055) ** 2.4)

    # Alternate: https://colour.readthedocs.io/en/latest/generated/colour.cctf_encoding.html
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


# =================================================================================================
# Functions to generate training data
# =================================================================================================
def generate_normalized_training_data(I_PP, I_ClippedPP, OG_mask):
    # Step 1: Sampling pixels from the original PP image and generate ground truth
    coords_5D, ground_truth = sampling(I_PP, I_ClippedPP, OG_mask)
    
    # Step 2: Normalize
    normalized_coords_5D = normalize(coords_5D)

    # Step 3: Encode
    train_data = encode(normalized_coords_5D)

    return train_data, ground_truth, coords_5D

def sampling(I_PP, I_ClippedPP, OG_mask):
    '''
    Select 2% of IG pixels and 20% of OG pixels uniformly over spatial coordinates.

    There are two methods:
        - Method 1: Use I_PP, I_ClippedPP, OG_mask which are direct output of gamut_reduction() method, with shape (3,512*512) 
        - Method 2: Use display_I_PP, display_I_ClippedPP, display_OG_mask which are in display form with shape (512,512,3)
    
    We used Method 2 because it is easier to find spatial coordinates.
    '''

    # Use only one channel of OG_mask since all 3 channels are identical (because all 3 channels represent 1 pixel which is either OG (1) or IG (0))
    one_channel_OG_mask = OG_mask[:,:,0]

    # Find spatial coordinates of IG and OG pixels separately
    # ASSUMPTION: x is row # (first number returned) and y is column # (second number returned)
    IG_spatial_coordinates = torch.nonzero(one_channel_OG_mask==0)
    OG_spatial_coordinates = torch.nonzero(one_channel_OG_mask==1)

    # Randomly sample from IG and OG pixels
    # ASSUMPTION: Uniform sampling in IG and uniform sampling in OG, not across ALL spatial coordinates
    IG_spatial_coordinates = IG_spatial_coordinates.cpu()
    OG_spatial_coordinates = OG_spatial_coordinates.cpu()    
    random_generator = np.random.default_rng()
    sample_IG_coords = random_generator.choice(IG_spatial_coordinates, int(0.02*len(IG_spatial_coordinates)), replace=False)
    sample_OG_coords = random_generator.choice(OG_spatial_coordinates, int(0.2*len(OG_spatial_coordinates)), replace=False)

    # save back to tensor (move to GPU)
    IG_spatial_coordinates = IG_spatial_coordinates.to(device)
    OG_spatial_coordinates = OG_spatial_coordinates.to(device) 
    
    # ===============================================================
    # in-gamut (IG) coords, (x, y, R, G, B)
    # ===============================================================
    sample_IG_coords = torch.tensor(sample_IG_coords).to(device)
    x_IG = sample_IG_coords[:, 0]
    y_IG = sample_IG_coords[:, 1]
    R_prime = I_ClippedPP[x_IG, y_IG, 0]
    G_prime = I_ClippedPP[x_IG, y_IG, 1]
    B_prime = I_ClippedPP[x_IG, y_IG, 2]
    IG_5d_coords = torch.stack((x_IG, y_IG, R_prime, G_prime, B_prime), dim=1)
    
    # ground truth
    R_IG = I_PP[x_IG, y_IG, 0]
    G_IG = I_PP[x_IG, y_IG, 1]
    B_IG = I_PP[x_IG, y_IG, 2]
    IG_gt_5d_coords = torch.stack((x_IG, y_IG, R_IG, G_IG, B_IG), dim=1)

    # ===============================================================
    # out-of-gamut (OG) coords, (x, y, R, G, B)
    # ===============================================================
    sample_OG_coords = torch.tensor(sample_OG_coords).to(device)
    x_OG = sample_OG_coords[:, 0]
    y_OG = sample_OG_coords[:, 1]
    R_prime = I_ClippedPP[x_OG, y_OG, 0]
    G_prime = I_ClippedPP[x_OG, y_OG, 1]
    B_prime = I_ClippedPP[x_OG, y_OG, 2]
    OG_5d_coords = torch.stack((x_OG, y_OG, R_prime, G_prime, B_prime), dim=1)

    # ground truth
    R_OG = I_PP[x_OG, y_OG, 0]
    G_OG = I_PP[x_OG, y_OG, 1]
    B_OG = I_PP[x_OG, y_OG, 2]
    OG_gt_5d_coords = torch.stack((x_OG, y_OG, R_OG, G_OG, B_OG), dim=1)

    all_5d_coords = torch.cat((IG_5d_coords, OG_5d_coords), dim=0)
    ground_truth_5d_coords = torch.cat((IG_gt_5d_coords, OG_gt_5d_coords), dim=0)
    
    return all_5d_coords, ground_truth_5d_coords

def normalize(all_unnormalized_5d_coords):
    # ASSUMPTION: normalize x wrt all x's selected, normalize y wrt to all y's selected, etc.
    # not doing: other assumption (if above doesn't work): normalize wrt to a set of 5 input values x,y,r_,g_,b_
    # normalize to range between -1 and 1
    # get the min and max values in each column
    min = all_unnormalized_5d_coords.min(axis=0).values
    max = all_unnormalized_5d_coords.max(axis=0).values
    all_normalized_5d_coords = 2*(all_unnormalized_5d_coords-min)/(max-min) - 1

    return all_normalized_5d_coords

def encodingFunctionGamma(z):
    K = 12          # number fixed in the paper
    result = []
    for i in range(0,K):
        result.append((torch.sin(2**i * torch.pi * z)))
        result.append((torch.cos(2**i * torch.pi * z)))
    
    res = torch.stack(result, dim=0)      # torch.Size([24, num samples])
    return res
    
def encode(all_normalized_5d_coords):
    encoded_coords = torch.cat([
        encodingFunctionGamma(all_normalized_5d_coords[:, 0]),    # x
        encodingFunctionGamma(all_normalized_5d_coords[:, 1]),    # y
        encodingFunctionGamma(all_normalized_5d_coords[:, 2]),    # R
        encodingFunctionGamma(all_normalized_5d_coords[:, 3]),    # G
        encodingFunctionGamma(all_normalized_5d_coords[:, 4]),    # B
    ]).T
    
    return encoded_coords

# =================================================================================================
# Functions for Gamut Expansion
# =================================================================================================
def gamut_expansion(I_ClippedPP):
    # Gamut Expansion Step, (just getting the 120D feature vectors for all pixels)
    # Method 2: Use display_I_PP, display_I_ClippedPP, display_OG_mask which are in display form with shape (512,512,3)
    # Assumption: using Method 2 (defined in utils.py sampling() method) due to ease in finding spatial coordinates

    # ASSUMPTION: "all pixels" refers to all pixels in the image (not just out-of-gamut)
    # From page 3, gamut expansion section:
    # "The extracted model predicts the residuals of all pixels and adds them to the IClippedPP to recover the color values as shown in Figure 3." 
    
    # get height and width
    height, width = I_ClippedPP.shape[0], I_ClippedPP.shape[1]
    x_range = torch.arange(height, device=device)
    y_range = torch.arange(width, device=device)

    # create mesh grid (coordinates)
    # https://pytorch.org/docs/stable/generated/torch.meshgrid.html
    x_coord, y_coord = torch.meshgrid(x_range, y_range, indexing="ij")

    # flatten the coordinates
    x_coord = x_coord.flatten()  # torch.Size([width*height])
    y_coord = y_coord.flatten()  # torch.Size([width*height])

    rgb = I_ClippedPP.reshape(-1, 3).T   # rgb shape torch.Size([3, width*height])
    all_5d_coords = torch.stack((
        x_coord,
        y_coord,
        rgb[0,:],
        rgb[1,:],
        rgb[2,:]
    ), dim=1)

    # normalize
    normalized_all_5d_coords = normalize(all_5d_coords)

    # encode
    encoded_all_5d_coords = encode(normalized_all_5d_coords)

    return encoded_all_5d_coords, all_5d_coords
