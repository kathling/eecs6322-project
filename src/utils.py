import numpy as np

def generatingTrainingInputAndGroundTruth(I_PP, I_ClippedPP, OG_mask):
    # Step 1: Sampling (also generates ground truth for those samples)
    # ground_truth: original PP values 5D coordinates (x, y, R, G, B)
    all_5d_coords, ground_truth = sampling(I_PP, I_ClippedPP, OG_mask)
    I_ClippedPP_5d_coords = all_5d_coords # note we need this for equation 3

    # Step 2: Normalize
    normalized_all_5d_coords = normalize(all_5d_coords)

    # Step 3: Encode
    final_training_input = encode(normalized_all_5d_coords)

    return final_training_input, ground_truth, I_ClippedPP_5d_coords


def sampling(I_PP, I_ClippedPP, OG_mask):
    # select 2% of IG pixels and 20% of OG pixels uniformly over spatial coordinates

    # Method 1: Use I_PP, I_ClippedPP, OG_mask which are direct output of gamut_reduction() method, with shape (3,512*512) 
    # Method 2: Use display_I_PP, display_I_ClippedPP, display_OG_mask which are in display form with shape (512,512,3)
    # Assumption: using Method 2 (defined in utils.py sampling() method) due to ease in finding spatial coordinates
    
    # print(OG_mask)
    # temp_OG_mask = np.where(OG_mask==0, 5, 10) # converts 0s into 5s and 1s into 10s
    # print("-------------")
    # print(temp_OG_mask)

    # Use only one channel of OG_mask since all 3 channels are identical (because all 3 channels represent 1 pixel which is either OG (1) or IG (0))
    one_channel_OG_mask = OG_mask[:,:,0]

    # Find spatial coordinates of IG and OG pixels separately
    # ASSUMPTION: x is row # (first number returned) and y is column # (second number returned)
    IG_spatial_coordinates = np.argwhere(one_channel_OG_mask==0)
    OG_spatial_coordinates = np.argwhere(one_channel_OG_mask==1)

    # Randomly sample from IG and OG pixels
    # ASSUMPTION: Uniform sampling in IG and uniform sampling in OG, not across ALL spatial coordinates
    random_generator = np.random.default_rng()
    sample_IG_coords = random_generator.choice(IG_spatial_coordinates, int(0.02*len(IG_spatial_coordinates)), replace=False)
    sample_OG_coords = random_generator.choice(OG_spatial_coordinates, int(0.2*len(OG_spatial_coordinates)), replace=False)

    sample_IG_coords = list(sample_IG_coords)
    sample_OG_coords = list(sample_OG_coords)

    # ------------ generate ground truth -----------
    ground_truth_5d_coords = []
    # ------------ generate ground truth -----------
    all_5d_coords = [] # list of lists
    for coords in sample_IG_coords:
        R_prime = I_ClippedPP[coords[0], coords[1], 0]
        G_prime = I_ClippedPP[coords[0], coords[1], 1]
        B_prime = I_ClippedPP[coords[0], coords[1], 2]
        FiveD_coords = [coords[0], coords[1], R_prime, G_prime, B_prime]
        all_5d_coords.append(FiveD_coords)

        # ------------ generate ground truth -----------
        R = I_PP[coords[0], coords[1], 0]
        G = I_PP[coords[0], coords[1], 1]
        B = I_PP[coords[0], coords[1], 2]
        ground_truth_FiveD_coords = [coords[0], coords[1], R, G, B]
        ground_truth_5d_coords.append(ground_truth_FiveD_coords)

    for coords in sample_OG_coords:
        R_prime = I_ClippedPP[coords[0], coords[1], 0]
        G_prime = I_ClippedPP[coords[0], coords[1], 1]
        B_prime = I_ClippedPP[coords[0], coords[1], 2]
        FiveD_coords = [coords[0], coords[1], R_prime, G_prime, B_prime]
        all_5d_coords.append(FiveD_coords)

        # ------------ generate ground truth -----------
        R = I_PP[coords[0], coords[1], 0]
        G = I_PP[coords[0], coords[1], 1]
        B = I_PP[coords[0], coords[1], 2]
        ground_truth_FiveD_coords = [coords[0], coords[1], R, G, B]
        ground_truth_5d_coords.append(ground_truth_FiveD_coords)

    # convert to array for easier normalization
    all_5d_coords = np.array(all_5d_coords)
    ground_truth_5d_coords = np.array(ground_truth_5d_coords)
    
    return all_5d_coords, ground_truth_5d_coords

def normalize(all_unnormalized_5d_coords):
    # ASSUMPTION: normalize x wrt all x's selected, normalize y wrt to all y's selected, etc.
    # not doing: other assumption (if above doesn't work): normalize wrt to a set of 5 input values x,y,r_,g_,b_
    # normalize to range between -1 and 1
    # get the min and max values in each column
    min = all_unnormalized_5d_coords.min(axis=0)
    max = all_unnormalized_5d_coords.max(axis=0)
    all_normalized_5d_coords = 2*(all_unnormalized_5d_coords-min)/(max-min) - 1

    return all_normalized_5d_coords

def encode(all_normalized_5d_coords):
    
    all_encoded_normalized_5d_coords = []
    for five_coords in all_normalized_5d_coords:
        x = encodingFunctionGamma(five_coords[0])
        y = encodingFunctionGamma(five_coords[1])
        R_ = encodingFunctionGamma(five_coords[2])
        G_ = encodingFunctionGamma(five_coords[3])
        B_ = encodingFunctionGamma(five_coords[4])
        encoded_coords = np.concatenate([x, y, R_, G_, B_])
        all_encoded_normalized_5d_coords.append(encoded_coords)

    # convert to array
    all_encoded_normalized_5d_coords = np.array(all_encoded_normalized_5d_coords)

    return all_encoded_normalized_5d_coords

def encodingFunctionGamma(z):
    K = 12
    result = []
    for i in range(0,K):
        result.append((np.sin(2**i * np.pi * z)))
        result.append((np.cos(2**i * np.pi * z)))
        
    return np.array(result)


# Gamut Expansion Step, (just getting the 120D feature vectors for all pixels)
# Method 2: Use display_I_PP, display_I_ClippedPP, display_OG_mask which are in display form with shape (512,512,3)
# Assumption: using Method 2 (defined in utils.py sampling() method) due to ease in finding spatial coordinates
def gamut_expansion(I_ClippedPP):
    # ASSUMPTION: "all pixels" refers to all pixels in the image (not just out-of-gamut)
    # From page 3, gamut expansion section:
    # "The extracted model predicts the residuals of all pixels and adds them 
    # to the IClippedPP to recover the color values as shown in Figure 3." 
    # normalize and encode all pixels
    # test line for windows computer
    all_5d_coords = [] # list of lists
    # get height and width
    height, width = I_ClippedPP.shape[0], I_ClippedPP.shape[1]
    for x in range(height):
        for y in range(width):
            R_prime = I_ClippedPP[x, y, 0]
            G_prime = I_ClippedPP[x, y, 1]
            B_prime = I_ClippedPP[x, y, 2]
            FiveD_coords = [x, y, R_prime, G_prime, B_prime]
            all_5d_coords.append(FiveD_coords)
    # convert to array for easier normalization

    # this is original I_ClippedPP 5D coordinates
    all_5d_coords = np.array(all_5d_coords)

    # normalize
    normalized_all_5d_coords = normalize(all_5d_coords)

    # encode
    encoded_all_5d_coords = encode(normalized_all_5d_coords)

    print('encoded: ', encoded_all_5d_coords)
    print('shape: ', encoded_all_5d_coords.shape)

    return encoded_all_5d_coords, all_5d_coords
