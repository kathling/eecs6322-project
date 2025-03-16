import numpy as np

def sampling(I_PP, I_ClippedPP):
    # TO DO: select 5% of IG pixels and 20% of OG pixels uniformly over spatial coordinates
    
    # ASSUMPTION: normalize x wrt all x's selected, normalize y wrt to all y's selected, etc.
    # other assumption (if above doesn't work): normalize wrt to a set of 5 input values x,y,r_,g_,b_
    pass

def encode(x, y, R_, G_, B_):
    # Step 1: Normalize each value to range [-1, 1]
    x = normalize(x)
    y = normalize(y)
    R_ = normalize(R_)
    G_ = normalize(G_)
    B_ = normalize(B_)

    # Step 2: Pass each value to encoding function gamma
    x = encodingFunctionGamma(x)
    y = encodingFunctionGamma(y)
    R_ = encodingFunctionGamma(R_)
    G_ = encodingFunctionGamma(G_)
    B_ = encodingFunctionGamma(B_)

    final_input = np.concatenate([x, y, R_, G_, B_])
    return final_input

def normalize(z):
    min = np.min(z)
    max = np.max(z)
    return 2*(z-min)/(max-min) - 1

def encodingFunctionGamma(z):
    K = 12
    result = []
    for i in range(0,K):
        result.append((np.sin(2**i * np.pi * z)))
        result.append((np.cos(2**i * np.pi * z)))
        
    return np.array(result)
        

def model_Input_Preprocessing(I_PP, I_ClippedPP):
    sampling(I_PP, I_ClippedPP)
    pass

res = encode(1,1,1,1,10)
print(res.shape)
print(res)