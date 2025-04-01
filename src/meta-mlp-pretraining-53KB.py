# Pre-training meta-GamutMLP (53KB) on 100 512x512 ProPhoto images in the train dataset.

import torch
import numpy as np
import cv2
import utils
import load
import time
import os

from model import GamutMLP_64HF_53KB
from metrics import RMSE, PSNR


# flag for gpu
device = 'cpu'
if (torch.cuda.is_available):
    device = torch.device('cuda')

META_EPOCH=2

if __name__ == "__main__":
    folder_path = "datasets/train/"
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_paths = image_paths[:100]

    # ==================================================================
    # Build model
    # 
    # moved before meta epoch because we are training the same model on 
    # all training images (not per image)
    # ==================================================================
    net = GamutMLP_64HF_53KB().to(device)
    loss_fn =  torch.nn.MSELoss()
    learning_rate = 0.01                                            # <<< CHANGED FROM 0.001 to 0.01
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate) # <<< CHANGED FROM ADAM TO SGD

    # ==================================================================
    # Train loop
    # ==================================================================
    print('training start') 
    training_start_time = time.time()   
    counter = 0
    for epoch in range(META_EPOCH):
        for img_path in image_paths:
            img_prophoto_gamma = cv2.imread(img_path)
            height, width = img_prophoto_gamma.shape[0], img_prophoto_gamma.shape[1]

            I_PP = img_prophoto_gamma.reshape(-1, 3).T # got this from gamut github code
            I_PP, I_ClippedPP, OG_mask = load.gamut_reduction(I_PP)
            display_OG_mask = (OG_mask).astype(np.float32) # convert to BW image, recall OG_mask is a boolean array
            display_I_PP = I_PP.T.reshape(height,width,3) # ASSUMPTION: we tried to "undo" line 112
            display_I_ClippedPP = I_ClippedPP.T.reshape(height,width,3)
            display_OG_mask = display_OG_mask.T.reshape(height,width,3)
            training_input, ground_truth, I_ClippedPP_5d_coords = utils.generatingTrainingInputAndGroundTruth(display_I_PP, display_I_ClippedPP, display_OG_mask)

            start_time = time.time() # optimization start time
            for iter in range(10000):                                       # <<< CHANGED FROM 9000 to 10000
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

                # # print out stats every 10 its
                # if iter % 1000 == 0:
                #     print(f'Iteration: {iter} | Loss: {loss.item()}')
            end_time = time.time() # optimization start time
            optim_time = end_time - start_time
            print('optimization time: ', optim_time)

            counter = counter+1
            if counter % 10 == 0:
                print('IMAGES TRAINED ON: ', counter)

    training_end_time = time.time()
    total_train_time = training_end_time - training_start_time
    print(f"TOTAL TRAINING TIME:: {total_train_time}")

    # save to file
    curr_path = os.getcwd()
    folder_name = 'logs'
    if not os.path.exists(f'{curr_path}/{folder_name}'):
        os.makedirs(folder_name)
    with open(f'{folder_name}/meta-mlp-pretrain-53KB-100im.txt', 'w') as f:
        f.write(f'total training time to pre-train GamutMLP (53KB) on 100 images: {total_train_time}\n')
    f.close()
    
    # save the model to a checkpoint for future loading
    folder_name = 'pretrained'
    if not os.path.exists(f'{curr_path}/{folder_name}'):
        os.makedirs(folder_name)
    torch.save(net.state_dict(), os.path.join('meta-mlp_state_dict_53KB-100.pt'))