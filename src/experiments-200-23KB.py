# To reproduce the second highlighted row in Table 1
# MLP (23KB) + enc.

import torch
import numpy as np
import cv2
import utils
import load
import time
from metrics import RMSE, PSNR
import os

from model import GamutMLP


# flag for gpu
device = 'cpu'
if (torch.cuda.is_available):
    device = torch.device('cuda')

if __name__ == "__main__":
    folder_path = "datasets/test/"
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_paths = image_paths[:5]

    # used to track statistics for Table 1
    rmse_loss = []
    rmse_og_loss = []
    psnr_loss = []
    psnr_og_loss = []
    optim_times = []

    counter = 0
    for img_path in image_paths:
        total_start_time = time.time()
        img_prophoto_gamma = cv2.imread(img_path)

        # get the height and width of the photo
        height, width = img_prophoto_gamma.shape[0], img_prophoto_gamma.shape[1]
        I_PP = img_prophoto_gamma.reshape(-1, 3).T # got this from gamut github code
        I_PP, I_ClippedPP, OG_mask = load.gamut_reduction(I_PP)
        display_OG_mask = (OG_mask).astype(np.float32) # convert to BW image, recall OG_mask is a boolean array
        display_I_PP = I_PP.T.reshape(height,width,3) # ASSUMPTION: we tried to "undo" line 112
        display_I_ClippedPP = I_ClippedPP.T.reshape(height,width,3)
        display_OG_mask = display_OG_mask.T.reshape(height,width,3)
        training_input, ground_truth, I_ClippedPP_5d_coords = utils.generatingTrainingInputAndGroundTruth(display_I_PP, display_I_ClippedPP, display_OG_mask)

        # build network
        net = GamutMLP().to(device)
        loss_fn =  torch.nn.MSELoss()
        learning_rate = 0.001
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    
        # ==================================================================
        # Train loop
        # ==================================================================
        print('training start')        
        start_time = time.time() # optimization start time
        for iter in range(9000): # num of optimization iterations
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

            # print out stats every 1000 its
            if iter % 1000 == 0:
                print(f'Iteration: {iter} | Loss: {loss.item()}')

        end_time = time.time() # optimization start time
        print("training complete")

        # Now, let's see how well GamutMLP restores the clipped PP image
        encoded_5d_coords, I_Clipped_PP_5d_coords = utils.gamut_expansion(display_I_ClippedPP)
        
        with torch.no_grad():
            net.eval()
            x = torch.tensor(encoded_5d_coords).float().to(device)
            pred_residuals = net(x).to('cpu')
        
        # Equation 3, the recovered/predicted wide-gamut PP image
        # convert to torch tensor
        all_pixels_restored_values = torch.tensor(I_Clipped_PP_5d_coords[:, -3:]).float() + pred_residuals

        # Restore the image
        restored_I_PP = all_pixels_restored_values.view(height, width, 3)
        restored_I_PP = restored_I_PP.detach().cpu().numpy()
        
        # calculate metrics for OG
        # convert OG (np array to tensor)
        tensor_display_OG_mask = torch.from_numpy(display_OG_mask)
        tensor_restored_I_PP = torch.from_numpy(restored_I_PP)
        tensor_display_I_PP = torch.from_numpy(display_I_PP)

        OG_pixels_restored_I_PP = tensor_restored_I_PP[tensor_display_OG_mask == 1]
        OG_pixels_display_I_PP = tensor_display_I_PP[tensor_display_OG_mask == 1]

        
        rmse = RMSE(tensor_display_I_PP, tensor_restored_I_PP)
        rmse_og = RMSE(OG_pixels_display_I_PP, OG_pixels_restored_I_PP)
        psnr = PSNR(tensor_display_I_PP, tensor_restored_I_PP)
        psnr_og = PSNR(OG_pixels_display_I_PP, OG_pixels_restored_I_PP)
        optim_time = end_time - start_time

        print('RMSE: ', rmse)
        print('RMSE for OG: ', rmse_og)
        print('PSNR: ', psnr)
        print('PSNR for OG: ', psnr_og)
        print('optimization time: ', optim_time)

        # track these metrics
        rmse_loss.append(rmse)
        rmse_og_loss.append(rmse_og)
        psnr_loss.append(psnr)
        psnr_og_loss.append(psnr_og)
        optim_times.append(optim_time)

        if counter % 1 == 0:
            print('EXAMPLES COVERED: ', counter+1)

        # let's just save the image 
        curr_path = os.getcwd()
        folder_name = 'restored/mlp23'
        if not os.path.exists(f'{curr_path}/{folder_name}'):
            os.makedirs(folder_name)
        file_name_w_ext = os.path.basename(img_path)

        print('saved image as: ', file_name_w_ext)
        save_img_path = os.path.join(folder_name, f'RESTORED_{file_name_w_ext}')
        restored_I_PP = (restored_I_PP * 255).astype(np.uint8)
        cv2.imwrite(save_img_path, restored_I_PP)

        # store these results for now in a temp text file
        with open('running_output.txt', "a") as f:
            f.write(file_name_w_ext+'\n')
            f.write(f'>>> RMSE: {rmse}\n')
            f.write(f'>>> RMSE OG: {rmse_og}\n')
            f.write(f'>>> PSNR: {psnr}\n')
            f.write(f'>>> PSNR OG: {psnr_og}\n')
            f.write(f'>>> OPTIM TIME: {optim_time}\n')
            f.write('=============================================\n')
        f.close()

        total_end_time = time.time()
        print("TOTAL TIME TO RUN THROUGH ONE IMAGE: ", total_end_time - total_start_time)

    # print out the average metrics
    print("="*80)
    print('FINAL RESULTS')
    print('>>> AVERAGE RMSE: ', sum(rmse_loss)/len(rmse_loss))
    print('>>> AVERAGE RMSE OG: ', sum(rmse_og_loss)/len(rmse_og_loss))
    print('>>> AVERAGE PSNR: ', sum(psnr_loss)/len(psnr_loss))
    print('>>> AVERAGE PSNR OG: ', sum(psnr_og_loss)/len(psnr_og_loss))
    print('>>> AVERAGE OPTIM TIME: ', sum(optim_times)/len(optim_times))
    print("="*80)

    # write to file
    curr_path = os.getcwd()
    folder_name = 'logs'
    if not os.path.exists(f'{curr_path}/{folder_name}'):
        os.makedirs(folder_name)
    
    with open(f'{folder_name}/mlp-23KB-experiment.txt', 'w') as f:
        f.write(f'>>> AVERAGE RMSE: {sum(rmse_loss)/len(rmse_loss)}\n')
        f.write(f'>>> AVERAGE RMSE OG: {sum(rmse_og_loss)/len(rmse_og_loss)}\n')
        f.write(f'>>> AVERAGE PSNR: {sum(psnr_loss)/len(psnr_loss)}\n')
        f.write(f'>>> AVERAGE PSNR OG: {sum(psnr_og_loss)/len(psnr_og_loss)}\n')
        f.write(f'>>> AVERAGE OPTIM TIME: {sum(optim_times)/len(optim_times)}\n')
        f.write('===============================================\n')
        f.write(f'rmse_loss: {rmse_loss}\n')
        f.write(f'rmse_og_loss: {rmse_og_loss}\n')
        f.write(f'psnr_loss: {psnr_loss}\n')
        f.write(f'psnr_og_loss: {psnr_og_loss}\n')
        f.write(f'optim_times: {optim_times}\n')
    f.close()