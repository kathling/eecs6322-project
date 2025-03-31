# TODO: reproduce Table 1 for MLP (53KB)+enc

# To reproduce the first highlighted row in Table 1
# MLP (53KB) + enc.

import torch
import numpy as np
import cv2
import utils
import load
import time
from metrics import RMSE, PSNR
import os

from model import GamutMLP_64HF_53KB


if __name__ == "__main__":
    # flag for gpu
    device = 'cpu'
    if (torch.cuda.is_available()):
        device = torch.device('cuda')
    print(device)

    folder_path = "datasets/test/"
    selected_imgs = ['color_286.png', 
                     'color_1188.png', 
                     'color_1637.png', 
                     'color_a2288-_DGW6237.png', 
                     'color_a3810-_DGW6236.png']
    image_paths = [os.path.join(folder_path, f) for f in selected_imgs]
    # image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    # image_paths = image_paths[:5]


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
        print('img path: ', img_path)
        print('height: ', height)
        print('width: ', width)
        I_PP = img_prophoto_gamma.reshape(-1, 3).T # got this from gamut github code
        I_PP, I_ClippedPP, OG_mask = load.gamut_reduction(I_PP)
        display_OG_mask = (OG_mask).astype(np.float32) # convert to BW image, recall OG_mask is a boolean array
        display_I_PP = I_PP.T.reshape(height,width,3) # ASSUMPTION: we tried to "undo" line 112
        display_I_ClippedPP = I_ClippedPP.T.reshape(height,width,3)
        display_OG_mask = display_OG_mask.T.reshape(height,width,3)
        training_input, ground_truth, I_ClippedPP_5d_coords = utils.generatingTrainingInputAndGroundTruth(display_I_PP, display_I_ClippedPP, display_OG_mask)

        # build network
        net = GamutMLP_64HF_53KB().to(device)
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

            # print out stats every 10 its
            if iter % 1000 == 0:
            # if True:
                print(f'Iteration: {iter} | Loss: {loss.item()}')
        end_time = time.time() # optimization start time
        print("training complete")

        # Now, let's see how well GamutMLP restores the clipped PP image
        print('gamut expansion start')
        encoded_5d_coords, I_Clipped_PP_5d_coords = utils.gamut_expansion(display_I_ClippedPP)
        print('gamut expansion complete')
        print('='*70)
        print('get final residuals start')
        with torch.no_grad():
            net.eval()
            x = torch.tensor(encoded_5d_coords).float().to(device)
            pred_residuals = net(x).to('cpu')
        print('get final residuals complete')
        print('='*70)
        # Equation 3, the recovered/predicted wide-gamut PP image
        # convert to torch tensor
        print('get restored PP start')
        all_pixels_restored_values = torch.tensor(I_Clipped_PP_5d_coords[:, -3:]).float() + pred_residuals

        # Restore the image
        restored_I_PP = all_pixels_restored_values.view(height, width, 3)
        # restored_I_PP = torch.zeros((height,width,3))
        # print('EEP: ', all_pixels_restored_values.shape)
        # for x in range(height):
        #     for y in range(width):
        #         idx = x * width + y
        #         restored_I_PP[x, y, 0] = all_pixels_restored_values[idx, 0]
        #         restored_I_PP[x, y, 1] = all_pixels_restored_values[idx, 1]
        #         restored_I_PP[x, y, 2] = all_pixels_restored_values[idx, 2]
        
        restored_I_PP = restored_I_PP.detach().cpu().numpy()
        print('shape: ', restored_I_PP.shape)
        print('get residual PP complete')
        
        # calculate metrics for OG
        # convert OG (np array to tensor)
        tensor_display_OG_mask = torch.from_numpy(display_OG_mask)
        tensor_restored_I_PP = torch.from_numpy(restored_I_PP)
        tensor_display_I_PP = torch.from_numpy(display_I_PP)

        # masked_restored_I_PP = tensor_display_OG_mask * tensor_restored_I_PP
        # masked_I_PP = tensor_display_OG_mask * tensor_display_I_PP

        # tensor([og_mask]==1)
        OG_pixels_restored_I_PP = tensor_restored_I_PP[tensor_display_OG_mask == 1]
        OG_pixels_display_I_PP = tensor_display_I_PP[tensor_display_OG_mask == 1]
        print('NUM 1 in OG mask: ', torch.sum(tensor_display_OG_mask).item())
        print('OG_pixels_restored_I_PP', OG_pixels_restored_I_PP.shape)
        print('OG_pixels_display_I_PP', OG_pixels_display_I_PP.shape)

        rmse = RMSE(tensor_display_I_PP, tensor_restored_I_PP)
        print('RMSE: ', rmse)
        psnr = PSNR(tensor_display_I_PP, tensor_restored_I_PP)
        print('PSNR: ', psnr)

        rmse_og = RMSE(OG_pixels_display_I_PP, OG_pixels_restored_I_PP)
        print('RMSE for OG: ', rmse_og)
        psnr_og = PSNR(OG_pixels_display_I_PP, OG_pixels_restored_I_PP)
        print('PSNR for OG: ', psnr_og)

        optim_time = end_time - start_time
        print('optimization time: ', optim_time)

        # track these metrics
        rmse_loss.append(rmse)
        rmse_og_loss.append(rmse_og)
        psnr_loss.append(psnr)
        psnr_og_loss.append(psnr_og)
        optim_times.append(optim_time)

        if counter % 1 == 0:
            print('EXAMPLES COVERED: ', counter)

        # let's view the image now
        # cv2.imshow('Displaying Original I_PP image', display_I_PP)
        # cv2.imshow('Displaying Clipped I_PP image', display_I_ClippedPP)
        # cv2.imshow('Restored I_ClippedPP', restored_I_PP)
        # cv2.moveWindow('Restored I_ClippedPP', 100, 100) 
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # let's just save the image 
        curr_path = os.getcwd()
        folder_name = 'restored/mlp53'
        if not os.path.exists(f'{curr_path}/{folder_name}'):
            os.makedirs(folder_name)
        file_name_w_ext = os.path.basename(img_path)

        print('saved image as: ', file_name_w_ext)
        save_img_path = os.path.join(folder_name, f'RESTORED_{file_name_w_ext}')
        restored_I_PP = (restored_I_PP * 255).astype(np.uint8)
        print(restored_I_PP.shape)
        print(restored_I_PP.dtype)
        print(type(restored_I_PP))
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
    print('DONE')
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
    
    with open(f'{folder_name}/mlp-53KB-experiment.txt', 'w') as f:
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