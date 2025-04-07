'''
This file handles actually running the experiments.
Please modify the parameters in experiment_config.py to run different experiments.

The avg RMSE, PSNR, and optimization time metrics are saved in the 'logs' folder.
'''

import os
import torch
import cv2
import time
import numpy as np

from model import GamutMLP, GamutMLP_64HF_53KB, GamutMLP_16HF_11KB, GamutMLP_128HF_137KB
from experiment_config import config
import utils
from train import train
from evaluate import restore_img, calc_metrics

def report_metrics(rmse_loss, rmse_og_loss, psnr_loss, psnr_og_loss, optim_times, logs_filename):
    '''
    Saved the RMSE, PSNR, and optimization time metrics in the logs folder.
    Input:
        rmse_loss: list of RMSE loss values
        rmse_og_loss: list of RMSE OG loss values
        psnr_loss: list of PSNR loss values
        psnr_og_loss: list of PSNR OG loss values
        optim_times: list of optimization times
    '''

    # Write to file
    curr_path = os.getcwd()
    folder_name = 'logs'
    if not os.path.exists(f'{curr_path}/{folder_name}'):
        os.makedirs(folder_name)
    with open(os.path.join(folder_name, logs_filename), 'w') as f:
        if config['pretrained']:
            f.write(f'MODEL: meta-{config["model"]}\n\n')
        else:
            f.write(f'MODEL: {config["model"]}\n\n')
        f.write('===============================================\n')
        f.write(f'AVERAGE RESULTS\n')
        f.write('===============================================\n')
        f.write(f'AVG RMSE: {np.mean(rmse_loss)}\n')
        f.write(f'AVG RMSE OG: {np.mean(rmse_og_loss)}\n')
        f.write(f'AVG PSNR: {np.mean(psnr_loss)}\n')
        f.write(f'AVG PSNR OG: {np.mean(psnr_og_loss)}\n')
        f.write(f'AVG OPTIM TIME: {np.mean(optim_times)}\n\n')
        f.write('===============================================\n')
        f.write(f'PER IMG RESULTS\n')
        f.write('===============================================\n')
        f.write(f'rmse_loss: {rmse_loss}\n')
        f.write(f'rmse_og_loss: {rmse_og_loss}\n')
        f.write(f'psnr_loss: {psnr_loss}\n')
        f.write(f'psnr_og_loss: {psnr_og_loss}\n')
        f.write(f'optim_times: {optim_times}\n')

def load_from_configs():
    # load model
    model_name = config['model']
    if model_name == 'GamutMLP':
        model = GamutMLP().to(device)
    elif model_name == 'GamutMLP_53KB':
        model = GamutMLP_64HF_53KB().to(device)
    elif model_name == 'GamutMLP_11KB':
        model = GamutMLP_16HF_11KB().to(device)
    elif model_name == 'GamutMLP_137KB':
        model = GamutMLP_128HF_137KB().to(device)
    else:
        raise ValueError(f"Please select a model from ['GamutMLP', 'GamutMLP_53KB', 'GamutMLP_11KB', 'GamutMLP_137KB']")

    is_pretrained = config['pretrained']
    # load pretrained weights 
    if is_pretrained:
        if model_name == 'GamutMLP':
            model.load_state_dict(torch.load(config['pretrained_path']))
        elif model_name == 'GamutMLP_53KB':
            model.load_state_dict(torch.load(config['pretrained_path']))
        else:
            raise ValueError("Pretrained weights only available for GamutMLP(23KB) and GamutMLP(53KB)")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return model, optimizer

if __name__ == "__main__":
    # =========================================================================
    # Setup based on config
    # =========================================================================
    device_name = config['device']
    device = torch.device(device_name if torch.cuda.is_available() else 'cpu')

    # =========================================================================
    # Load the test dataset
    # =========================================================================
    folder_path = "datasets/test/"
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # The paper's configuration (see Page 4, Section 3.2)
    model_name = config['model']
    loss_fn = torch.nn.MSELoss()
    learning_rate = 0.001
    is_pretrained = config['pretrained']
    iterations = 9000 if not is_pretrained else 1200

    print(f'Running experiment for {model_name} model.')

    # Set file name for logging
    log_filename = f'{model_name}-experiment.txt'
    if is_pretrained:
        log_filename = f'meta-{log_filename}'
    # ========================================================================
    # Run the experiment
    # ========================================================================
    rmse_loss = []
    rmse_og_loss = []
    psnr_loss = []
    psnr_og_loss = []
    optim_times = []
    for img_path in image_paths:
        # Load the image
        img_prophoto_gamma = cv2.imread(img_path)
        height, width = img_prophoto_gamma.shape[0], img_prophoto_gamma.shape[1]
        I_PP = img_prophoto_gamma.reshape(-1, 3).T
        I_PP = torch.tensor(I_PP, device=device)        # move to GPU

        # Gamut reduction step
        I_PP, I_ClippedPP, OG_mask = utils.gamut_reduction(I_PP, device)
        display_OG_mask = OG_mask.float()
        display_I_PP = I_PP.T.reshape(height,width,3)
        display_I_ClippedPP = I_ClippedPP.T.reshape(height,width,3)
        display_OG_mask = display_OG_mask.T.reshape(height,width,3)

        # Get training input and ground truth (sample pixels)
        training_input, ground_truth, I_ClippedPP_5d_coords = utils.generate_normalized_training_data(display_I_PP, display_I_ClippedPP, display_OG_mask)

        # Load the settings from the config file
        model, optimizer = load_from_configs()
        optim_start_time = time.time()  # optimization start time
        train(model, training_input, I_ClippedPP_5d_coords, ground_truth, optimizer, loss_fn, iterations, device)
        optim_end_time = time.time()

        total_optim_time = optim_end_time - optim_start_time
        
        # Gamut expansion step
        with torch.no_grad():
            encoded_5d_coords, I_Clipped_PP_5d_coords = utils.gamut_expansion(display_I_ClippedPP)
            restored_I_PP = restore_img(model, 
                                encoded_5d_coords, 
                                I_Clipped_PP_5d_coords, 
                                height, 
                                width, 
                                device)
        rmse, psnr, rmse_og, psnr_og = calc_metrics(restored_I_PP, display_I_PP, display_OG_mask)
        rmse_loss.append(rmse)
        rmse_og_loss.append(rmse_og)
        psnr_loss.append(psnr)
        psnr_og_loss.append(psnr_og)
        optim_times.append(total_optim_time)

        # print('img optimization time: ', total_optim_time)

        # Update the file every 20 images
        if len(rmse_loss) % 20 == 0:
            print(f'Processed {len(rmse_loss)} images.')
            report_metrics(rmse_loss, rmse_og_loss, psnr_loss, psnr_og_loss, optim_times, log_filename)
         
        del model, training_input, I_PP, I_ClippedPP, OG_mask, display_I_PP, display_I_ClippedPP, display_OG_mask
        del encoded_5d_coords, I_Clipped_PP_5d_coords, ground_truth, restored_I_PP
        torch.cuda.empty_cache()        # clear GPU memory
    
    print("Saving final results in: ", log_filename)
    report_metrics(rmse_loss, rmse_og_loss, psnr_loss, psnr_og_loss, optim_times, log_filename)