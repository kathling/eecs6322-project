'''
Handles pretraining the GamutMLP model on the train dataset.
This is done for GamutMLP (23KB) and GamutMLP_53KB (53KB) to reproduce Table 1.
Pretrained weights are saved in the 'weights' folder.

Run this file to get the pretrained weights.

It takes around 4 hours to pretrain the model.
'''

import torch
import os
import cv2
import time
from evaluate import restore_img
from model import GamutMLP, GamutMLP_64HF_53KB
import utils
from train import train

def preprocess_images(image_paths):
    '''
    Preprocess the images for training to avoid applying gamut reduction in every epoch.
    We save I_PP, I_ClippedPP, and OG_mask for each image in the dataset.
    Saves the preprocess images in a separate folder 'dataset/processed/' for faster training.
    Input:
        image_paths: list of image paths to preprocess
    '''
    print('Preprocessing images...')
    preprocessed_imgs = []
    for img_path in image_paths:
        img_prophoto_gamma = cv2.imread(img_path)
        height, width = img_prophoto_gamma.shape[0], img_prophoto_gamma.shape[1]

        I_PP = img_prophoto_gamma.reshape(-1, 3).T
        I_PP = torch.tensor(I_PP, device=device)        # move to GPU 
        I_PP, I_ClippedPP, OG_mask = utils.gamut_reduction(I_PP, device)

        preprocessed_imgs.append((I_PP.cpu(), I_ClippedPP.cpu(), OG_mask.cpu(), height, width))
    
    # Save preprocessed images to file
    if not os.path.exists('datasets/processed/'):
        os.makedirs('datasets/processed/')
    torch.save(preprocessed_imgs, 'datasets/processed/preprocessed_imgs.pth')

    torch.cuda.empty_cache()  # clear GPU memory


def pretrain(model_name, image_paths, checkpoint_path, META_EPOCH=2):
    '''
    Pretrain the GamutMLP model on the train dataset.
    Basically just a wrapper for the train function in train.py.

    Input:
        model_name: name of the model to pretrain (GamutMLP or GamutMLP_53KB)
        image_paths: list of image paths to preprocess
        checkpoint_path: path to save the pretrained weights
        META_EPOCH: number of epochs to pretrain the model (we set the default to 2)
    '''
    # Load the model to pretrain
    if model_name == 'GamutMLP':
        model = GamutMLP().to(device)
    elif model_name == 'GamutMLP_53KB':
        model = GamutMLP_64HF_53KB().to(device)
    else:
        raise ValueError("Invalid pre-train GamutMLP model name. Choose from [GamutMLP, GamutMLP_53KB]")
    
    # meta-GamutMLP pre-training hyperparameters specified in the paper (see Page 4, Section 3.2)
    loss_fn =  torch.nn.MSELoss()
    learning_rate = 0.01                                            
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    iterations = 10000

    # Preload the images if they are available, else preprocess them
    if not os.path.exists('datasets/processed/preprocessed_imgs.pth'):
        preprocess_images(image_paths)
    preprocessed_imgs = torch.load('datasets/processed/preprocessed_imgs.pth',
                                    map_location='cpu')
    
    # Save the model weights
    curr_path = os.getcwd()
    folder_name = 'pretrained_weights'
    if not os.path.exists(f'{curr_path}/{folder_name}'):
        os.makedirs(folder_name)

    print('Starting pretraining...')
    # Pretrain the model
    model.train()  # set model to training mode
    start_time = time.time()    # pretraining start time
    for epoch in range(META_EPOCH):
        counter = 0
        for I_PP, I_ClippedPP, OG_mask, height, width in preprocessed_imgs:
            # move images to GPU
            I_PP = I_PP.to(device)
            I_ClippedPP = I_ClippedPP.to(device)
            OG_mask = OG_mask.to(device)

            display_I_PP = I_PP.T.reshape(height,width,3)
            display_I_ClippedPP = I_ClippedPP.T.reshape(height,width,3)
            display_OG_mask = OG_mask.float().T.reshape(height,width,3)

            # check if these are also on GPU
            # print(f'display_I_PP device: {display_I_PP.device}')
            # print(f'display_I_ClippedPP device: {display_I_ClippedPP.device}')
            # print(f'display_OG_mask device: {display_OG_mask.device}')
            training_input, ground_truth, I_ClippedPP_5d_coords = utils.generate_normalized_training_data(display_I_PP, display_I_ClippedPP, display_OG_mask)
            train(model, training_input, I_ClippedPP_5d_coords, ground_truth, optimizer, loss_fn, iterations, device)

            counter += 1
            # save checkpoint every 100 images
            if counter % 100 == 0:
                print(f'Epoch: {epoch}, Image: {counter}/{len(image_paths)}')
                torch.save(model.state_dict(), checkpoint_path)

    end_time = time.time()      # pretraining end time
    total_train_time = end_time - start_time
    print(f'Pretraining time: {total_train_time} seconds')

    # Save time it took to pretrain model into a log file
    folder_name = 'logs'
    if not os.path.exists(f'{curr_path}/{folder_name}'):
        os.makedirs(folder_name)
    with open(f'{folder_name}/meta-mlp-pretrain-23KB.txt', 'w') as f:
        f.write(f'Total training time to pre-train GamutMLP (23KB): {total_train_time}\n')
    f.close()

if __name__ == '__main__':
    device = 'cpu'
    if (torch.cuda.is_available()):
        device = torch.device('cuda')

    # Load the train dataset
    folder_path = "datasets/train/"
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # This is what you need to modify to change the model type
    model_name = 'GamutMLP'
    checkpoint_path = 'pretrained_weights/metamlp_23KB.pth'
    META_EPOCH=1

    pretrain(model_name, image_paths, checkpoint_path, META_EPOCH)
    print('Pretraining completed.')