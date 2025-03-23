import torch
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('agg')

from matplotlib import pyplot as plt
import numpy as np
import cv2

from model import GamutMLP
import utils
import load


# # ==================================================================
# # Copy your compute accuracy function from Part1 here
# # I want you to use that because it uses Numpy, we will learn to convert torch->numpy tensors
# # IMPORTANT: Numpy only, I DON'T WANT TO SEE PYTORCH HERE
# # ==================================================================
# def compute_accuracy(prediction,gt_logits):
#     # acc = ?????
#     pred_labels = np.argmax(prediction, axis=1)
    
#     # slight change from Part1-skeleton; didn't need to squeeze gt_logits because already same dimensions as pred_labels
#     compare_result = np.equal(pred_labels, gt_logits)
    
#     numCorrect = compare_result.sum()
    
#     acc = numCorrect/np.size(prediction, 0)
#     return acc

# load in one image
img_prophoto_gamma = cv2.imread('datasets/train/standard_r9e70c57ft.png')
# cv2.imshow('ORIGINAL', img_prophoto_gamma) 
I_PP = img_prophoto_gamma.reshape(-1, 3).T # got this from gamut github code
I_PP, I_ClippedPP, OG_mask = load.gamut_reduction(I_PP)
display_OG_mask = (OG_mask).astype(np.float32) # convert to BW image, recall OG_mask is a boolean array
display_I_PP = I_PP.T.reshape(512,512,3) # ASSUMPTION: we tried to "undo" line 112
display_I_ClippedPP = I_ClippedPP.T.reshape(512,512,3)
display_OG_mask = display_OG_mask.T.reshape(512,512,3) # A white pixel represents OG pixel (value = 1/True), black represents IG (value = 0/False)

print('I_PP_Clipped', I_ClippedPP)
print('shape of I_PP_Clipped', I_ClippedPP.shape)

# training_input: the 120D-feature vector mapped(x, y, R', G', B') values
# ground_truth: the original PP values (x, y, R, G, B)
# I_ClippedPP_5d_coords: the clipped PP values (x, y, R', G', B')
training_input, ground_truth, I_ClippedPP_5d_coords = utils.generatingTrainingInputAndGroundTruth(display_I_PP, display_I_ClippedPP, display_OG_mask)

# print("="*50)
# print(training_input)
# print('>>> shape:', training_input.shape)
# print("="*50)
# print(ground_truth)
# print('>>> shape:', ground_truth.shape)
# print("="*50)
# print(I_ClippedPP_5d_coords)
# print('>>> shape:', I_ClippedPP_5d_coords.shape)
# build network
net = GamutMLP()

# L2 (MSE) Loss Function described in Eq. 5
# https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
loss_fn =  torch.nn.MSELoss()

# build optimizer, and set learning rate
# https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
learning_rate = 0.001
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

if __name__ == '__main__':
    # ==================================================================
    # Train loop
    # ==================================================================
#     train_losses = [] # this stuff here is to track training statistics
#     train_accuracies = []
#     val_iterations = []
#     val_losses = []
#     val_accuracies = []
    
    for iter in range(9000): # num of optimization iterations
        net.train() # good practice, some networks have training dependent behaviour
        optimizer.zero_grad()
        x = torch.tensor(training_input).float()
        pred_residuals = net(x)
        
        # Equation 3, the recovered/predicted wide-gamut PP image
        # convert to torch tensor
        sampled_pixels_restored_values = torch.tensor(I_ClippedPP_5d_coords[:, -3:]).float() + pred_residuals
        # print('sampled_pixels_restored_values', sampled_pixels_restored_values)
        # print('sampled_pixels_restored_values shape', sampled_pixels_restored_values.shape)

        '''
        # saved to restore the img
        I_PP_hat = torch.clone(torch.tensor(display_I_ClippedPP))
        for i in range(I_ClippedPP_5d_coords.shape[0]):
            row = I_ClippedPP_5d_coords[i]
            x = row[0].astype(int)
            y = row[1].astype(int)
            
            # populate the predicted img with restored RGB values
            I_PP_hat[x, y, 0] = sampled_pixels_restored_values[i, 0]
            I_PP_hat[x, y, 1] = sampled_pixels_restored_values[i, 1]
            I_PP_hat[x, y, 2] = sampled_pixels_restored_values[i, 2]
        np_I_PP_hat = I_PP_hat.detach().cpu().numpy()
        '''
        # print('type np_I_PP_hat', type(np_I_PP_hat))
        # cv2.imshow('Displaying I_ClippedPP image using OpenCV', np_I_PP_hat) 
        # cv2.waitKey(0)

        # Equation 5, use L2 loss
        # convert to torch tensor
        # ground_truth = torch.tensor(ground_truth).float()
        # print('I_PP_hat', I_PP_hat)
        # print('I_PP_hat shape', I_PP_hat.shape)
        # print('ground_truth', ground_truth)
        # print('ground_truth shape', ground_truth.shape)
        ground_truth_RGB = torch.tensor(ground_truth[:, -3:]).float()
        loss = loss_fn(sampled_pixels_restored_values, ground_truth_RGB)

        loss.backward()
        optimizer.step()

        # print out stats every 10 its
        if iter % 10 == 0:
            print(f'Iteration: {iter} | Loss: {loss.item()}')
        #     # print(f'Iteration: {iter} | Loss: {loss.item()} | Accuracy: {accuracy}')

        # for i in range(training_input.shape[0]):
        #     # get the training input for the i'th pixel sampled
        #     x = training_input[i]
        #     x = torch.tensor(x).float()
            
        #     pred = net(x)
        #     print('pred residuals', pred)
        #     break

    print('Training complete')
    # Now, let's see how well GamutMLP restores the clipped PP image
    encoded_5d_coords, I_Clipped_PP_5d_coords = utils.gamut_expansion(display_I_ClippedPP)
    with torch.no_grad():
        net.eval()
        x = torch.tensor(encoded_5d_coords).float()
        pred_residuals = net(x)

    # Equation 3, the recovered/predicted wide-gamut PP image
    # convert to torch tensor
    all_pixels_restored_values = torch.tensor(I_Clipped_PP_5d_coords[:, -3:]).float() + pred_residuals

    # Restore the image
    restored_I_PP = torch.zeros((512, 512, 3))
    for x in range(512):
        for y in range(512):
            idx = x * 512 + y
            restored_I_PP[x, y, 0] = all_pixels_restored_values[idx, 0]
            restored_I_PP[x, y, 1] = all_pixels_restored_values[idx, 1]
            restored_I_PP[x, y, 2] = all_pixels_restored_values[idx, 2]
    
    restored_I_PP = restored_I_PP.detach().cpu().numpy()
    cv2.imshow('Displaying Original I_PP image', display_I_PP)
    cv2.imshow('Displaying Clipped I_PP image', display_I_ClippedPP)
    cv2.imshow('Displaying Restored I_ClippedPP image', restored_I_PP) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#         for im, label in dataloader_train: # the dataloader is an iterator, will shuffle, then go through all data once,
#             # Center data and pad to 32x32, for convienence
#             im = im - 0.5
#             im = torch.nn.functional.pad(im,[2,2,2,2,0,0,0,0])

#             # Zero out the gradients, just like in Part 1
#             # take a look at: https://pytorch.org/docs/stable/optim.html
#             # ?????
#             optimizer.zero_grad()

#             # get prediction from network, calling the object like a function invokes its forward() method
#             pred = net(im)

#             # compute the loss
#             loss = loss_fn(pred, label) # ?????

#             # compute the accuracy, we will have to convert to numpy tensorfs first
#             # .cpu() moves the tensor to the device, here it is pointless because it is already cpu, but must be done when we use gpus
#             # .detach() tells torch that gradients will not flow through this operation, MUST be done before we get the numpy array
#             # .numpy() gets the numpy array
#             accuracy = compute_accuracy(pred.cpu().detach().numpy(),label.cpu().detach().numpy())

#             # compute gradient in respect to the loss. Take a look here: https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
#             # ?????
#             loss.backward()

#             # take a step of SGD, call this through the optimizer!
#             # take a look here: https://pytorch.org/docs/stable/optim.html
#             # ?????
#             optimizer.step()

#             # store stats for this iteration
#             # note, if the tensor is a scalar, .item() retrieves the value as a python primative
#             train_losses.append(loss.item())
#             train_accuracies.append(accuracy)

#             # print out stats every 10 its
#             if global_it % 10 == 0:
#                 print(f'Iteration: {global_it} | Loss: {loss.item()} | Accuracy: {accuracy}')

#             # increment
#             global_it += 1

#         print('Validating...')
#         net.eval() # good practice, some networks have evaluation dependent behaviour
#         cur_val_losses = []
#         cur_val_accuracies = []
#         for im, label in dataloader_val:
#             # we don't need gradients so we wrap with this to not compute them
#             # NOTE: much of this section loois like the training section!
#             with torch.no_grad():
#                 # Center data and pad to 32x32, for convienence
#                 # ?????
#                 im = im - 0.5
#                 im = torch.nn.functional.pad(im,[2,2,2,2,0,0,0,0])

#                 # get prediction from network
#                 # ?????
#                 pred = net(im)

#                 # compute loss
#                 # ?????
#                 loss = loss_fn(pred, label)

#                 # compute accuracy
#                 # ?????
#                 accuracy = compute_accuracy(pred.cpu().detach().numpy(),label.cpu().detach().numpy())

#                 # collect stats for this iteration
#                 cur_val_losses.append(loss.item())
#                 cur_val_accuracies.append(accuracy.item())
#         # take mean over entire validation set
#         avg_val_loss = np.array(cur_val_losses).mean()
#         avg_val_accuracies = np.array(cur_val_accuracies).mean()
#         val_iterations.append(global_it)
#         val_losses.append(avg_val_loss)
#         val_accuracies.append(avg_val_accuracies)
#         print(f'Average validation loss: {avg_val_loss}')
#         print('Saving checkpoint')

#         # save parameters of network
#         # Note other things can have states (certain optimizers)
#         # you would need to save these things to resume training
#         # but we're not doing that here
#         torch.save(net.state_dict(),f'epoch-{epoch:04d}.pth')

#     print('Training complete')
#     print('Plotting training stats')

#     plt.plot(train_losses,label='Training')
#     plt.plot(val_iterations,val_losses,label='Validation')
#     plt.xlabel('Iteration')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.savefig('training-loss.png')
#     plt.clf()

#     plt.plot(train_accuracies,label='Training')
#     plt.plot(val_iterations,val_accuracies,label='Validation')
#     plt.xlabel('Iteration')
#     plt.ylabel('Accuracy')
#     plt.legend()
#     plt.savefig('training-acc.png')