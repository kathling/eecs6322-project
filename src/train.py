'''
Handles training a GamutMLP model.
'''

def train(GamutMLP, train_input, I_ClippedPP_5d_coords, ground_truth, 
          optimizer, loss_fn, iterations, device):
    '''
    Input:
        GamutMLP: GamutMLP model
        train_input: training input (120D coordinates for each pixel)
        I_ClippedPP_5d_coords: 5D coordinates of the clipped PP image
        ground_truth: ground truth of original PP image (5D coordinates of the PP image)
        optimizer: Adam (standard optimization), SGD (faster optimization, pre-trained)
        loss_fn: L2 (standard optimization)
        iterations: 9000 (standard optimization), 1200 (pre-trained)
        device: 'cuda' (GPU), 'cpu' (CPU)
    '''

    GamutMLP.train() # good practice, some networks have training dependent behaviour
    for iter in range(iterations): # num of optimization iterations
        optimizer.zero_grad()
        
        pred_residuals = GamutMLP(train_input)
        
        # Equation 3, the recovered/predicted wide-gamut PP image
        restored_pixel_values = I_ClippedPP_5d_coords[:, -3:] + pred_residuals

        # Equation 5, loss function
        ground_truth_RGB = ground_truth[:, -3:].detach().float().to(device)
        loss = loss_fn(restored_pixel_values, ground_truth_RGB)

        loss.backward()
        optimizer.step()

        # # print out stats every 10 its
        # if iter % 1000 == 0:
        #     print(f'Iteration: {iter} | Loss: {loss.item()}')