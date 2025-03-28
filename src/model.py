'''
Main file.
Re-create the GamutMLP model.
'''

import torch
import torch.nn as nn

class GamutMLP(nn.Module):
    def __init__(self):
        super(GamutMLP, self).__init__()

        # define the layers
        '''
        From the paper:
        The GamutMLP has three linear layers. 
        The first two are fully connected ReLU layers with 32 output features. 
        The last layer outputs three values and has no activation function. 
        ---
        The 5D coordinate and color input (x,y,R ,G ,B )is encoded as a 120D-feature 
        vector before passing it to the MLP.

        This model's corresponding model size is 23KB (as specified in the paper).
        '''
        # self.model = nn.Sequential(
        #     nn.Linear(120, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 3)
        # )
        self.fc1 = nn.Linear(120, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 3)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

# TODO: define their custom loss function
# TODO: implement the optimization