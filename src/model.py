'''
Main file.
Re-create the GamutMLP models.
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
        The 5D coordinate and color input (x,y,R,G,B) is encoded as a 120D-feature 
        vector before passing it to the MLP.

        This model's corresponding model size is 23KB (as specified in the paper).
        '''
        self.fc1 = nn.Linear(120, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 3)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class GamutMLP_64HF_53KB(nn.Module):
    def __init__(self):
        super(GamutMLP_64HF_53KB, self).__init__()
        '''
        This model's corresponding model size is 53KB (as specified in the paper).
        The hidden features are changed from 32 to size 64.
        '''
        # define the layers
        self.fc1 = nn.Linear(120, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 3)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class GamutMLP_16HF_11KB(nn.Module):
    def __init__(self):
        super(GamutMLP_16HF_11KB, self).__init__()
        '''
        This model's corresponding model size is 11KB (as specified in the paper).
        The hidden features are changed from 32 to size 16.
        '''
        # define the layers
        self.fc1 = nn.Linear(120, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 3)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class GamutMLP_128HF_137KB(nn.Module):
    def __init__(self):
        super(GamutMLP_128HF_137KB, self).__init__()
        '''
        This model's corresponding model size is 137KB (as specified in the paper).
        The hidden features are changed from 32 to size 128.
        '''
        # define the layers
        self.fc1 = nn.Linear(120, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 3)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x