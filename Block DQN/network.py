# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import Dataset

class QNetwork(nn.Module):
    
    def __init__(self, obs_space, action_space):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride = 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(in_features=10368, out_features=256)
        self.out = nn.Linear(in_features=256, out_features=action_space)
        self.relu = nn.Tanh()
        
    def forward(self, x):
        conv = self.relu(self.bn1(self.conv1(x)))
        conv = self.relu(self.bn2(self.conv2(conv)))
        
        fully = conv.view(conv.shape[0], -1)
        
        fully = self.relu(self.fc1(fully))
        out = self.out(fully)
        
        return out
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return torch.FloatTensor(image)
        
class PrepareData(Dataset):
    def __init__(self, states, actions, rewards, next_state, transform = None):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.next_state = next_state
        self.transforms = transform
        
    def __len__(self):
        return len(self.actions)
    
    def __getitem__(self, idx):
        
        state = self.states[idx]
        if self.transforms:
            state = self.transforms(state)
            
        sample = {'states': state, 'actions': torch.LongTensor([self.actions[idx]]), 
                  'y_value' : torch.FloatTensor([self.rewards[idx]])}
        return sample
    
class PrepareNextStateData(Dataset):
    def __init__(self, states, transform = None):
        self.states = states
        self.transforms = transform
        
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        s = self.states[idx]
        if self.transforms:
            s = self.transforms(s)
            
        return s
            
        

#net = QNetwork((160, 160, 4), 4)
#
#val = np.random.randn(160, 160, 4)
#
#trans = ToTensor()
#img = trans(val)
#print(img.shape)
#
#print(net(torch.FloatTensor(img)))
        