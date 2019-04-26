import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Model for DQN and Double DQN"""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        h1, h2, h3 = 64, 64, 16
        
        self.fc1 = nn.Linear(state_size, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
class DuelingQNetwork(nn.Module):
    """Model for Dueling DQN"""
    
    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.action_size = action_size
        
        h1, h2, h3 = 32, 64, 64
        hv, ha = 128, 128
        
        self.fc1 = nn.Linear(state_size, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        
        
        self.fc_v1 = nn.Linear(h3, hv)
        self.fc_a1 = nn.Linear(h3, ha)
        
        self.fc_v2 = nn.Linear(hv, 1)
        self.fc_a2 = nn.Linear(ha, action_size)
    
    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        v = F.relu(self.fc_v1(x))
        v = F.relu(self.fc_v2(v))
        
        a = F.relu(self.fc_a1(x))
        a = F.relu(self.fc_a2(a))
        
        x = v + a - a.mean(1).unsqueeze(1).expand(x.size(0), self.action_size)
        
        return x
