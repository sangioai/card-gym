import torch.nn as nn
import gymnasium as gym
import torch.nn.functional as F

##################################
# DQN 
##################################

# --- residual class ------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        residual = x
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.relu(x + residual)

# --- Model class ------------------------------------------------

class DQN(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        **args
    ):
        """
        n_layers = number of residual blocks
        """
        super().__init__()

        # Input projection
        self.input_layer = nn.Linear(state_dim, hidden_dim)

        # Residual blocks
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim) for _ in range(n_layers)]
        )

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.input_layer(x))

        for block in self.blocks:
            x = block(x)

        return self.output_layer(x)
    
class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, **args):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        f = self.feature(x)
        v = self.value(f)
        a = self.advantage(f)
        return v + a - a.mean(dim=1, keepdim=True)

##################################
# Utility Functions
##################################

# --- DQN loading ------------------------------------------------

def create_dqn_from_env(env, observation_space=None, action_space=None, dqn_class=DQN, **args):
    # retrive spaces
    observation_space = env.observation_space if not observation_space else observation_space
    action_space = env.action_space if not action_space else action_space
    # retrive observation and action dimensions
    state_dim = observation_space.shape[0] if not isinstance(env, gym.vector.VectorEnv) else observation_space.shape[-1]
    action_dim = action_space.n if not isinstance(env, gym.vector.VectorEnv) else action_space.nvec[-1]
    # create q-model
    q_net = dqn_class(state_dim, action_dim, **args)
    return q_net