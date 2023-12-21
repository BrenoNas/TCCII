import gymnasium as gym
import torch as th
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):

        super().__init__(observation_space, features_dim=1)

        extractors = {}

        for key, subspace in observation_space.spaces.items():
            
            if key == "image":
                extractors[key] = nn.Sequential(
                    nn.Conv2d(4, 16, 8, stride=4), 
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.Conv2d(16, 32, 4, stride=2),
                    nn.BatchNorm2d(32), 
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 3, stride=2),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Conv2d(64, 32, 1, stride=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.Flatten()
                )
                
            elif key == "vector":
                # Run through a simple MLP
                extractors[key] = nn.Linear(subspace.shape[0], 64)
        
        self.extractors = nn.ModuleDict(extractors)

        self._features_dim = 512

        self.layer3 = nn.Sequential((nn.Linear(800 + 64, self._features_dim)),
                                    nn.ReLU(),
                                    )
    
    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        x = th.cat(encoded_tensor_list, dim=-1)
        return self.layer3(x)