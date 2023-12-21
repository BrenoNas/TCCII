import numpy as np
import airsim

import gymnasium as gym
from gymnasium import spaces


class AirSimEnv(gym.Env):
    

    def __init__(self, image_shape):
        self.observation_space = spaces.Box(0, 255, shape=image_shape, dtype=np.uint8)
        self.viewer = None

    def __del__(self):
        raise NotImplementedError()

    def getObservation(self):
        raise NotImplementedError()

    def computeReward(self):
        raise NotImplementedError()

    def step(self, action):
        raise NotImplementedError()

    def render(self):
        return self._get_obs()
