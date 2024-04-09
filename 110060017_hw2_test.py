import os
import math
from pathlib import Path
import numpy as np
from collections import deque

import cv2

import gym
import torch
import torch.nn as nn
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT


class GRAY_SCALE_OBSERVATION:
    def __init__(self):
        pass

    def forward(self, observation):
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        return observation


class CROP_OBSERVATION:
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def forward(self, observation):
        observation = observation[self.y1 : self.y2, self.x1 : self.x2]
        return observation


class RESIZE_OBSERVATION:
    def __init__(self, shape):
        self.shape = shape

    def forward(self, observation):
        observation = cv2.resize(observation, self.shape, interpolation=cv2.INTER_AREA)
        return observation


class BLUR_OBSERVATION:
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def forward(self, observation):
        observation = cv2.GaussianBlur(observation, (self.kernel_size, self.kernel_size), 0)
        return observation


class FRAME_STACKING:
    def __init__(self, stack_size):
        self.stack_size = stack_size
        self.stack = deque(maxlen=stack_size)

    def forward(self, observation):
        self.stack.append(observation)
        if len(self.stack) < self.stack_size:
            while len(self.stack) < self.stack_size:
                self.stack.append(observation)
        return np.stack(self.stack, axis=0)


class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super(NoisyLinear, self).__init__(in_features, out_features)

        self.sigma = 0.5

        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features).fill_(self.sigma / math.sqrt(self.in_features)))
        self.register_buffer("weight_epsilon", torch.zeros(out_features, in_features))

        self.bias_sigma = nn.Parameter(torch.Tensor(out_features).fill_(self.sigma / math.sqrt(self.in_features)))
        self.register_buffer("bias_epsilon", torch.zeros(out_features))

    def forward(self, x):
        # return nn.functional.linear(x, self.weight + self.weight_sigma * self.weight_epsilon, self.bias + self.bias_sigma * self.bias_epsilon)
        return nn.functional.linear(x, self.weight, self.bias)


class NETWORK(nn.Module):
    def __init__(self, action_space=SIMPLE_MOVEMENT):
        super(NETWORK, self).__init__()

        # State shape: 4x128x128 (4 frames of 128x128 pixels)
        # Action space: 5 for SIMPLE_MOVEMENT, 7 for RIGHT_ONLY, 12 for COMPLEX_MOVEMENT

        self.feature_map = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=5, stride=2, padding=0),  # 4x128x128 -> 16x62x62
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=0),  # 16x62x62 -> 32x29x29
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=0),  # 32x29x29 -> 32x13x13
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=40, kernel_size=5, stride=2, padding=0),  # 32x13x13 -> 40x5x5
            nn.LeakyReLU(),
            nn.Flatten(),  # 40x5x5 -> 1000
        )

        self.advantage = nn.Sequential(
            NoisyLinear(1000, 350),
            nn.LeakyReLU(),
            NoisyLinear(350, len(action_space)),
        )

        self.value = nn.Sequential(
            NoisyLinear(1000, 350),
            nn.LeakyReLU(),
            NoisyLinear(350, 1),
        )

    def forward(self, x):
        # Transform the range of x from [0, 255] to [0, 1]
        x = x / 255.0

        x = self.feature_map(x)

        advantage = self.advantage(x)
        value = self.value(x)

        q_value = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_value


class Agent:
    def __init__(self):
        self.config = {
            "frame_skipping": 4,
            "epsilon": 0.02,
            "action_space": SIMPLE_MOVEMENT,
        }

        self.network = NETWORK(self.config["action_space"]).float()
        self.action_space_len = len(self.config["action_space"])

        self.last_action = 0
        self.current_frame = 0

        self.RESIZE_OBSERVATION_ = RESIZE_OBSERVATION(shape=(128, 128))
        self.GRAY_SCALE_OBSERVATION_ = GRAY_SCALE_OBSERVATION()
        self.FRAME_STACKING_ = FRAME_STACKING(stack_size=4)
        self.CROP_OBSERVATION_ = CROP_OBSERVATION(x1=10, y1=35, x2=230, y2=225)
        self.BLUR_OBSERVATION_ = BLUR_OBSERVATION(kernel_size=5)

        self.LoadModel()

    def ProcessObservation(self, observation):
        observation = observation.astype(np.uint8)
        observation = self.CROP_OBSERVATION_.forward(observation)
        observation = self.BLUR_OBSERVATION_.forward(observation)
        observation = self.RESIZE_OBSERVATION_.forward(observation)
        observation = self.GRAY_SCALE_OBSERVATION_.forward(observation)
        observation = self.FRAME_STACKING_.forward(observation)
        return observation

    def GreedyPolicy(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array(state), dtype=torch.int8).unsqueeze(0)
            action = torch.argmax(self.network(state)).item()
        return action

    def RandomPolicy(self, state=None):
        return np.random.randint(self.action_space_len)

    def EpsilonGreedyPolicy(self, state):
        # Exploration
        if np.random.rand() < self.config["epsilon"]:
            return self.RandomPolicy()
        # Exploitation
        else:
            return self.GreedyPolicy(state)

    def act(self, observation):
        if self.current_frame == 0:
            observation = self.ProcessObservation(observation)
            self.last_action = self.GreedyPolicy(observation)
        self.current_frame = (self.current_frame + 1) % self.config["frame_skipping"]
        return self.last_action

    def SaveModel(self):
        # Save model parameters and config
        current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        data_path = current_dir / "110060017_hw2_data.py"

        torch.save({"model": self.network.state_dict()}, data_path)

    def LoadModel(self):
        current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        data_path = current_dir / "110060017_hw2_data.py"

        data = torch.load(data_path)
        self.network.load_state_dict(data["model"])
