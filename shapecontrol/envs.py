import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

class robmodelEnv(gym.Env):
    metadata = {"render_modes": None, "render_fps": 4}
    def __init__(self, render_mode=None, size=3):
        self.size = size  # The size of feature points
        self.window_size = 512  # The size of the PyGame window
        self.tubelength=10 # The length of the connected tube
        self.points=3 #number of feature points
        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        # target : position of size feature points
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, self.window_size - 1, shape=(2,self.points+2), dtype=np.float32),
                "target": spaces.Box(0, self.window_size - 1, shape=(2,self.points+2), dtype=np.float32),
            }
        )

        # We have 4 actions, angle velocities of two robots
        self.action_space = spaces.Box(-100,100,shape=(2,4),dtype=np.float32)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

        def _get_obs(self):
           return {"agent": self._agent_location, "target": self._target_location}

        def _get_info(self):
           return {
            "error": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
                )
            }
        # return a tuple of the initial observation and some auxiliary information.
           
        def reset(self, seed=None, options=None):
            # We need the following line to seed self.np_random
            super().reset(seed=seed)

            # Choose the agent's location uniformly at random
            self._agent_location = self.np_random.integers(0, self.window_size, size=2, dtype=float)
            self._agent_location=[self._agent_location,self._agent_location+ self.tubelength*self.np_random.integers(-1, 1, size=2, dtype=float)]

            # We will sample the target's location randomly until it does not coincide with the agent's location
            self._target_location = self._agent_location
            while np.array_equal(self._target_location, self._agent_location):
                self._target_location = self.np_random.integers(
                    0, self.size, size=2, dtype=int
                )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
