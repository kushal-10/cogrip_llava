## Working Grid environment

import gym
from gym import spaces
import pygame
import numpy as np


class GridWorldEnv(gym.Env):
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 2}

    def __init__(self, render_mode=None, size=5, grid_info=None, agent_pos=None, target_pos=None):
        self.size = size  # The size of the square grid
        self.window_size = 500  # The size of the PyGame window
        self.grid_info = grid_info
        self.agent_pos = agent_pos
        self.target_pos = target_pos

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 5 actions, corresponding to "right", "up", "left", "down", "wait" and "grip".
        self.action_space = spaces.Discrete(6)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]), # Right
            1: np.array([0, 1]), # Down
            2: np.array([-1, 0]), # Left
            3: np.array([0, -1]), # Up
            4: np.array([0, 0]),
            5: np.array([1, 1])
        }

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
        return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        # super().reset(seed=seed)

        self._agent_location = self.agent_pos
        self._target_location = self.target_pos

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # An episode is done if the agent has reached the target
        # terminated = np.array_equal(self._agent_location, self._target_location)
        terminated = 0
        for i in range(len(self.target_pos)):
            if np.array_equal(self.agent_pos, self.target_pos[i]):
                terminated = 1
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _draw_rect(self, canvas, color, pos, pix_square_size):
        # Ensure pos is a tuple of integers
        if not isinstance(pos, (tuple, list)):
            pos = list(pos)
            
        
        pygame.draw.rect(
            canvas,
            color,
            pygame.Rect(
                pix_square_size * np.array(pos),  # Ensure pos is multiplied correctly
                (pix_square_size, pix_square_size),
            ),
        )

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = int(
                self.window_size / self.size
        )  # The size of a single grid square in pixels

        # Draw Pieces
        for piece in self.grid_info:
            for pos in piece["piece_grids"]:
                self._draw_rect(canvas, piece["piece_colour"], pos, pix_square_size)

        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 0),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

if __name__ == '__main__':
    env = GridWorldEnv(size=20)
    env.reset()
    env.render()
    env.close()