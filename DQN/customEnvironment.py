
import gymnasium as gym


class PacmanRewardWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        death_penalty=-10,
        time_penalty=-0.01,
        max_deaths=3
    ):
        super().__init__(env)

        self.death_penalty = death_penalty
        self.time_penalty = time_penalty
        self.max_deaths = max_deaths

        self.previous_lives = 0
        self.deaths = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.previous_lives = self.env.unwrapped.ale.lives()
        self.deaths = 0

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Small penalty for every step
        reward += self.time_penalty

        # Check whether Pac-Man has lost a life
        current_lives = self.env.unwrapped.ale.lives()

        if current_lives < self.previous_lives:
            reward += self.death_penalty
            self.deaths += self.previous_lives - current_lives

        self.previous_lives = current_lives

        if self.deaths >= self.max_deaths:
            terminated = True

        info["lives"] = current_lives
        info["deaths"] = self.deaths

        return obs, reward, terminated, truncated, info
