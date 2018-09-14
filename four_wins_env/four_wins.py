import gym
import numpy as np
import random


ROW_COUNT = 6
COLUMN_COUNT = 7
REWARDS = [0, 100, -100]  # running, win, loose/unallowed move
RENDER_SIGNS = [" ", "x", "o"]


class FourWinsEnv(gym.Env):
    """
    Define a simple Banana environment.

    The environment defines which actions can be taken at which point and
    when the agent receives which reward.
    """

    def __init__(self, play_both=False):
        """
        Creates a four-wins environment
        Args:
            play_both(bool): if both players are played by the agent
        """
        self.__version__ = "0.1.0"
        # logging.info("BananaEnv - Version {}".format(self.__version__))
        print(f"FourWinsEnv - Version{self.__version__}")

        self.action_space = gym.spaces.Discrete(COLUMN_COUNT)
        self.field = np.full((COLUMN_COUNT, ROW_COUNT), -1, dtype=np.int)
        self.chip_count_per_column = np.zeros(COLUMN_COUNT, dtype=np.int)
        self.current_player = int(random.getrandbits(1) + 1)
        self.play_both = play_both
        self.reward_range = (min(REWARDS), max(REWARDS))

        # Observation
        field_space = gym.spaces.Box(np.full_like(self.field, -1), np.full_like(self.field, 1), dtype=np.int)
        gym.spaces.Tuple((field_space, gym.spaces.Discrete(2)))
        self.observation_space = field_space

    def render(self, mode='human'):
        field_to_print = list(reversed(np.transpose(self.field)))
        print("--" * COLUMN_COUNT + "-")
        for row in field_to_print:
            print("|" + "|".join([RENDER_SIGNS[field+1] for field in row]) + "|")
        print("--" * COLUMN_COUNT + "-")

    def step(self, action):
        """
        The agent takes an action in the environment
        Args:
            action(int): the number of the column to insert the chip
        Returns:
            (tuple(array, int)) observation: the current field after the action and the current player
            (float) reward: the reward created by the action
            (bool) done: if the game is over
            info: nothing so far
        """
        if not self.action_space.contains(action):
            raise AttributeError("Not an allowed action")
        if self.chip_count_per_column[action] == ROW_COUNT:
            return self._get_state(), REWARDS[2], False, None
        self.field[action][self.chip_count_per_column[action]] = self.current_player
        self.chip_count_per_column[action] += 1
        won = self._check_for_end(action)
        reward = REWARDS[won]
        done = won > 0
        self.current_player = (self.current_player + 1) % 2
        if not self.play_both and self.current_player == 1:
            new_obs, adv_reward, done, _ = self.step(np.random.randint(self.action_space.n))
            return new_obs, -adv_reward, done, None
        return self._get_state(), reward, done, None

    def _check_for_end(self, last_action):
        row_number = self.chip_count_per_column[last_action] - 1
        if row_number >= 3:
            current_column_last_four = self.field[last_action][row_number - 3:row_number + 1]
            if len(np.unique(current_column_last_four)) == 1:  # all elements equal
                return current_column_last_four[0]
        # check row
        same_next_to_chip = 0
        color = self.field[last_action][row_number]
        for i in range(last_action, max(last_action - 4, 0), -1):
            if self.field[i][row_number] == color:
                same_next_to_chip += 1
            else:
                break
        for i in range(min(last_action + 1, COLUMN_COUNT-1), min(last_action + 4, COLUMN_COUNT)):
            if self.field[i][row_number] == color:
                same_next_to_chip += 1
            else:
                break
        if same_next_to_chip >= 4:
            return color
        return 0

    # def _get_reward(self):
    #     """Reward is given only for winning or loosing so far."""
    #     return 0

    def reset(self):
        """
        Resets the environment
        Returns:
            (tuple(array,int)) an initial observation
        """
        self.field = np.full((COLUMN_COUNT, ROW_COUNT), -1, dtype=np.int)
        self.chip_count_per_column = np.zeros(COLUMN_COUNT, dtype=np.int)
        self.current_player = int(random.getrandbits(1))
        return self._get_state()

    def _get_state(self):
        """Get the observation."""
        return self.field, self.current_player
