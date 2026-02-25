from ..env import *
from .logic import *
from .hmi import *
import gymnasium as gym
import numpy as np

class SetEnv(ParametricEnv):

    MAX_BOARD = 12
    EMPTY_SLOT = -1

    def __init__(self, force_sets=False):

        # --- Action space: pick 3 board indices ---
        action_space = gym.spaces.Discrete(self.MAX_BOARD ** 3)

        # --- Observation space ---
        # observation_space = gym.spaces.Dict({
        #     "board": gym.spaces.Box(
        #         low=-1,
        #         high=2,
        #         shape=(self.MAX_BOARD, 4),
        #         dtype=np.int8
        #     ),
        #     "deck_size": gym.spaces.Discrete(82),
        # })
        observation_space =  gym.spaces.Box(
                low=-1,
                high=2,
                shape=(self.MAX_BOARD, 4),
                dtype=np.int8
            )

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            action_mapper=lambda a: tuple(int(x) for x in np.unravel_index(a, [SetEnv.MAX_BOARD, SetEnv.MAX_BOARD, SetEnv.MAX_BOARD])),
            observation_mapper=lambda o: (o, {}),
            transition_fn=self.transition_fn,
            rewarder_fn=self.rewarder_fn
        )

        self.force_sets = force_sets
        self.game = None

    @staticmethod
    def get_board_size(board):
        return np.count_nonzero(np.all(board != SetEnv.EMPTY_SLOT, axis=1))

    # -------------------------------------------------
    # Encoding
    # -------------------------------------------------

    def encode_card(self, card):
        return np.array([
            card.number - 1,
            card.shape.value,
            card.color.value,
            card.shading.value
        ], dtype=np.int8)

    def obs_from_game(self):
        board = np.full((self.MAX_BOARD, 4), self.EMPTY_SLOT, dtype=np.int8)

        for i, card in enumerate(self.game.board):
            board[i] = self.encode_card(card)

        return board
        # {
        #     "board": board,
        #     "deck_size": len(self.game.deck.cards),
        # }

    # -------------------------------------------------
    # Transition
    # -------------------------------------------------

    def transition_fn(self, obs, action):
        """
        Applies action to the internal SetGame.
        Returns new raw observation dict.
        """

        i, j, k = action
        board_size = len(self.game.board)

        # Only modify game if indices are valid and distinct
        if len({i, j, k}) == 3 and max(i, j, k) < board_size:
            cards = [self.game.board[i],
                     self.game.board[j],
                     self.game.board[k]]

            if self.game.is_set(cards):
                self.game.remove_set([i, j, k])

        return self.obs_from_game()

    # -------------------------------------------------
    # Reward
    # -------------------------------------------------

    def rewarder_fn(self, obs, action, new_obs):

        i, j, k = action
        board_size = SetEnv.get_board_size(obs)

        reward = 0.0
        truncated = False
        terminated = self.game.is_terminal()

        # Duplicate indices
        if len({i, j, k}) < 3:
            reward = -0.5

        # Out of range
        elif max(i, j, k) >= board_size:
            reward = -1.0

        else:
            # Compare board sizes to detect correct set
            if SetEnv.get_board_size(new_obs) != board_size:
                reward = +1.0
            else:
                reward = -0.1

        # Optional small time penalty (helps exploration)
        reward -= 0.01

        return float(reward), truncated, terminated

    # -------------------------------------------------
    # Reset
    # -------------------------------------------------

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed, options)

        self.game = SetGame(force_sets=self.force_sets)
        obs = self.obs_from_game()

        obs, info = self.observation_mapper(obs)
        self.observation = obs

        return obs, info

    # -------------------------------------------------
    # Rendering
    # -------------------------------------------------

    def render(self, mode="rgb_array"):
        return render_board(self.game.board)    

if __name__ == "__main__":
    import gymnasium as gym
    import numpy as np

    env = SetEnv(force_sets=True)

    obs, info = env.reset()
    print("Initial board size:", len(obs["board"]))
    print("Initial deck size:", obs["deck_size"])

    done = False
    step_count = 0
    total_reward = 0

    while not done:
        # Random action: 3 indices
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        total_reward += reward
        step_count += 1

        print(
            f"Step {step_count} | "
            f"Action: {action} | "
            f"Reward: {reward:.2f} | "
            f"Board size: {len(obs['board'])} | "
            f"Deck size: {obs['deck_size']}"
        )

    print("Episode finished.")
    print("Total reward:", total_reward)