from ..env import *
from .logic import *
from .hmi import *
import gymnasium as gym
import numpy as np


##################################
# Briscola Env Class
##################################

class BriscolaEnv(ParametricEnv):
    """
    Gym-style environment wrapping the BriscolaGame.
    Observation: Dict with each player's hand (encoded) and briscola suit
    Action: index of card to play from hand
    """

    def __init__(self, bot=False):
        # Define action space: max 3 cards in hand
        action_space = gym.spaces.Discrete(3)  # each player plays a card index 0..2

        # Observation: hand of each player (encoded as rank*4 + suit) + briscola suit
        # We'll encode empty slots as -1
        # Shape: (2 players, 3 cards), plus briscola suit as integer 0..3
        obs_space = gym.spaces.Dict({
            "scores": gym.spaces.Box(low=0, high=120, shape=(2,), dtype=np.int8),
            "hand_suit": gym.spaces.Box(low=-1, high=3, shape=(3,), dtype=np.int8),
            "hand_ranks": gym.spaces.Box(low=0, high=40, shape=(3,), dtype=np.int8),
            "trick_suit": gym.spaces.Box(low=-1, high=3, shape=(2,), dtype=np.int8),
            "trick_ranks": gym.spaces.Box(low=0, high=40, shape=(2,), dtype=np.int8),
            "briscola_suit": gym.spaces.Discrete(4),
            "briscola_rank": gym.spaces.Box(low=0, high=40, shape=(1,), dtype=np.int8),
            "deck": gym.spaces.Discrete(40)
        })

        # obs_space = gym.spaces.Dict({
        #     "scores": gym.spaces.Box(low=0, high=120, shape=(2, 1), dtype=np.int8),
        #     "hand_suit": gym.spaces.Box(low=-1, high=3, shape=(2, 3), dtype=np.int8),
        #     "hand_ranks": gym.spaces.Box(low=0, high=40, shape=(2, 3), dtype=np.int8),
        #     "trick": gym.spaces.Box(low=-1, high=40, shape=(2,), dtype=np.int8),
        #     "briscola": gym.spaces.Discrete(4),
        # })

        super().__init__(
            observation_space=obs_space,
            action_space=action_space,
            action_mapper=lambda a: int(a),
            observation_mapper = lambda o:(o,dict(raw=o,processed=BriscolaEnv.observation_to_string(o))),
            transition_fn=self.transition_fn,
            rewarder_fn=self.rewarder_fn
        )

        self.bot = bot
        self.game = None

    @staticmethod
    def observation_to_string(obs):
        _obs = dict()
        _obs.update(**obs)
        hand_suit = _obs.pop("hand_suit")
        hand_ranks = _obs.pop("hand_ranks")
        trick_suit = _obs.pop("trick_suit")
        trick_ranks = _obs.pop("trick_ranks")
        briscola_suit = _obs.pop("briscola_suit")
        briscola_rank = _obs.pop("briscola_rank")
        _obs["briscola"] = SUIT_CODE[Suit(briscola_suit)]
        _obs["scores"] = f"""{_obs["scores"][0]}-{_obs["scores"][1]}"""
        _obs["hands"] = [sorted([card_to_code(Card(int(r),Suit(s))) for s,r in zip(hand_suit, hand_ranks) if s != -1])]
        _obs["trick"] = sorted([card_to_code(Card(int(r),Suit(s))) for s,r in zip(trick_suit, trick_ranks) if s != -1])
        return DEFAULT_GAME_STRINGIFY(**_obs)
    
    
    def obs_from_game(self):
        """
        Build observation dict from current BriscolaGame state
        (player 0 perspective)
        """

        game = self.game
        # get current player - if bot, display always player 0 first
        current_player = game.current_player if not self.bot else 0
        p0 = game.players[current_player]

        # --- Scores ---
        scores = [game.players[current_player].score, game.players[1-current_player].score]
        scores = np.array(scores, dtype=np.int8)

        # --- Hand (player 0 only) ---
        hand_suit, hand_ranks = [], []
        for card in p0.hand:
            hand_suit += [card.suit.value]
            hand_ranks += [card.rank]
        hand_suit = np.array(hand_suit, dtype=np.int8)
        hand_ranks = np.array(hand_ranks, dtype=np.int8)
        
        # --- Trick (cards on table) ---
        trick_suit, trick_ranks = [], []
        for card in game.trick:
            trick_suit += [card.suit.value]
            trick_ranks += [card.rank]
        trick_suit = np.array(trick_suit, dtype=np.int8)
        trick_ranks = np.array(trick_ranks, dtype=np.int8)

        # --- Briscola ---
        briscola_suit = game.briscola_suit.value
        briscola_rank = np.array([game.briscola_card.rank], dtype=np.int8) if game.briscola_card else 2

        # --- Deck size ---
        deck = len(game.deck.cards)

        return {
            "scores": scores,
            "hand_suit": hand_suit,
            "hand_ranks": hand_ranks,
            "trick_suit": trick_suit,
            "trick_ranks": trick_ranks,
            "briscola_suit": briscola_suit,
            "briscola_rank": briscola_rank,
            "deck": deck,
        }


    # --- Observation Encoding ---------------------
    def encode_card(self, card: Card):
        if card is None:
            return -1
        # Encode as rank_index * 4 + suit_index
        return (card.rank - 1) * 4 + card.suit.value

    # --- Transition Function -----------------------
    def transition_fn(self, obs, action):
        """
        We assume current_player = 0 always for the agent
        action = card index for player 0
        We'll pick a random action for opponent
        """
        # performa action
        res = self.game.sequential_step(action)
        # random bot response - respond to trick
        if self.bot and not self.game.is_terminal() and not res["winner"]:
            a1 = random.randrange(len(self.game.players[1 - self.game.current_player].hand))
            res = self.game.sequential_step(a1)
        # random bot response - begin trick
        if self.bot and not self.game.is_terminal() and res["winner"]==1:
            a1 = random.randrange(len(self.game.players[self.game.current_player].hand))
            res = self.game.sequential_step(a1)
        # convert game state
        obs = self.obs_from_game()
        return obs  # we'll encode observation later via observation_mapper

    # --- Reward Function ---------------------------
    def rewarder_fn(self, obs, action, new_obs):
        # new_obs is the info dict returned by step
        reward = 0
        terminated = self.game.is_terminal()
        truncated = False
        point_diffs = new_obs["scores"] - obs["scores"]
        point_diffs = point_diffs if min(point_diffs) == 0 else new_obs["scores"][-1::-1] - obs["scores"]
        # reward is my_gain - opp_gain
        points = point_diffs[0] - point_diffs[1] 
        # calculate reward
        reward += points
        return float(reward), truncated, terminated

    # --- Gym Reset ---------------------------------
    def reset(self, seed=None, options=None):
        # super resets seeds
        obs, info = super().reset(seed, options)
        # init game
        self.game = BriscolaGame()
        obs = self.obs_from_game()
        # map
        obs, info = self.observation_mapper(obs)
        self.observation = obs
        return obs, info

    # --- Gym Step ----------------------------------
    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        return observation, reward, terminated, truncated, info
    

    # --- Gym Renderer ----------------------------------
    def render(self, mode='human'):
        if mode == 'human':
            rgb = render_game(self.game, trick=self.game.trick, render=True, return_rgb=True) 
        elif mode == 'rgb_array':
            rgb = render_game(self.game, trick=self.game.trick, render=False, return_rgb=True) 
        return rgb


# --- Register Gym Environment ----------------------
gym.register(id="Briscola-v0", entry_point=BriscolaEnv)


##################################
# Main
##################################

if __name__ == "__main__":
    env = BriscolaEnv(bot=False)
    obs, info = env.reset()
    # perform some steps
    print(info)
    obs, r, done, trun, info = env.step(0)
    print(info, r, obs["scores"], done, trun)
    obs, r, done, trun, info = env.step(0)
    print(info, r, done, trun)
    # render env
    import PIL
    PIL.Image.fromarray(env.render(mode="rgb_array"))