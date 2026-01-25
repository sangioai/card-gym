
import gymnasium as gym
from typing import Optional

##################################
# Env Utilites functions
##################################

# --- Env creation fn ------------------------------------------------

def create_env_from_spaces(observation_space, action_space, vectorizion_factor=1, **args):
    # create env - vectorized if needed
    if vectorizion_factor > 1:
        env = gym.make_vec("gymnasium_env/ParametricEnv-v0", num_envs=vectorizion_factor, vectorization_mode="sync"
            , observation_space=observation_space
            , action_space=action_space
            , **args
        )
    else:
        env = gym.make("gymnasium_env/ParametricEnv-v0"
            , observation_space=observation_space
            , action_space=action_space
            , **args
        )
    return env


##################################
# Env Classes
##################################

# --- Paramentric Env ------------------------------------------------

class ParametricEnv(gym.Env):
    def __init__(self
                , observation_space: gym.Space
                , action_space: gym.Space
                , action_mapper : callable = lambda a:a
                , observation_mapper: Optional[callable] = lambda o:(o,{})
                , transition_fn : callable = lambda o,a:o
                , rewarder_fn : callable = lambda o,a,new_o:(1.0,False,False)
                ):
        super().__init__()
        # init data
        self.observation_space = observation_space
        self.action_space = action_space
        self.action_mapper = action_mapper
        self.observation_mapper = observation_mapper
        self.transition_fn = transition_fn if transition_fn else lambda o: self.observation_space.sample()
        self.rewarder_fn = rewarder_fn
        self.observation = None
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[dict]:
        # reset seed
        super().reset(seed=seed)
        self.observation_space.seed(seed=seed)
        # sample from obs space
        self.observation = self.observation_space.sample()
        # transoform observation - if needed
        self.observation, info = self.observation_mapper(self.observation)
        return self.observation, info
    
    def step(self, action):
        action = self.action_mapper(action)
        # Let's update the envirorment
        new_observation = self.transition_fn(self.observation, action)
        # Let the rewarder_fn generate the reward
        reward, truncated, terminated = self.rewarder_fn(self.observation, action, new_observation)
        # transoform observation - if needed
        self.observation, info = self.observation_mapper(new_observation)
        # add info to output
        info.update(dict(prev_action=action))
        return self.observation, reward, terminated, truncated, info

# register env to gym
gym.register(id="gymnasium_env/ParametricEnv-v0", entry_point=ParametricEnv)

##################################
# Main
##################################

if __name__ == "__main__":
    env = gym.make_vec("gymnasium_env/ParametricEnv-v0")
    print(env)