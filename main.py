
if __name__ == "__main__":    
    from memory.reply import *
    from training_scripts import *
    from test_scripts import *
    from games.briscola.env import *
    from memory.retriever import *

    ####################### create env
    env = BriscolaEnv(bot=False)
    # load embedder
    model = load_text_embedder()
    # wrapper to make the observation space in emnbedding space
    from gymnasium.wrappers import TransformObservation
    env = TransformObservation(env
                            , lambda obs: embed_state(model, BriscolaEnv.observation_to_string(obs))
                            , env.observation_space)
    
    ####################### create dqn network
    from agents.dqn import *
    obs_space = gym.spaces.Box(-np.inf, np.inf, (DEFAULT_EMBEDDER_DIM,), np.float32)
    dqn = create_dqn_from_env(env, observation_space=obs_space, hidden_dim=128, n_layers=5)

    ####################### train env
    train_dqn(env, dqn, verbose_level=1, batch_size = 16, episodes=10)
    
    ####################### eval model
    dqn.eval()
    with torch.no_grad():
        test_dqn(env, dqn, episodes=1)