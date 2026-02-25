import argparse
import numpy as np
import gymnasium as gym
import stable_baselines3
import matplotlib.pyplot as plt
from games.set.env import SetEnv
from games.briscola.env import BriscolaEnv
from gymnasium.wrappers import TransformObservation
from stable_baselines3.common.monitor import Monitor
from training.baseline3 import CustomReplayBuffer, LossCallback
from agents.dqn import *
from memory.reply import *
from training_scripts import *
from memory.retriever import *
from training.baseline3 import *
from training.contrastive import *

# =========================================================
# Briscola
# =========================================================
def run_briscola():

    print("Running Briscola training...")

    # ---- create env
    env = BriscolaEnv(bot=True)

    # ---- load embedder
    model = load_text_embedder()

    # ---- wrap observation into embedding space
    env = TransformObservation(
        env,
        lambda obs: embed_state(
            model,
            BriscolaEnv.observation_to_string(obs)
        ),
        env.observation_space
    )

    # ---- create dqn network
    obs_space = gym.spaces.Box(
        -np.inf, np.inf,
        (DEFAULT_EMBEDDER_DIM,),
        np.float32
    )

    dqn = create_dqn_from_env(
        env,
        observation_space=obs_space,
        hidden_dim=128,
        n_layers=5
    )

    # ---- train
    train_dqn(
        env,
        dqn,
        verbose_level=1,
        batch_size=16,
        episodes=300
    )


# =========================================================
# Set
# =========================================================
def run_set():

    print("Running Set training...")

    # ---- Create environment ----
    _env = SetEnv(force_sets=True)
    env = gym.wrappers.TimeLimit(_env, max_episode_steps=1)
    env = Monitor(env)

    # ---- Create model ----
    model = stable_baselines3.DQN(
        "MlpPolicy",
        env,
        verbose=0,
        buffer_size=1_000_000,
        replay_buffer_class=CustomReplayBuffer,
        replay_buffer_kwargs=dict(env=_env)
    )

    # ---- Train ----
    loss_callback = LossCallback()
    model.learn(
        total_timesteps=100000,
        callback=loss_callback,
        progress_bar=True
    )

    print("Sample custom info:", model.replay_buffer.custom_info[:5])

    # ---- Plot rewards ----
    episode_rewards = env.get_episode_rewards()

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, alpha=0.4)
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid()

    # ---- Plot loss ----
    def moving_average(data, window=50):
        return np.convolve(data, np.ones(window)/window, mode='valid')

    smoothed_loss = moving_average(loss_callback.losses, 50)

    plt.subplot(1, 2, 2)
    plt.plot(loss_callback.losses, alpha=0.3)
    plt.plot(smoothed_loss)
    plt.title("DQN Training Loss")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.grid()

    plt.tight_layout()
    plt.show()


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Train RL agents on Briscola or Set."
    )

    parser.add_argument(
        "--game",
        type=str,
        required=True,
        choices=["briscola", "set"],
        help="Select which game to train."
    )

    args = parser.parse_args()

    if args.game == "briscola":
        run_briscola()
    elif args.game == "set":
        run_set()


if __name__ == "__main__":
    main()