import time
import torch


def test_dqn(
    env,
    dqn,
    episodes=10,
    verbose_level=1,
    render_delay=0.8,
):
    """
    Run trained DQN in inference mode and render the Briscola game.
    No learning, no exploration.

    Args:
        env: BriscolaEnv (possibly wrapped with TransformObservation)
        dqn: trained DQN network
        episodes: number of test episodes
        verbose_level: 0 = silent, 1 = episode summary
        render_delay: seconds to wait between renders
    """

    device = next(dqn.parameters()).device
    dqn.eval()  # IMPORTANT: inference mode

    for ep in range(episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0.0
        step = 0

        if verbose_level:
            print(f"\n Episode {ep + 1}/{episodes}")

        while not (terminated or truncated):
            # --- Render current game state ---
            env.render()
            time.sleep(render_delay)

            # --- DQN inference ---
            obs_tensor = torch.tensor(
                obs, dtype=torch.float32, device=device
            ).unsqueeze(0)  # (1, obs_dim)

            with torch.no_grad():
                q_values = dqn(obs_tensor)
                action = int(torch.argmax(q_values, dim=1).item())

            # --- Environment step ---
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1

            if verbose_level > 1:
                print(
                    f"step={step} action={action} reward={reward} info={info}"
                    f"terminated={terminated}"
                )

        # --- Final render (end of episode) ---
        env.render()

        if verbose_level:
            print(
                f"Episode finished | steps={step} | total_reward={total_reward}"
            )

    dqn.train()  # restore training mode if needed
