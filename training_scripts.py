import random
import torch
import torch.nn as nn
import torch.optim as optim
from memory.reply import *


##################################
# Training Utilities
##################################

# --- DQN Training ------------------------------------------------

def train_dqn(env, q_net, episodes = 300, criterion=nn.MSELoss, optimizer=optim.Adam, buffer=RandomReplayBuffer(), lr_scheduler=None, writer=None, verbose_level=0, **args):
    # trainin hyperparameters
    gamma = args.get("gamma", 0.99)
    batch_size = args.get("batch_size", 16)
    epsilon = args.get("epsilon", 1.0)
    epsilon_decay = args.get("epsilon_decay", 0.995)
    epsilon_min = args.get("epsilon_min", 0.05)
    lr = args.get("lr", 1e-4)
    # create optimizer
    optimizer = optimizer(q_net.parameters(), lr=lr)

    for episode in range(episodes):
        # reset episode
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            # ε-greedy action
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                # retrive best action
                with torch.no_grad():
                    state_t = torch.FloatTensor(state)
                    action = torch.argmax(q_net(state_t)).item()

            # perform best action on env -> observe new state
            next_state, reward, done, truncated, info = env.step(action)
            # save to reply buffer
            buffer.push(state, action, reward, next_state, done, info)
            # update state and values
            state = next_state
            total_reward += reward

            if verbose_level>1:
                print(f"Env next state:",next_state)
                print(f"Env reward:",reward)
                print(f"Env done:",done)
                print(f"Env truncated:",truncated)
                print(f"Env info:",info)

            # Learn - after reaching batch size
            if len(buffer) >= batch_size:
                # sample from reply buffer
                s, a, r, s2, d, i = buffer.sample(batch_size)
                # casting
                s = torch.FloatTensor(s)
                a = torch.LongTensor(a)
                r = torch.FloatTensor(r)
                s2 = torch.FloatTensor(s2)
                d = torch.FloatTensor(d)
                if verbose_level>1:
                    print("s",s)
                    print("a",a)
                    print("r",r)
                    print("s2",s2)
                    print("d",d)
                    print("s",s)
                    print("s",i)
                # REPLAY: re-run q-network on reply state + gather Q-value of previously chosen actions
                q_values = q_net(s).gather(1, a.unsqueeze(1)).squeeze()
                # run q-network on new_state - get max Q-value
                next_q = q_net(s2).max(1)[0]
                if verbose_level>1:
                    print("next_q",next_q)
                # compute target reward
                target = r + gamma * next_q * (1 - d)
                if verbose_level>1:
                    print("target",target)
                # compte loss - avoid gradient flow on bootstrapped target
                loss = criterion()(q_values, target.detach())
                if verbose_level>1:
                    print("loss",loss)
                # run backprop
                if verbose_level>1:
                    print("backporpr",loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if verbose_level>1:
                    print("back done",loss)

            if done or truncated:
                break
        # epsilon decay
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        if verbose_level>0: print(f"Episode {episode}, Reward: {total_reward}")
    return total_reward
