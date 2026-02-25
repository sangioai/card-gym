# card-gym
Just RL agents playing cards to play around with memory and RL algorithms.

<img width="331" height="336" alt="image" src="https://github.com/user-attachments/assets/e5a4b531-cde5-47ba-b575-ca2aab53f8f1" />


## Introduction
This repository implements customizable OpenAI-style Gym environment for card games and reinforcement learning agents.
It provides a simple interface to train and evaluate RL agents (e.g., DQN, PPO) on card game tasks.

## Usage
Clone the repo, install the requirement.txt and run the following for briscola:
  ```python
  python main.py --game briscola
  ```
  
For the Set Game
  ```python
  python main.py --game briscola
  ```

## 🗺️ Roadmap

### 🃁 Card Games

Planned environments include:

* **Macchiavelli** (Italian trick-taking card game)
* **Scopa**

---

### 🧠 Reinforcement Learning Algorithms

Currently implemented:

* **DQN**

Planned additions:

* **PPO (Proximal Policy Optimization)**
* **Actor-Critic**
* **RLHF (Reinforcement Learning from Human Feedback)**
* **Multi-agent RL** (for competitive card games)

---

### 💾 Memory

Currently implemented:

* **Replay Buffer**

Planned additions:

* **Retrieval Replay Buffer** (WIP)
* **Long-term Memory Modules (e.g. Titans)**


## 🤝 Contributing

Contributions are very welcome!
Feel free to open issues or PRs for:

* New card game environments
* Additional RL algorithms
* Performance or stability improvements
* Documentation and examples


## 📄 License

Apache2.0 License.
