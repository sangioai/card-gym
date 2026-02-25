import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.common.callbacks import ProgressBarCallback
from torchvision import transforms
from PIL import Image
from .contrastive import *
import requests

# ---- Utiliy fn ----


# model downloader
def download_model(model_filename):
    urls = [
        "http://sangiorgi.me/ML4CV/best.pt.partaa",
        "http://sangiorgi.me/ML4CV/best.pt.partab",
        "http://sangiorgi.me/ML4CV/best.pt.partac",
    ]
    with open(model_filename, "wb") as outfile:
        for url in urls:
            print(f"Downloading {url}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()

            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    outfile.write(chunk)
    print("Done.")

# ImageNet normalization (ResNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def obs_to_tensor(obs, render_fn, device="cpu"):
    """
    Converts a SetEnv observation into a normalized 224x224 tensor suitable for ResNets.
    
    Args:
        obs: Observation from SetEnv (board array)
        render_fn: Function to render obs to RGB array, e.g., env.render
        device: "cpu" or "cuda"
        
    Returns:
        Tensor of shape (3, 224, 224), normalized
    """
    # Render observation to RGB image (H, W, C) uint8
    img_array = render_fn(obs)
    
    # Convert to PIL image
    img = Image.fromarray(img_array)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    tensor = transform(img).to(device)
    return tensor

# ---- Custom Callback to Log Loss ----
class LossCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.losses = []

    def _on_step(self) -> bool:
        # Get latest logged loss (if available)
        if "train/loss" in self.model.logger.name_to_value:
            loss = self.model.logger.name_to_value["train/loss"]
            self.losses.append(loss)
        return True

from memory.retriever import *
from stable_baselines3.common.type_aliases import (
    ReplayBufferSamples,
)
# ---- Custom Replay Buffer ----
class CustomReplayBuffer(ReplayBuffer):
    def __init__(self, *args, **kwargs):
        self.env = kwargs.pop("env", None)
        super().__init__(*args, **kwargs)
        self.custom_info = []  # Example: store extra info about transitions
        # Load model
        model_filename = f"models/set/best.pt"
        if not os.path.exists(model_filename): download_model(model_filename)
        self.model = torch.load(model_filename, weights_only=False)
        self.model.eval()
        # Build index
        self.index = build_index(256)

    @torch.no_grad
    def add(self, obs, next_obs, action, reward, done, infos):
        # obs to image
        tensor_imgs = [obs_to_tensor(o, render_fn=self.env.render, device="cuda") for o in obs]
        embeddings = self.model(torch.stack(tensor_imgs))
        # Add embeddings to index  <-- THIS WAS MISSING
        add_embeddings(self.index, embeddings.cpu().numpy())
        # Example: store a custom metric per transition
        self.custom_info.append({"reward": reward, "done": done})
        return super().add(obs, next_obs, action, reward, done, infos)

    @torch.no_grad
    def sample(self, batch_size: int, env=None) -> RolloutBufferSamples:
        # You can customize sampling here if needed
        batch = super().sample(batch_size, env)
        # For example, add custom info to batch if desired
        # obs to image
        tensor_imgs = [obs_to_tensor(o, render_fn=self.env.render, device="cuda") for o in self.env.observation]
        embeddings = self.model(torch.stack(tensor_imgs))
        # perform search
        res = search(
            query_embeddings=embeddings.cpu().numpy(),
            documents=self.observations,
            index=self.index,
            k=batch_size,
            return_idx=True
        )
        if isinstance(res, list):
            indices = [i for d,i in res][0]
            # print("res", res)
            # print("tensor_imgs", tensor_imgs)
            # print("embeddings", embeddings)
        else:
            docs, indices = res
        if isinstance(indices, np.ndarray):
            indices = indices[0]
        # score results
        data = (
            self.observations[indices, 0, :],
            self.actions[indices, 0, :],
            self.next_observations[indices, 0, :],
            self.dones[indices, 0].reshape(-1, 1),
            self.rewards[indices, 0].reshape(-1, 1),
        )
        out = ReplayBufferSamples(*tuple(map(self.to_torch, data)))
        return out
    