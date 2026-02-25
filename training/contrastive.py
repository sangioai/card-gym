import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
import math
from typing import Tuple


# =========================================================
# 1. Projection Head
# =========================================================

class ProjectionHead(nn.Module):
    """
    2-layer MLP projection head (SimCLR style)
    """
    def __init__(self, in_dim: int, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


# =========================================================
# 2. Supervised InfoNCE Loss
# =========================================================

class SupConLoss(nn.Module):
    """
    Supervised contrastive loss (multi-positive InfoNCE)
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor):
        """
        features: (B, D) normalized embeddings
        labels:   (B,)
        """

        device = features.device
        B = features.shape[0]

        # similarity matrix
        sim_matrix = torch.matmul(features, features.T) / self.temperature

        # mask to remove self-comparisons
        mask = torch.eye(B, device=device).bool()
        sim_matrix = sim_matrix.masked_fill(mask, -1e9)

        # positive mask (same identity)
        labels = labels.contiguous().view(-1, 1)
        positive_mask = torch.eq(labels, labels.T).float().to(device)
        positive_mask = positive_mask.masked_fill(mask, 0)

        # log-softmax
        log_prob = F.log_softmax(sim_matrix, dim=1)

        # compute mean log-likelihood over positives
        positives_per_row = positive_mask.sum(1)
        loss = -(positive_mask * log_prob).sum(1) / (positives_per_row + 1e-8)

        return loss.mean()
    
# =========================================================
# 2.1 ArcFace Loss (Margin-Based Loss)
# =========================================================

class ArcFaceLoss(nn.Module):
    def __init__(self, embedding_dim, num_classes, margin=0.5, scale=30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.ce = nn.CrossEntropyLoss()

        self.margin = margin
        self.scale = scale

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings)
        weight = F.normalize(self.weight)

        cosine = torch.matmul(embeddings, weight.T)
        theta = torch.acos(torch.clamp(cosine, -1 + 1e-7, 1 - 1e-7))
        target_logits = torch.cos(theta + self.margin)

        one_hot = F.one_hot(labels, num_classes=weight.shape[0]).float()
        logits = cosine * (1 - one_hot) + target_logits * one_hot
        logits *= self.scale

        return self.ce(logits, labels)
    

# =========================================================
# 2.2 Hard Negative Miner (Modular)
# =========================================================

class HardNegativeMiner:
    def __init__(self, top_k=5):
        self.top_k = top_k

    def mine(self, embeddings, labels):
        embeddings = F.normalize(embeddings, dim=1)
        sim_matrix = torch.matmul(embeddings, embeddings.T)

        labels = labels.view(-1, 1)
        negative_mask = labels != labels.T

        hard_negatives = []
        for i in range(len(embeddings)):
            neg_sims = sim_matrix[i][negative_mask[i]]
            if len(neg_sims) == 0:
                continue
            topk_vals, _ = torch.topk(neg_sims, min(self.top_k, len(neg_sims)))
            hard_negatives.append(topk_vals.mean())

        if len(hard_negatives) == 0:
            return 0.0

        return torch.stack(hard_negatives).mean()

# =========================================================
# 3. Contrastive Model Wrapper (Model-Agnostic)
# =========================================================

class ContrastiveModel(nn.Module):
    """
    Wraps any encoder + projection head
    """
    def __init__(self, encoder: nn.Module, embedding_dim: int, proj_dim: int = 256):
        super().__init__()
        self.encoder = encoder
        self.projector = ProjectionHead(embedding_dim, proj_dim)

    def forward(self, x):
        features = self.encoder(x)
        if isinstance(features, (tuple, list)):
            features = features[0]
        return self.projector(features)


# =========================================================
# 4. Trainer
# =========================================================


class ContrastiveTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        loss_module,
        hard_miner=None,
        lr=3e-4,
        device="cuda",
        save_dir="checkpoints"
    ):
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = train_loader
        self.loss_module = loss_module.to(device)
        self.hard_miner = hard_miner

        self.optimizer = torch.optim.AdamW(
            list(model.parameters()) +
            list(loss_module.parameters()),
            lr=lr
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )

        self.scaler = torch.amp.GradScaler()
        self.best_val_loss = float("inf")
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            embeddings = self.model(images)
            loss = self.loss_module(embeddings, labels)

            # Optional hard negative mining
            if self.hard_miner is not None:
                hard_penalty = self.hard_miner.mine(
                    embeddings.detach(), labels
                )
                loss = loss + 0.1 * hard_penalty

            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

        self.scheduler.step()
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def eval_epoch(self):
        self.model.eval()
        total_loss = 0

        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                embeddings = self.model(images)
                loss = self.loss_module(embeddings, labels)

            total_loss += loss.item()

        return total_loss / len(self.val_loader)
    
    def save_checkpoint(self, name):
        torch.save({
            "model": self.model.state_dict(),
            "loss_module": self.loss_module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }, os.path.join(self.save_dir, name))

    def fit(self, epochs: int):
        from tqdm import tqdm
        for epoch in tqdm(range(epochs)):
            train_loss = self.train_epoch()
            val_loss = self.eval_epoch()

            print(
                f"Epoch {epoch+1:03d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )

            # Save last model
            self.save_checkpoint("last_model.pt")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best_model.pt")


# =========================================================
# 5. Example Usage
# =========================================================

if __name__ == "__main__":
    import os
    import timm
    import torch
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader, Subset
    from sklearn.model_selection import train_test_split
    import torchvision.transforms as transforms
    FOLDER_DATASET = "test"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Example encoder (model-agnostic)
    model_name = "resnet101"
    encoder = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=0
    )

    embedding_dim = encoder.num_features
    proj_dim = 256

    model = ContrastiveModel(
        encoder=encoder,
        embedding_dim=embedding_dim,
        proj_dim=proj_dim
    )

    transform = transforms.Compose([
        # transforms.Resize(224),
        transforms.RandomResizedCrop(224, (0.8,1)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4,0.4,0.4,0.1),
        transforms.ToTensor(), 
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(), 
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # Load full dataset
    dataset = ImageFolder(FOLDER_DATASET, transform=transform)
    val_dataset = ImageFolder(FOLDER_DATASET, transform=val_transform)

    # Get targets (class labels)
    targets = dataset.targets

    # Stratified split
    train_idx, test_idx = train_test_split(
        range(len(targets)),
        test_size=0.2,
        stratify=targets,
        random_state=42
    )

    # Create subsets
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_dataset)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=20)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=20)
    # loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=20)

    use_arcface = True
    use_hard_mining = True

    trainer = ContrastiveTrainer(
          model
        , train_loader
        , val_loader
        , SupConLoss(0.07) if not use_arcface else ArcFaceLoss(proj_dim, len(dataset.classes), margin=0.5, scale=30.)
        , hard_miner= HardNegativeMiner(top_k=5) if use_hard_mining else None
        , lr=3e-4
        , device=device
    )
    # trainer = ContrastiveTrainer(model, loader, lr=3e-4, device=device)
    epochs = 100
    trainer.fit(epochs=epochs)