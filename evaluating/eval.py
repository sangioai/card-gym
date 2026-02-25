
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


class ContrastiveMonitor:
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device

    # ---------------------------------------------------
    # Extract embeddings from loader
    # ---------------------------------------------------
    @torch.no_grad()
    def extract_features(self, loader):
        self.model.eval()
        features = []
        labels = []
        from tqdm import tqdm
        for images, lbls in tqdm(loader):
            images = images.to(self.device)
            emb = self.model(images)
            emb = F.normalize(emb, dim=1)
            features.append(emb.cpu())
            labels.append(lbls)

        features = torch.cat(features)
        labels = torch.cat(labels)

        return features, labels

    # ---------------------------------------------------
    # k-NN accuracy
    # ---------------------------------------------------
    def knn_accuracy(self, features, labels, k=5):
        knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
        knn.fit(features.numpy(), labels.numpy())
        preds = knn.predict(features.numpy())
        acc = (preds == labels.numpy()).mean()
        return acc

    # ---------------------------------------------------
    # Nearest centroid accuracy
    # ---------------------------------------------------
    def centroid_accuracy(self, features, labels):
        centroids = []
        centroid_labels = []

        for lbl in labels.unique():
            mask = labels == lbl
            centroids.append(features[mask].mean(dim=0))
            centroid_labels.append(lbl.item())

        centroids = torch.stack(centroids)

        sims = torch.matmul(features, centroids.T)
        preds = sims.argmax(dim=1)
        pred_labels = torch.tensor(
            [centroid_labels[p.item()] for p in preds]
        )

        acc = (pred_labels == labels).float().mean().item()
        return acc

    # ---------------------------------------------------
    # Positive vs Negative similarity
    # ---------------------------------------------------
    def similarity_metrics(self, features, labels):
        features = F.normalize(features, dim=1)
        sim_matrix = torch.matmul(features, features.T)

        labels = labels.view(-1, 1)
        positive_mask = (labels == labels.T)
        negative_mask = ~positive_mask

        positive_mask.fill_diagonal_(False)

        pos_sim = sim_matrix[positive_mask].mean().item()
        neg_sim = sim_matrix[negative_mask].mean().item()

        return pos_sim, neg_sim

    # ---------------------------------------------------
    # Collapse detector
    # ---------------------------------------------------
    def feature_std(self, features):
        return features.std(dim=0).mean().item()

    # ---------------------------------------------------
    # Full evaluation
    # ---------------------------------------------------
    def evaluate(self, loader, k=5):
        features, labels = self.extract_features(loader)

        knn_acc = self.knn_accuracy(features, labels, k)
        centroid_acc = self.centroid_accuracy(features, labels)
        pos_sim, neg_sim = self.similarity_metrics(features, labels)
        feat_std = self.feature_std(features)

        return {
            "knn_acc": knn_acc,
            "centroid_acc": centroid_acc,
            "pos_sim": pos_sim,
            "neg_sim": neg_sim,
            "feature_std": feat_std,
        }
    

# -------------------------------------------------------
# Build class centroids from dataset
# -------------------------------------------------------

@torch.no_grad()
def compute_centroids(model, loader, device="cuda"):
    model.eval()
    features = []
    labels = []
    from tqdm import tqdm
    for images, lbls in tqdm(loader):
        images = images.to(device)
        embeddings = model(images)
        features.append(embeddings.cpu())
        labels.append(lbls)

    features = torch.cat(features)
    labels = torch.cat(labels)

    centroids = {}
    for label in labels.unique():
        mask = labels == label
        centroids[int(label)] = features[mask].mean(dim=0)

    return centroids


# -------------------------------------------------------
# Predict via nearest centroid
# -------------------------------------------------------

def predict_with_centroids(embeddings, centroids):
    preds = []
    centroid_matrix = torch.stack(list(centroids.values()))
    centroid_labels = list(centroids.keys())

    sims = cosine_similarity(
        embeddings.cpu().numpy(),
        centroid_matrix.cpu().numpy()
    )

    best_idx = sims.argmax(axis=1)
    for idx in best_idx:
        preds.append(centroid_labels[idx])

    return preds


# -------------------------------------------------------
# Visualization
# -------------------------------------------------------

@torch.no_grad()
def visualize_batch(model, loader, centroids, class_names, device="cuda"):
    model.eval()
    images, labels = next(iter(loader))

    images_device = images.to(device)
    embeddings = model(images_device)

    preds = predict_with_centroids(embeddings, centroids)

    images = images.cpu()
    labels = labels.cpu()

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()

    for i in range(min(len(images), 8)):
        img = images[i].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)

        axes[i].imshow(img)
        axes[i].axis("off")

        true_label = class_names[labels[i]]
        pred_label = class_names[preds[i]]

        color = "green" if preds[i] == labels[i] else "red"

        axes[i].set_title(
            f"True: {true_label}\nPred: {pred_label}",
            color=color
        )

    plt.tight_layout()
    plt.show()