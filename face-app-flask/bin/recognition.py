import os
import torch
import pickle
import numpy as np
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms, models

FILE_PATH = 'embeddings.pkl'

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        # Load a pre-trained ResNet18 model
        backbone = models.resnet18(pretrained=True)
        modules = list(backbone.children())[:-1]  # Remove the final classification layer
        self.feature_extractor = nn.Sequential(*modules)  # Use everything except the final FC layer

        # Add a new fully connected layer to project features into a lower-dimensional embedding space
        self.fc = nn.Linear(backbone.fc.in_features, embedding_dim)

    def forward_once(self, x):
        # Pass input through the feature extractor
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc(x)             # Project to embedding_dim
        return F.normalize(x, p=2, dim=1)  # Normalize embeddings to have unit L2 norm

    def forward(self, datapoint):
        # Forward pass for a triplet: anchor (a), positive (p), negative (n)
        return self.forward_once(datapoint)

transform = transforms.Compose([
    transforms.Resize((256, 256)),              # Resize images to 256x256
    transforms.CenterCrop(224),                 # Crop the center 224x224 patch (standard size for ResNet)
    transforms.ToTensor(),                      # Convert PIL images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize using ImageNet statistics
                         std=[0.229, 0.224, 0.225]),
])

# ------------------------------------------------- #
# ------------------------------------------------- #

def load_model(model_path="model.pth", device="cpu"):
    loaded_model = SiameseNetwork().to(device)
    loaded_model.load_state_dict(torch.load(model_path, map_location=device))
    loaded_model.eval()
    print("Model loaded and set to eval mode.")
    return loaded_model

def compute_embedding(image_path, loaded_model, device="cpu"):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = loaded_model(image)
    return embedding.cpu().numpy()  # Convert to numpy array for easier handling

# ------------------------------------------------- #
# ------------------------------------------------- #

def save_embedding(name, new_embedding):
    if os.path.exists(FILE_PATH):
        with open(FILE_PATH, 'rb') as f:
            embeddings = pickle.load(f)
    else:
        embeddings = []
        print("File doesn't exist!")
    for embedding in embeddings:
        if name in embedding[0]:
            raise ValueError(f"Name '{name}' already exists. Choose a different name.")
    embeddings.append([name, new_embedding])
    with open(FILE_PATH, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f"Saved embedding for '{name}'.")

def show_all_embeddings():
    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError("No embeddings file found.")
    with open(FILE_PATH, 'rb') as f:
        embeddings = pickle.load(f)
    print(f"Total embeddings stored: {len(embeddings)}")
    for data in embeddings:
        print(data[0], data[1].shape)

def get_embedding(name):
    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError("No embeddings file found.")
    with open(FILE_PATH, 'rb') as f:
        embeddings = pickle.load(f)
    for entry_name, embedding in embeddings:
        if entry_name == name:
            return embedding
    raise KeyError(f"No embedding found for name '{name}'.")

def find_most_similar(query_embedding):
    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError("No embeddings file found.")
    with open(FILE_PATH, 'rb') as f:
        embeddings = pickle.load(f)
    best_name = None
    best_similarity = -1
    for name, emb in embeddings:
        sim = F.cosine_similarity(torch.Tensor(query_embedding), torch.Tensor(emb)).item()
        if sim > best_similarity:
            best_similarity = sim
            best_name = name
    return best_name, best_similarity

def average_image_embeddings(model, *images):
    embs = []
    for img in images:
        emb = compute_embedding(img, model)  # your existing function
        embs.append(emb)
    if not embs:
        raise ValueError("No images provided to compute average embedding.")
    tensor_embs = [
        torch.tensor(e, dtype=torch.float)
        for e in embs
    ]
    stacked = torch.stack(tensor_embs, dim=0)      # shape: (N_images, D)
    avg_emb = stacked.mean(dim=0)           # shape: (D,)
    return avg_emb

# ------------------------------------------------- #
# ------------------------------------------------- #