import os
import torch
import torchvision.transforms as T
from PIL import Image

DATASET = "datasets/dataset-1"
DB_DIR = os.path.join(DATASET, "frames-IMG_3597")
QUERY_DIR = os.path.join(DATASET, "frames-IMG_3598")

transform = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_images(directory):
    paths = sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".png")
    ])
    return paths

def extract_features(model, paths, device):
    features = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model(tensor)
        features.append(feat.squeeze(0))
    return torch.stack(features)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading DINOv2...")
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
model.eval().to(device)

db_paths = load_images(DB_DIR)
query_paths = load_images(QUERY_DIR)
print(f"Database frames (3597): {len(db_paths)}")
print(f"Query frames   (3598): {len(query_paths)}")

print("Extracting database features...")
db_feats = extract_features(model, db_paths, device)

print("Extracting query features...")
query_feats = extract_features(model, query_paths, device)

db_feats_norm = db_feats / db_feats.norm(dim=1, keepdim=True)
query_feats_norm = query_feats / query_feats.norm(dim=1, keepdim=True)

print("\n--- Matches (3598 query → 3597 best match) ---")
for q_idx, q_path in enumerate(query_paths):
    sims = query_feats_norm[q_idx] @ db_feats_norm.T
    best_idx = sims.argmax().item()
    best_sim = sims[best_idx].item()
    q_name = os.path.basename(q_path)
    db_name = os.path.basename(db_paths[best_idx])
    print(f"  {q_name}  →  {db_name}  (cosine sim: {best_sim:.4f})")
