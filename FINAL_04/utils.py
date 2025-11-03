# utils.py (with deterministic seeding for consistent splits and weights)
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from collections import Counter
import numpy as np
import random

# ------------------------------------------------------------
# ✅ Fix random seeds globally for full reproducibility
# ------------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

def load_client_data(csv_path):
    df = pd.read_csv(csv_path)
    
    # --- Feature Engineering ---
    df["first_use_date"] = pd.to_datetime(df["first_use_date"], errors="coerce")
    df["first_use_year"] = df["first_use_date"].dt.year.fillna(0).astype(int)
    df["first_use_month"] = df["first_use_date"].dt.month.fillna(0).astype(int)
    df["first_use_day"] = df["first_use_date"].dt.day.fillna(0).astype(int)
    df["first_use_hour"] = df["first_use_date"].dt.hour.fillna(0).astype(int)
    
    # Drop unnecessary cols
    df = df.drop(columns=["user", "first_use_date"], errors='ignore')
    
    # --- Features & labels ---
    X = df.drop(columns=["malicious"])
    y = df["malicious"].astype(int)

    # --- CLASS WEIGHTS CALCULATION (same formula) ---
    class_counts = Counter(y.values)
    num_samples = len(y)
    num_classes = 2  # Binary classification

    weights = []
    for i in range(num_classes):
        weight = num_samples / (num_classes * class_counts.get(i, 1))
        weights.append(weight)
        
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32)
    print(f"Calculated Class Weights for {csv_path}: {class_weights_tensor.tolist()}")

    # --- Convert to tensors ---
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.long)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    
    # --- Deterministic Train-Test Split ---
    n = len(dataset)
    train_size = int(0.8 * n)
    test_size = n - train_size

    # Use a generator with fixed seed for reproducibility
    generator = torch.Generator().manual_seed(SEED)
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size], generator=generator)
    
    in_dim = X.shape[1]
    
    return train_ds, test_ds, in_dim, class_weights_tensor

def load_full_dataset(csv_path):
    """
    Loads and preprocesses a full dataset from a CSV file without splitting it.
    This is intended for final hold-out evaluation.
    """
    df = pd.read_csv(csv_path)

    # --- Feature Engineering (same as before) ---
    df["first_use_date"] = pd.to_datetime(df["first_use_date"], errors="coerce")
    df["first_use_year"] = df["first_use_date"].dt.year.fillna(0).astype(int)
    df["first_use_month"] = df["first_use_date"].dt.month.fillna(0).astype(int)
    df["first_use_day"] = df["first_use_date"].dt.day.fillna(0).astype(int)
    df["first_use_hour"] = df["first_use_date"].dt.hour.fillna(0).astype(int)
    df = df.drop(columns=["user", "first_use_date"], errors='ignore')

    # --- Features & labels ---
    X = df.drop(columns=["malicious"])
    y = df["malicious"].astype(int)

    # --- Convert to tensors ---
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.long)
    
    # --- Create dataset without splitting ---
    dataset = TensorDataset(X_tensor, y_tensor)
    in_dim = X.shape[1]
    
    return dataset, in_dim