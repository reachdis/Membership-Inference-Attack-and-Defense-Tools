"""
End-to-end demo for the DP-SGD defense.

This demo mirrors the style of the attack demos:
1. Load a real tabular dataset.
2. Save it to CSV if needed, then reload from CSV.
3. Train a standard non-private baseline model.
4. Train a DP-SGD defended model through the unified defense interface.
5. Compare train/test utility and inspect the defended outputs.

Run
---
python Defense/dp_sgd_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Defense.base import DefenseInput
from Defense.dp_sgd import DPSGDDefense


DATA_DIR = Path(__file__).resolve().parent / "demo_data"
DATA_PATH = DATA_DIR / "breast_cancer.csv"


class TabularMLP(nn.Module):
    """Simple classifier used in the demo."""

    def __init__(self, input_dim: int, num_classes: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def ensure_breast_cancer_csv(path: Path) -> None:
    """Export the sklearn breast cancer dataset to CSV once."""
    if path.exists():
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    dataset = load_breast_cancer(as_frame=True)
    df = dataset.frame.copy().rename(columns={"target": "label"})
    df.to_csv(path, index=False)


def load_tabular_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load features and labels from a CSV file with a `label` column."""
    df = pd.read_csv(path)
    if "label" not in df.columns:
        raise ValueError("CSV file must contain a 'label' column.")
    x = df.drop(columns=["label"]).to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int64)
    return x, y


def train_baseline_model(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    device: torch.device,
    epochs: int = 40,
    lr: float = 1e-3,
    batch_size: int = 32,
) -> nn.Module:
    """Train a standard non-private baseline model."""
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
    return model.eval()


def predict_labels(
    model: nn.Module,
    samples: torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Predict class labels."""
    loader = DataLoader(TensorDataset(samples), batch_size=batch_size, shuffle=False)
    all_preds = []

    model.eval()
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.detach().cpu())

    return torch.cat(all_preds, dim=0).numpy().astype(np.int64)


def classification_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    """Compute accuracy from predicted labels."""
    return float(np.mean(preds.astype(np.int64) == labels.astype(np.int64)))


def main() -> None:
    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1-2: prepare and load a real tabular CSV dataset.
    ensure_breast_cancer_csv(DATA_PATH)
    x_all, y_all = load_tabular_csv(DATA_PATH)

    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        x_all,
        y_all,
        test_size=0.3,
        stratify=y_all,
        random_state=seed,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_raw).astype(np.float32)
    x_test = scaler.transform(x_test_raw).astype(np.float32)

    input_dim = x_train.shape[1]
    num_classes = int(y_all.max()) + 1

    train_x_tensor = torch.tensor(x_train, dtype=torch.float32)
    train_y_tensor = torch.tensor(y_train, dtype=torch.long)
    test_x_tensor = torch.tensor(x_test, dtype=torch.float32)
    test_y_tensor = torch.tensor(y_test, dtype=torch.long)

    # Step 3: train a standard non-private baseline model.
    baseline_model = TabularMLP(input_dim=input_dim, num_classes=num_classes)
    baseline_model = train_baseline_model(
        model=baseline_model,
        train_x=train_x_tensor,
        train_y=train_y_tensor,
        device=device,
        epochs=40,
        lr=1e-3,
        batch_size=32,
    )
    baseline_train_preds = predict_labels(baseline_model, train_x_tensor, device=device)
    baseline_test_preds = predict_labels(baseline_model, test_x_tensor, device=device)
    baseline_train_acc = classification_accuracy(baseline_train_preds, y_train)
    baseline_test_acc = classification_accuracy(baseline_test_preds, y_test)

    # Step 4: train a DP-SGD defended model through the defense interface.
    defense_input = DefenseInput(
        model_factory=lambda: TabularMLP(input_dim=input_dim, num_classes=num_classes),
        train_data=train_x_tensor,
        train_labels=train_y_tensor,
        test_data=test_x_tensor,
        test_labels=test_y_tensor,
        samples=test_x_tensor[:16],
        labels=test_y_tensor[:16],
        defense_config={
            "epochs": 5,
            "batch_size": 32,
            "learning_rate": 1e-3,
            "noise_multiplier": 0.6,
            "max_grad_norm": 1.0,
        },
        eval_config={
            "compute_utility": True,
        },
        metadata={
            "dataset_name": "breast_cancer_csv",
            "defense_name": "dp_sgd",
        },
    )

    defense = DPSGDDefense(
        batch_size=32,
        epochs=5,
        learning_rate=1e-3,
        noise_multiplier=0.6,
        max_grad_norm=1.0,
        device=str(device),
    )
    defense_output = defense.run(defense_input)

    defended_model = defense_output.defended_model
    if defended_model is None:
        raise RuntimeError("DP-SGD defense did not return a defended model.")

    defended_train_preds = predict_labels(defended_model, train_x_tensor, device=device)
    defended_test_preds = predict_labels(defended_model, test_x_tensor, device=device)
    defended_train_acc = classification_accuracy(defended_train_preds, y_train)
    defended_test_acc = classification_accuracy(defended_test_preds, y_test)

    # Step 5: print results.
    print("=" * 60)
    print("Dataset")
    print("=" * 60)
    print(f"Dataset CSV:         {DATA_PATH}")
    print(f"Train shape:         {x_train.shape}")
    print(f"Test shape:          {x_test.shape}")

    print("\n" + "=" * 60)
    print("Baseline Model")
    print("=" * 60)
    print(f"Train Accuracy:      {baseline_train_acc:.4f}")
    print(f"Test Accuracy:       {baseline_test_acc:.4f}")

    print("\n" + "=" * 60)
    print("DP-SGD Defended Model")
    print("=" * 60)
    print(f"Train Accuracy:      {defended_train_acc:.4f}")
    print(f"Test Accuracy:       {defended_test_acc:.4f}")

    if defense_output.evaluation is not None:
        utility_metrics = defense_output.evaluation.utility_metrics or {}
        privacy_metrics = defense_output.evaluation.privacy_metrics or {}
        efficiency_metrics = defense_output.evaluation.efficiency_metrics or {}

        print(f"Eval Train Accuracy: {utility_metrics.get('train_accuracy', float('nan')):.4f}")
        print(f"Eval Test Accuracy:  {utility_metrics.get('test_accuracy', float('nan')):.4f}")
        print(f"Noise Multiplier:    {privacy_metrics.get('noise_multiplier', float('nan')):.4f}")
        print(f"Max Grad Norm:       {privacy_metrics.get('max_grad_norm', float('nan')):.4f}")
        if "train_time" in efficiency_metrics:
            print(f"Train Time (s):      {efficiency_metrics['train_time']:.4f}")

    if defense_output.protected_outputs is not None:
        print("\nFirst 16 defended predictions:")
        print(np.asarray(defense_output.protected_outputs))


if __name__ == "__main__":
    main()
