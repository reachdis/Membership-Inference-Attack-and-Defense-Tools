"""
End-to-end demo for the classic shadow-based membership inference attack.

This demo is intentionally kept separate from `shadow_based.py`.
`shadow_based.py` should remain a reusable implementation module, while this
file shows developers how to call the interface from loading tabular data all
the way to evaluating the attack.

Demo pipeline
-------------
1. Load a real tabular dataset: breast cancer classification.
2. Save it to CSV if the local demo CSV does not already exist.
3. Reload the CSV from disk as a standard tabular file.
4. Split data into target-model data and shadow-model auxiliary data.
5. Train a target MLP classifier.
6. Train several shadow models.
7. Run the shadow-based MIA against a mixture of target members and non-members.
8. Evaluate the MIA performance.

Run
---
python Attack/shadow_based_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

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

from Attack.shadow_based import AttackInput, ShadowBasedAttack


DATA_DIR = Path(__file__).resolve().parent / "demo_data"
DATA_PATH = DATA_DIR / "breast_cancer.csv"


class TabularMLP(nn.Module):
    """Simple target/shadow model used in the demo."""

    def __init__(self, input_dim: int, num_classes: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def ensure_breast_cancer_csv(path: Path) -> None:
    """Export a real tabular dataset to CSV once so the demo loads from file."""
    if path.exists():
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    dataset = load_breast_cancer(as_frame=True)
    df = dataset.frame.copy() # type: ignore
    df.rename(columns={"target": "label"}, inplace=True)
    df.to_csv(path, index=False)


def load_tabular_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load features and labels from a CSV file."""
    df = pd.read_csv(path)
    if "label" not in df.columns:
        raise ValueError("CSV file must contain a 'label' column.")

    x = df.drop(columns=["label"]).to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int64)
    return x, y


def train_classifier(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    device: torch.device,
    epochs: int = 60,
    lr: float = 1e-3,
    batch_size: int = 64,
) -> nn.Module:
    """Train a small MLP classifier."""
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
    return model


def classification_accuracy(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
) -> float:
    """Compute classifier accuracy."""
    model.eval()
    with torch.no_grad():
        logits = model(x.to(device))
        preds = logits.argmax(dim=1).cpu()
    return float((preds == y.cpu()).float().mean().item())


def build_shadow_splits(
    shadow_x: np.ndarray,
    shadow_y: np.ndarray,
    num_shadow_models: int,
    seed: int,
) -> List[Dict[str, torch.Tensor]]:
    """Create shadow member/non-member splits for multiple shadow models."""
    shadow_splits: List[Dict[str, torch.Tensor]] = []

    for split_seed in range(num_shadow_models):
        train_x, test_x, train_y, test_y = train_test_split(
            shadow_x,
            shadow_y,
            test_size=0.5,
            stratify=shadow_y,
            random_state=seed + split_seed,
        )
        shadow_splits.append(
            {
                "train_samples": torch.tensor(train_x, dtype=torch.float32),
                "train_labels": torch.tensor(train_y, dtype=torch.long),
                "test_samples": torch.tensor(test_x, dtype=torch.float32),
                "test_labels": torch.tensor(test_y, dtype=torch.long),
            }
        )

    return shadow_splits


def main() -> None:
    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1-3: prepare and load a real tabular CSV dataset.
    ensure_breast_cancer_csv(DATA_PATH)
    x_all, y_all = load_tabular_csv(DATA_PATH)

    # Split the dataset into:
    # - target pool: used to build target-model members/non-members
    # - shadow pool: used only for shadow-model training
    x_target_pool, x_shadow_pool, y_target_pool, y_shadow_pool = train_test_split(
        x_all,
        y_all,
        test_size=0.5,
        stratify=y_all,
        random_state=seed,
    )

    # Split target pool into target members and target non-members.
    x_target_train_raw, x_target_test_raw, y_target_train, y_target_test = train_test_split(
        x_target_pool,
        y_target_pool,
        test_size=0.75,
        stratify=y_target_pool,
        random_state=seed + 1,
    )

    # Standardize with target-train statistics.
    scaler = StandardScaler()
    x_target_train = scaler.fit_transform(x_target_train_raw).astype(np.float32)
    x_target_test = scaler.transform(x_target_test_raw).astype(np.float32) # type: ignore
    x_shadow_pool = scaler.transform(x_shadow_pool).astype(np.float32) # type: ignore

    input_dim = x_target_train.shape[1]
    num_classes = int(y_all.max()) + 1

    # Step 4-5: train the target model.
    target_model = TabularMLP(input_dim=input_dim, num_classes=num_classes)
    target_model = train_classifier(
        model=target_model,
        train_x=torch.tensor(x_target_train, dtype=torch.float32),
        train_y=torch.tensor(y_target_train, dtype=torch.long),
        device=device,
        epochs=150,
        lr=1e-3,
        batch_size=32,
    )

    target_train_acc = classification_accuracy(
        target_model,
        torch.tensor(x_target_train, dtype=torch.float32),
        torch.tensor(y_target_train, dtype=torch.long),
        device,
    )
    target_test_acc = classification_accuracy(
        target_model,
        torch.tensor(x_target_test, dtype=torch.float32),
        torch.tensor(y_target_test, dtype=torch.long),
        device,
    )

    # Step 6: build shadow-model training splits.
    shadow_splits = build_shadow_splits(
        shadow_x=x_shadow_pool,
        shadow_y=y_shadow_pool,
        num_shadow_models=8,
        seed=seed + 10,
    )

    # Step 7: assemble the attacked samples.
    # Members are the target model's train samples.
    # Non-members are a balanced subset of the target model's held-out test samples
    # so that attack accuracy is easier to interpret in the demo.
    rng = np.random.default_rng(seed + 20)
    nonmember_indices = rng.choice(len(x_target_test), size=len(x_target_train), replace=False)
    x_target_test_attack = x_target_test[nonmember_indices]
    y_target_test_attack = y_target_test[nonmember_indices]

    attack_samples = np.concatenate([x_target_train, x_target_test_attack], axis=0)
    attack_labels = np.concatenate([y_target_train, y_target_test_attack], axis=0)
    membership_labels = np.concatenate(
        [
            np.ones(len(x_target_train), dtype=np.int64),
            np.zeros(len(x_target_test_attack), dtype=np.int64),
        ],
        axis=0,
    )

    attack_input = AttackInput(
        target_model=target_model,
        samples=torch.tensor(attack_samples, dtype=torch.float32),
        labels=torch.tensor(attack_labels, dtype=torch.long),
        membership_labels=torch.tensor(membership_labels, dtype=torch.long),
        shadow_data={
            "model_factory": lambda: TabularMLP(input_dim=input_dim, num_classes=num_classes),
            "shadow_splits": shadow_splits,
            "train_config": {
                "epochs": 60,
                "lr": 1e-3,
                "batch_size": 32,
                "test_size": 0.5,
            },
        },
        config={
            "num_classes": num_classes,
            "batch_size": 32,
        },
        metadata={
            "dataset_name": "breast_cancer_csv",
            "attack_name": "shadow_based",
        },
    )

    attack = ShadowBasedAttack(
        batch_size=32,
        attack_hidden_dim=64,
        attack_lr=1e-3,
        attack_epochs=20,
        per_class=True,
        device=str(device),
    )
    attack_output = attack.run(attack_input)

    # Step 8: report results.
    print("=" * 60)
    print("Target Model Performance")
    print("=" * 60)
    print(f"Dataset CSV:      {DATA_PATH}")
    print(f"Train Accuracy:   {target_train_acc:.4f}")
    print(f"Test Accuracy:    {target_test_acc:.4f}")

    print("\n" + "=" * 60)
    print("Shadow-Based MIA Performance")
    print("=" * 60)
    print(f"Attacked samples: {len(attack_samples)}")
    print(f"Members:          {int(membership_labels.sum())}")
    print(f"Non-members:      {int((1 - membership_labels).sum())}")

    if attack_output.evaluation is not None:
        eval_result = attack_output.evaluation
        print(f"Attack Accuracy:  {eval_result.accuracy:.4f}")
        print(f"Attack AUROC:     {eval_result.auroc:.4f}")
        print(f"TPR@1%FPR:        {eval_result.tpr_at_fpr['1%']:.4f}") # type: ignore
        print(f"TPR@0.1%FPR:      {eval_result.tpr_at_fpr['0.1%']:.4f}") # type: ignore

    print("\nFirst 10 membership scores:")
    print(np.asarray(attack_output.membership_scores[:10]))


if __name__ == "__main__":
    main()
