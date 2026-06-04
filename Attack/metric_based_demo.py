"""
End-to-end demo for metric-based membership inference attacks.

Demonstrates all five metric-based attack methods:
    loss, correctness, confidence, entropy, modified_entropy

Run
---
python Attack/metric_based_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Attack.metric_based import (
    AttackInput,
    LossAttack,
    CorrectnessAttack,
    ConfidenceAttack,
    EntropyAttack,
    ModifiedEntropyAttack,
)


class TabularMLP(nn.Module):
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


def train_classifier(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    device: torch.device,
    epochs: int = 60,
    lr: float = 1e-3,
    batch_size: int = 64,
) -> nn.Module:
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
    return model


def classification_accuracy(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, device: torch.device
) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(x.to(device))
        preds = logits.argmax(dim=1).cpu()
    return float((preds == y.cpu()).float().mean().item())


def print_results(name: str, attack_output) -> None:
    print(f"\n--- {name} ---")
    if attack_output.evaluation is not None:
        e = attack_output.evaluation
        print(f"  Accuracy:  {e.accuracy:.4f}" if e.accuracy else "  Accuracy:  N/A")
        print(f"  AUROC:     {e.auroc:.4f}" if e.auroc else "  AUROC:     N/A")
        if e.tpr_at_fpr:
            print(f"  TPR@1%FPR: {e.tpr_at_fpr['1%']:.4f}")
            print(f"  TPR@0.1%FPR: {e.tpr_at_fpr['0.1%']:.4f}")
    print(f"  First 5 scores: {np.asarray(attack_output.membership_scores[:5]).round(4)}")


def main() -> None:
    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    dataset = load_breast_cancer()
    x_all = dataset.data.astype(np.float32)
    y_all = dataset.target.astype(np.int64)

    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        x_all, y_all, test_size=0.5, stratify=y_all, random_state=seed
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_raw).astype(np.float32)
    x_test = scaler.transform(x_test_raw).astype(np.float32)

    input_dim = x_train.shape[1]
    num_classes = len(np.unique(y_all))

    # Train target model
    target_model = TabularMLP(input_dim=input_dim, num_classes=num_classes)
    target_model = train_classifier(
        target_model,
        torch.tensor(x_train),
        torch.tensor(y_train, dtype=torch.long),
        device,
        epochs=80,
    )

    train_acc = classification_accuracy(
        target_model, torch.tensor(x_train), torch.tensor(y_train, dtype=torch.long), device
    )
    test_acc = classification_accuracy(
        target_model, torch.tensor(x_test), torch.tensor(y_test, dtype=torch.long), device
    )
    print(f"Target model - Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

    # Prepare attack samples (balanced members/non-members)
    rng = np.random.default_rng(seed + 1)
    n_attack = min(len(x_train), len(x_test)) // 2
    nonmember_idx = rng.choice(len(x_test), size=n_attack, replace=False)
    member_idx = rng.choice(len(x_train), size=n_attack, replace=False)

    attack_samples = np.concatenate([x_train[member_idx], x_test[nonmember_idx]], axis=0)
    attack_labels = np.concatenate([y_train[member_idx], y_test[nonmember_idx]], axis=0)
    membership_labels = np.concatenate(
        [np.ones(n_attack, dtype=np.int64), np.zeros(n_attack, dtype=np.int64)], axis=0
    )

    print(f"\nAttack samples: {len(attack_samples)} (members={n_attack}, non-members={n_attack})")

    attack_input = AttackInput(
        target_model=target_model,
        samples=torch.tensor(attack_samples, dtype=torch.float32),
        labels=torch.tensor(attack_labels, dtype=torch.long),
        membership_labels=torch.tensor(membership_labels, dtype=torch.long),
        config={"batch_size": 64},
        metadata={"dataset_name": "breast_cancer"},
    )

    # Run all five metric-based attacks
    print("\n" + "=" * 60)
    print("Metric-Based Attack Results")
    print("=" * 60)

    attacks = {
        "Loss": LossAttack(batch_size=64, device=str(device)),
        "Correctness": CorrectnessAttack(batch_size=64, device=str(device)),
        "Confidence": ConfidenceAttack(batch_size=64, device=str(device)),
        "Entropy": EntropyAttack(batch_size=64, device=str(device)),
        "Modified Entropy": ModifiedEntropyAttack(batch_size=64, device=str(device)),
    }

    for name, attack in attacks.items():
        output = attack.run(attack_input)
        print_results(name, output)


if __name__ == "__main__":
    main()
