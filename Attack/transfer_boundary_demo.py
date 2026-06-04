"""
End-to-end demo for Transfer Attack and Boundary-Attack.

Note: BoundaryAttack requires additional dependencies:
    pip install foolbox adversarial-robustness-toolbox

Tests both label-only MIA methods from:
Li et al., "Membership Leakage in Label-Only Exposures" (CCS 2021)

Run
---
python Attack/transfer_boundary_demo.py
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

from Attack.transfer_attack import AttackInput, TransferAttack, BoundaryAttack


class TabularMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_classifier(model, train_x, train_y, device, epochs=60, lr=1e-3, batch_size=64):
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    model.to(device)
    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            opt.zero_grad()
            crit(model(bx.to(device)), by.to(device)).backward()
            opt.step()
    return model


def accuracy(model, x, y, device):
    model.eval()
    with torch.no_grad():
        preds = model(x.to(device)).argmax(dim=1).cpu()
    return float((preds == y.cpu()).float().mean().item())


def main():
    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    data = load_breast_cancer()
    x_all = data.data.astype(np.float32)
    y_all = data.target.astype(np.int64)

    # Split: target pool, reference pool, shadow pool
    x_target, x_aux, y_target, y_aux = train_test_split(
        x_all, y_all, test_size=0.5, stratify=y_all, random_state=seed
    )
    x_aux_train, x_aux_test, y_aux_train, y_aux_test = train_test_split(
        x_aux, y_aux, test_size=0.3, stratify=y_aux, random_state=seed + 1
    )
    x_target_train, x_target_test, y_target_train, y_target_test = train_test_split(
        x_target, y_target, test_size=0.5, stratify=y_target, random_state=seed + 2
    )

    scaler = StandardScaler()
    x_target_train = scaler.fit_transform(x_target_train).astype(np.float32)
    x_target_test = scaler.transform(x_target_test).astype(np.float32)
    x_aux_train = scaler.transform(x_aux_train).astype(np.float32)
    x_aux_test = scaler.transform(x_aux_test).astype(np.float32)

    input_dim = x_target_train.shape[1]
    num_classes = len(np.unique(y_all))

    # Train target model
    target_model = TabularMLP(input_dim, num_classes)
    target_model = train_classifier(
        target_model,
        torch.tensor(x_target_train),
        torch.tensor(y_target_train, dtype=torch.long),
        device,
    )
    print(f"Target model - Train: {accuracy(target_model, torch.tensor(x_target_train), torch.tensor(y_target_train, dtype=torch.long), device):.4f}, "
          f"Test: {accuracy(target_model, torch.tensor(x_target_test), torch.tensor(y_target_test, dtype=torch.long), device):.4f}")

    # Attack samples: balanced members/non-members
    n_attack = min(len(x_target_train), len(x_target_test)) // 2
    rng = np.random.default_rng(seed + 3)
    member_idx = rng.choice(len(x_target_train), size=n_attack, replace=False)
    nonmember_idx = rng.choice(len(x_target_test), size=n_attack, replace=False)

    attack_x = np.concatenate([x_target_train[member_idx], x_target_test[nonmember_idx]])
    attack_y = np.concatenate([y_target_train[member_idx], y_target_test[nonmember_idx]])
    mem_labels = np.concatenate([np.ones(n_attack, dtype=np.int64), np.zeros(n_attack, dtype=np.int64)])

    # ===================================================================
    # Transfer Attack
    # ===================================================================
    print("\n" + "=" * 60)
    print("Transfer Attack")
    print("=" * 60)

    attack_input = AttackInput(
        target_model=target_model,
        samples=torch.tensor(attack_x),
        labels=torch.tensor(attack_y, dtype=torch.long),
        membership_labels=torch.tensor(mem_labels, dtype=torch.long),
        reference_data={
            "aux_samples": x_aux_train,
            "aux_labels": y_aux_train,
        },
        config={"batch_size": 64, "num_classes": num_classes, "transfer_epochs": 50},
    )

    attack = TransferAttack(batch_size=64, transfer_epochs=50, device=str(device))
    output = attack.run(attack_input)

    print(f"Attacked: {len(attack_x)} (members={n_attack}, non-members={n_attack})")
    if output.evaluation:
        e = output.evaluation
        print(f"  Accuracy:  {e.accuracy:.4f}")
        print(f"  AUROC:     {e.auroc:.4f}")
        if e.tpr_at_fpr:
            print(f"  TPR@1%FPR: {e.tpr_at_fpr['1%']:.4f}")
    print(f"  First 5 scores: {output.membership_scores[:5].round(4)}")

    # ===================================================================
    # Boundary Attack
    # ===================================================================
    print("\n" + "=" * 60)
    print("Boundary Attack")
    print("=" * 60)

    try:
        attack_b = BoundaryAttack(device=str(device))
        output_b = attack_b.run(attack_input)
        print(f"Attacked: {len(attack_x)}")
        if output_b.evaluation:
            print(f"  AUROC: {output_b.evaluation.auroc:.4f}")
    except ImportError as e:
        print(f"(Skipped: {e})")


if __name__ == "__main__":
    main()
