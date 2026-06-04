"""
End-to-end demo for LOGAN Attack (discriminator confidence).

Wraps the gen_mem_inf library.
Reference: Hayes et al., "LOGAN: Membership Inference Attacks Against
Generative Models" (PoPETs 2019)

This demo uses a simple synthetic setup to verify the interface works.
For real use, replace with actual GAN models trained on image datasets.

Run
---
python Attack/logan_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Attack.logan_attack import AttackInput, LOGANAttack


class SimpleDiscriminator(nn.Module):
    """Simple discriminator for demo purposes."""

    def __init__(self, input_dim: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def infer(self, x: torch.Tensor) -> torch.Tensor:
        """Interface expected by LOGANAttack wrapper."""
        return self.forward(x).squeeze(-1)


def main():
    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    input_dim = 10
    n_members = 100
    n_nonmembers = 100

    # Generate synthetic data: members follow one distribution, non-members another
    rng = np.random.default_rng(seed)
    member_data = rng.normal(loc=0.5, scale=0.5, size=(n_members, input_dim)).astype(np.float32)
    nonmember_data = rng.normal(loc=-0.5, scale=0.5, size=(n_nonmembers, input_dim)).astype(np.float32)

    # Train a discriminator on member data (label=1) and random noise (label=0)
    discriminator = SimpleDiscriminator(input_dim).to(device)
    opt = torch.optim.Adam(discriminator.parameters(), lr=1e-3)
    crit = nn.BCELoss()

    for _ in range(200):
        disc_mem = torch.tensor(member_data).to(device)
        disc_noise = torch.randn(n_members, input_dim).to(device)

        opt.zero_grad()
        loss_real = crit(discriminator(disc_mem).squeeze(), torch.ones(n_members, device=device))
        loss_fake = crit(discriminator(disc_noise).squeeze(), torch.zeros(n_members, device=device))
        (loss_real + loss_fake).backward()
        opt.step()

    discriminator.eval()

    # Attack: higher discriminator output = more likely member
    attack_samples = np.concatenate([member_data, nonmember_data])
    membership_labels = np.concatenate([
        np.ones(n_members, dtype=np.int64),
        np.zeros(n_nonmembers, dtype=np.int64),
    ])

    attack_input = AttackInput(
        target_model=discriminator,
        samples=torch.tensor(attack_samples),
        membership_labels=torch.tensor(membership_labels, dtype=torch.long),
        config={"attack_mode": "white_box", "batch_size": 32},
    )

    attack = LOGANAttack(batch_size=32, device=str(device))
    output = attack.run(attack_input)

    print(f"\nLOGAN Attack Results")
    print(f"Attacked: {len(attack_samples)} (members={n_members}, non-members={n_nonmembers})")
    if output.evaluation:
        e = output.evaluation
        print(f"  Accuracy:  {e.accuracy:.4f}")
        print(f"  AUROC:     {e.auroc:.4f}")
        if e.tpr_at_fpr:
            print(f"  TPR@1%FPR: {e.tpr_at_fpr['1%']:.4f}")

    # Check: members should have higher discriminator confidence
    member_scores = output.membership_scores[:n_members]
    nonmember_scores = output.membership_scores[n_members:]
    print(f"  Mean member score:      {member_scores.mean():.4f}")
    print(f"  Mean non-member score:  {nonmember_scores.mean():.4f}")


if __name__ == "__main__":
    main()
