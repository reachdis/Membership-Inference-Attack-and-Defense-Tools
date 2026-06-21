"""
Transfer Attack: Label-only MIA via transfer model.

Note: BoundaryAttack requires additional dependencies:
    pip install foolbox adversarial-robustness-toolbox

Wraps the Decision-based-MIA library (Li et al., CCS 2021).
https://github.com/zhenglisec/Decision-based-MIA

Attack principle from the paper:
1. Train a shadow model on auxiliary reference data.
2. Query the target model: compare target predictions to ground truth.
3. Use shadow model's loss/entropy/confidence as membership signals.

Unified convention:
    higher membership score -> more likely member
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, TensorDataset

_LIB_DIR = Path(__file__).resolve().parent / "transfer_boundary_lib"
if str(_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(_LIB_DIR))


@dataclass
class AttackInput:
    target_model: Optional[Any]
    samples: Any
    labels: Optional[Any] = None
    membership_labels: Optional[Any] = None
    signals: Optional[Dict[str, Any]] = None
    reference_data: Optional[Dict[str, Any]] = None
    shadow_data: Optional[Dict[str, Any]] = None
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    accuracy: Optional[float] = None
    auroc: Optional[float] = None
    tpr_at_fpr: Optional[Dict[str, float]] = None
    extra_metrics: Optional[Dict[str, Any]] = None


@dataclass
class AttackOutput:
    membership_scores: Any
    membership_preds: Optional[Any] = None
    evaluation: Optional[EvaluationResult] = None
    intermediate_outputs: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAttack:
    def fit(self, attack_input: AttackInput) -> "BaseAttack":
        return self

    def infer(self, attack_input: AttackInput) -> AttackOutput:
        raise NotImplementedError

    def evaluate(self, attack_output: AttackOutput, attack_input: AttackInput) -> EvaluationResult:
        y_true = _to_numpy_1d(attack_input.membership_labels)
        y_score = _to_numpy_1d(attack_output.membership_scores)
        y_pred = _to_numpy_1d(attack_output.membership_preds) if attack_output.membership_preds is not None else (y_score >= 0.5).astype(np.int64)
        return EvaluationResult(
            accuracy=float(accuracy_score(y_true, y_pred)),
            auroc=_safe_auroc(y_true, y_score),
            tpr_at_fpr={"1%": _tpr_at_fpr(y_true, y_score, 0.01), "0.1%": _tpr_at_fpr(y_true, y_score, 0.001)},
        )

    def run(self, attack_input: AttackInput) -> AttackOutput:
        self.fit(attack_input)
        output = self.infer(attack_input)
        if attack_input.membership_labels is not None:
            output.evaluation = self.evaluate(output, attack_input)
        return output


class TransferMLP(nn.Module):
    """Shadow model for the transfer attack, following the paper's design."""

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransferAttack(BaseAttack):
    """
    Transfer Attack: label-only MIA using a shadow (transfer) model.

    Required:
        - target_model: classification model
        - samples: target samples to attack
        - labels: ground-truth labels for the target samples
        - reference_data["aux_samples"], reference_data["aux_labels"]: data for training the transfer model

    The attack follows the Decision-based-MIA library's AdversaryOne approach:
    1. Train a shadow/transfer model on auxiliary data
    2. Query the target model to get predictions
    3. For samples where target prediction matches ground truth, use shadow model's loss
       For samples where target prediction differs, assign max loss (100)
    4. Score = -loss (higher = more likely member)
    """

    def __init__(self, batch_size: int = 128, transfer_epochs: int = 60,
                 transfer_lr: float = 1e-3, device: Optional[str] = None) -> None:
        self.batch_size = batch_size
        self.transfer_epochs = transfer_epochs
        self.transfer_lr = transfer_lr
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.shadow_model: Optional[nn.Module] = None
        self.num_classes: Optional[int] = None

    def fit(self, attack_input: AttackInput) -> "TransferAttack":
        ref = attack_input.reference_data
        if ref is None:
            raise ValueError("reference_data with aux_samples/aux_labels is required for TransferAttack.")

        aux_x = _to_np(ref["aux_samples"]).astype(np.float32)
        aux_y = _to_np(ref["aux_labels"]).astype(np.int64)
        input_dim = aux_x.shape[1]
        self.num_classes = int(attack_input.config.get("num_classes", aux_y.max() + 1))

        self.shadow_model = TransferMLP(input_dim, self.num_classes, 64).to(self.device)
        loader = DataLoader(TensorDataset(torch.tensor(aux_x), torch.tensor(aux_y)),
                           batch_size=self.batch_size, shuffle=True)
        opt = torch.optim.Adam(self.shadow_model.parameters(), lr=self.transfer_lr)
        crit = nn.CrossEntropyLoss()
        self.shadow_model.train()
        for _ in range(int(attack_input.config.get("transfer_epochs", self.transfer_epochs))):
            for bx, by in loader:
                opt.zero_grad()
                crit(self.shadow_model(bx.to(self.device)), by.to(self.device)).backward()
                opt.step()
        return self

    def infer(self, attack_input: AttackInput) -> AttackOutput:
        if self.shadow_model is None:
            raise RuntimeError("TransferAttack must be fitted before infer().")

        target_model = attack_input.target_model
        if target_model is None:
            raise ValueError("target_model is required")

        samples = _to_np(attack_input.samples).astype(np.float32)
        labels = _to_np(attack_input.labels).astype(np.int64)
        n = len(samples)

        # Get target model predictions
        target_model.eval()
        loader = DataLoader(TensorDataset(torch.tensor(samples)), batch_size=self.batch_size, shuffle=False)
        t_preds = []
        with torch.no_grad():
            for (bx,) in loader:
                t_preds.append(target_model(bx.to(self.device)).argmax(dim=1).detach().cpu())
        t_preds = torch.cat(t_preds).numpy()

        # Get shadow model predictions and compute losses
        self.shadow_model.eval()
        losses = np.zeros(n, dtype=np.float64)
        with torch.no_grad():
            for i, (bx,) in enumerate(DataLoader(
                TensorDataset(torch.tensor(samples)), batch_size=self.batch_size, shuffle=False
            )):
                start = i * self.batch_size
                end = start + bx.shape[0]
                bx, by = bx.to(self.device), torch.tensor(labels[start:end]).to(self.device)
                slogits = self.shadow_model(bx)

                for j in range(bx.shape[0]):
                    idx = start + j
                    if t_preds[idx] != labels[idx]:
                        losses[idx] = 100.0  # max loss for wrong predictions (paper convention)
                    else:
                        losses[idx] = F.cross_entropy(slogits[j:j+1], by[j:j+1]).item()

        scores = -losses  # higher = more likely member
        preds = (scores >= 0.5).astype(np.int64)

        return AttackOutput(
            membership_scores=scores, membership_preds=preds,
            intermediate_outputs={"losses": losses, "target_predictions": t_preds},
            metadata={"attack_name": "transfer_attack"},
        )


class BoundaryAttack(BaseAttack):
    """
    Boundary-Attack: Label-only MIA via decision-boundary distance.

    Wraps the Decision-based-MIA library's AdversaryTwo_HopSkipJump approach.

    Required:
        - target_model: classification model
        - samples: target samples to attack
        - labels: ground-truth labels

    The attack uses HopSkipJump to find the minimal perturbation to cross
    the decision boundary. Members are closer to the boundary -> smaller distance.

    Note: This attack requires the `foolbox` and `art` packages.
    """

    def __init__(self, batch_size: int = 1, max_iter: int = 50, max_eval: int = 10000,
                 device: Optional[str] = None) -> None:
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.max_eval = max_eval
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def infer(self, attack_input: AttackInput) -> AttackOutput:
        try:
            from art.estimators.classification import PyTorchClassifier
            from art.attacks.evasion import HopSkipJump
            from foolbox.distances import l2
        except ImportError as e:
            raise ImportError(
                "BoundaryAttack requires `foolbox` and `adversarial-robustness-toolbox`. "
                f"Install with: pip install foolbox adversarial-robustness-toolbox\nOriginal error: {e}"
            )

        target_model = attack_input.target_model
        if target_model is None:
            raise ValueError("target_model is required for BoundaryAttack")

        samples = _to_np(attack_input.samples).astype(np.float32)
        labels = _to_np(attack_input.labels).astype(np.int64)

        num_classes = int(attack_input.config.get("num_classes", int(labels.max()) + 1))
        input_shape = samples.shape[1:]
        if len(input_shape) == 1:
            input_shape = input_shape[0]

        art_classifier = PyTorchClassifier(
            model=target_model,
            clip_values=(float(samples.min()), float(samples.max())),
            loss=F.cross_entropy,
            input_shape=input_shape,
            nb_classes=num_classes,
        )

        attack = HopSkipJump(
            classifier=art_classifier,
            targeted=False,
            max_iter=int(attack_input.config.get("max_iter", self.max_iter)),
            max_eval=int(attack_input.config.get("max_eval", self.max_eval)),
        )

        distances = np.zeros(len(samples), dtype=np.float64)
        for i in range(len(samples)):
            x = samples[i:i+1]
            y = labels[i]
            logit = art_classifier.predict(x)
            pred = np.argmax(logit)
            if pred != y:
                distances[i] = 0.0
            else:
                try:
                    x_adv = attack.generate(x=np.array(x))
                    x_adv = np.array(x_adv)
                    distances[i] = float(l2(x, x_adv))
                except Exception:
                    distances[i] = float("inf")

        # Members are CLOSER to the boundary -> higher score = smaller distance
        max_finite = distances[distances < float("inf")].max() if np.any(distances < float("inf")) else 1.0
        scores = np.where(distances < float("inf"),
                         1.0 - distances / (max_finite + 1e-10),
                         0.0)
        preds = (scores >= 0.5).astype(np.int64)

        return AttackOutput(
            membership_scores=scores, membership_preds=preds,
            intermediate_outputs={"distances": distances},
            metadata={"attack_name": "boundary_attack", "max_iter": self.max_iter},
        )


# ============================================================================
# Helpers
# ============================================================================

def _to_numpy_1d(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy().reshape(-1) if isinstance(value, torch.Tensor) else np.asarray(value).reshape(-1)


def _to_np(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    return float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) >= 2 else None


def _tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, fpr_threshold: float) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.0
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(tpr[int(np.argmin(np.abs(fpr - fpr_threshold)))])


__all__ = [
    "AttackInput", "AttackOutput", "EvaluationResult", "BaseAttack",
    "TransferAttack", "BoundaryAttack",
]
