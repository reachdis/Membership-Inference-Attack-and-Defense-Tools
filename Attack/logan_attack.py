"""
LOGAN Attack: Discriminator-confidence based MIA for generative models.

Wraps the gen_mem_inf library (Hayes et al., PoPETs 2019).
https://github.com/jhayes14/gen_mem_inf

Attack principle:
- White-box: Use the target GAN's discriminator confidence as membership score.
  Members (training images) get higher discriminator output than non-members.
- Black-box: Train a surrogate GAN on the target generator's outputs,
  then use the surrogate discriminator.

Unified convention:
    higher discriminator confidence -> more likely member
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, TensorDataset

_LIB_DIR = Path(__file__).resolve().parent / "gen_mem_inf"
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
        y_pred = (_to_numpy_1d(attack_output.membership_preds) if attack_output.membership_preds is not None
                   else (y_score >= 0.5).astype(np.int64))
        return EvaluationResult(
            accuracy=float(accuracy_score(y_true, y_pred)),
            auroc=_safe_auroc(y_true, y_score),
            tpr_at_fpr={"1%": _tpr_at_fpr(y_true, y_score, 0.01),
                         "0.1%": _tpr_at_fpr(y_true, y_score, 0.001)},
        )

    def run(self, attack_input: AttackInput) -> AttackOutput:
        self.fit(attack_input)
        output = self.infer(attack_input)
        if attack_input.membership_labels is not None:
            output.evaluation = self.evaluate(output, attack_input)
        return output


class LOGANAttack(BaseAttack):
    """
    LOGAN: Membership inference against generative models via discriminator confidence.

    Two modes:
    - "white_box": target_model is the target discriminator (netBBD from the library)
    - "black_box": train a surrogate discriminator on target generator outputs

    Config options:
        attack_mode: "white_box" | "black_box"
        nz, nc, ngf, ndf, ngpu: GAN architecture params
        surrogate_niter, surrogate_lr: surrogate training params
    """

    def __init__(self, batch_size: int = 64, device: Optional[str] = None) -> None:
        self.batch_size = batch_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._surrogate_disc: Optional[nn.Module] = None

    def fit(self, attack_input: AttackInput) -> "LOGANAttack":
        mode = attack_input.config.get("attack_mode", "white_box")
        if mode == "black_box":
            shadow = attack_input.shadow_data
            if shadow is None:
                raise ValueError("shadow_data required for black-box LOGAN")
            if "surrogate_discriminator" in shadow:
                self._surrogate_disc = shadow["surrogate_discriminator"]
            elif "target_generator" in shadow:
                self._surrogate_disc = self._train_surrogate(attack_input)
            else:
                raise ValueError("Need target_generator or surrogate_discriminator in shadow_data")
        return self

    def infer(self, attack_input: AttackInput) -> AttackOutput:
        mode = attack_input.config.get("attack_mode", "white_box")
        if mode == "white_box":
            discriminator = attack_input.target_model
            if discriminator is None:
                raise ValueError("target_model (discriminator) required for white-box LOGAN")
        else:
            discriminator = self._surrogate_disc
            if discriminator is None:
                raise RuntimeError("Surrogate discriminator not fitted. Call fit() first.")

        scores = self._get_discriminator_scores(discriminator.infer, attack_input.samples)
        preds = (scores >= 0.5).astype(np.int64)
        return AttackOutput(
            membership_scores=scores, membership_preds=preds,
            intermediate_outputs={"discriminator_confidences": scores},
            metadata={"attack_name": "logan", "attack_mode": mode},
        )

    def _get_discriminator_scores(self, model_fn, samples: Any) -> np.ndarray:
        loader = DataLoader(TensorDataset(_to_tensor(samples)), batch_size=self.batch_size, shuffle=False)
        all_scores = []
        with torch.no_grad():
            for (bx,) in loader:
                all_scores.append(model_fn(bx.to(self.device)).detach().cpu())
        return torch.cat(all_scores).numpy().astype(np.float64)

    def _train_surrogate(self, attack_input: AttackInput) -> nn.Module:
        from models import Discriminator, Generator, weights_init

        target_gen = attack_input.shadow_data["target_generator"]
        nz = int(attack_input.config.get("nz", 100))
        nc = int(attack_input.config.get("nc", 3))
        ngf = int(attack_input.config.get("ngf", 64))
        ndf = int(attack_input.config.get("ndf", 64))
        ngpu = int(attack_input.config.get("ngpu", 1))
        n_iter = int(attack_input.config.get("surrogate_niter", 100))
        lr = float(attack_input.config.get("surrogate_lr", 0.0002))
        beta1 = float(attack_input.config.get("surrogate_beta1", 0.5))

        netG = Generator(ngpu, nz, ngf, nc).to(self.device)
        netG.apply(weights_init)
        netD = Discriminator(ngpu, nc, ndf).to(self.device)
        netD.apply(weights_init)

        crit = nn.BCELoss()
        optD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
        optG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

        target_gen.eval()
        bs = int(attack_input.config.get("surrogate_batch_size", 64))

        for _ in range(n_iter):
            noise = torch.randn(bs, nz, 1, 1, device=self.device)
            with torch.no_grad():
                real = target_gen(noise)

            # Train D
            netD.zero_grad()
            label = torch.full((bs,), 1.0, device=self.device)
            errD_real = crit(netD(real), label)
            errD_real.backward()
            fake = netG(torch.randn(bs, nz, 1, 1, device=self.device))
            label.fill_(0.0)
            errD_fake = crit(netD(fake.detach()), label)
            errD_fake.backward()
            optD.step()

            # Train G
            netG.zero_grad()
            label.fill_(1.0)
            errG = crit(netD(fake), label)
            errG.backward()
            optG.step()

        netD.eval()
        return netD


# ============================================================================
# Helpers
# ============================================================================

def _to_tensor(value: Any) -> torch.Tensor:
    return value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=torch.float32)


def _to_numpy_1d(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy().reshape(-1) if isinstance(value, torch.Tensor) else np.asarray(value).reshape(-1)


def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    return float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) >= 2 else None


def _tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, fpr_threshold: float) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.0
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(tpr[int(np.argmin(np.abs(fpr - fpr_threshold)))])


__all__ = [
    "AttackInput", "AttackOutput", "EvaluationResult", "BaseAttack", "LOGANAttack",
]
