"""
Minimal DP-SGD defense example built on top of Defense/base.py.

This implementation is intentionally simple and self-contained:
- training-time defense
- PyTorch classification models
- per-sample gradient clipping
- Gaussian noise injection

It is designed as a concrete example of how to use the Defense base classes.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from Defense.base import BaseDefense, DefenseEvaluationResult, DefenseInput, DefenseOutput


class DPSGDDefense(BaseDefense):
    """Training-time DP-SGD defense for classification models."""

    name = "dp_sgd"
    defense_family = "differential_privacy"
    defense_mode = "training_time"
    supported_model_types = ["classifier"]
    required_input_keys = ["model_factory", "train_data", "train_labels"]
    optional_input_keys = ["val_data", "val_labels", "test_data", "test_labels", "samples", "labels"]

    def __init__(
        self,
        batch_size: int = 128,
        epochs: int = 20,
        learning_rate: float = 1e-3,
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
        device: Optional[str] = None,
    ) -> None:
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.defended_model: Optional[nn.Module] = None
        self.training_history: List[Dict[str, float]] = []
        self.last_runtime_seconds: Optional[float] = None

    def fit(self, defense_input: DefenseInput) -> "DPSGDDefense":
        if defense_input.model_factory is None:
            raise ValueError("DPSGDDefense requires defense_input.model_factory.")
        if defense_input.train_data is None or defense_input.train_labels is None:
            raise ValueError("DPSGDDefense requires train_data and train_labels.")

        config = defense_input.defense_config
        batch_size = int(config.get("batch_size", self.batch_size))
        epochs = int(config.get("epochs", self.epochs))
        learning_rate = float(config.get("learning_rate", self.learning_rate))
        noise_multiplier = float(config.get("noise_multiplier", self.noise_multiplier))
        max_grad_norm = float(config.get("max_grad_norm", self.max_grad_norm))

        model = defense_input.model_factory()
        if not isinstance(model, nn.Module):
            raise TypeError("model_factory must return a torch.nn.Module for DPSGDDefense.")
        model = model.to(self.device)

        train_loader = _build_loader(
            defense_input.train_data,
            defense_input.train_labels,
            batch_size=batch_size,
            shuffle=True,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss(reduction="none")

        self.training_history = []
        start_time = time.time()

        model.train()
        for epoch in range(epochs):
            epoch_loss_sum = 0.0
            epoch_example_count = 0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_size_actual = batch_x.shape[0]

                per_sample_losses = criterion(model(batch_x), batch_y)
                clipped_grads = self._compute_clipped_grads(
                    model=model,
                    per_sample_losses=per_sample_losses,
                    max_grad_norm=max_grad_norm,
                )
                self._apply_private_update(
                    model=model,
                    optimizer=optimizer,
                    clipped_grads=clipped_grads,
                    batch_size=batch_size_actual,
                    noise_multiplier=noise_multiplier,
                    max_grad_norm=max_grad_norm,
                )

                epoch_loss_sum += float(per_sample_losses.detach().sum().cpu().item())
                epoch_example_count += batch_size_actual

            self.training_history.append(
                {
                    "epoch": float(epoch + 1),
                    "train_loss": epoch_loss_sum / max(epoch_example_count, 1),
                }
            )

        self.last_runtime_seconds = time.time() - start_time
        self.defended_model = model.eval()
        return self

    def infer(self, defense_input: DefenseInput) -> DefenseOutput:
        if self.defended_model is None:
            raise RuntimeError("DPSGDDefense must be fitted before infer().")

        protected_outputs = None
        if defense_input.samples is not None:
            protected_outputs = self._predict_labels(self.defended_model, defense_input.samples)

        return DefenseOutput(
            defended_model=self.defended_model,
            protected_predictor=self.defended_model,
            protected_outputs=protected_outputs,
            artifacts={
                "training_history": self.training_history,
                "dp_config": {
                    "batch_size": self.batch_size,
                    "epochs": self.epochs,
                    "learning_rate": self.learning_rate,
                    "noise_multiplier": self.noise_multiplier,
                    "max_grad_norm": self.max_grad_norm,
                },
            },
            intermediate_outputs={
                "training_history": self.training_history,
            },
            metadata={
                "defense_name": self.name,
                "defense_family": self.defense_family,
                "defense_mode": self.defense_mode,
            },
        )

    def evaluate(
        self,
        defense_output: DefenseOutput,
        defense_input: DefenseInput,
    ) -> DefenseEvaluationResult:
        utility_metrics: Dict[str, float] = {}
        efficiency_metrics: Dict[str, float] = {}

        model = defense_output.defended_model
        if model is None:
            raise ValueError("No defended model available for evaluation.")

        if defense_input.test_data is not None and defense_input.test_labels is not None:
            test_preds = self._predict_labels(model, defense_input.test_data)
            test_labels = _to_numpy_1d(defense_input.test_labels)
            utility_metrics["test_accuracy"] = float(np.mean(test_preds == test_labels))

        if defense_input.train_data is not None and defense_input.train_labels is not None:
            train_preds = self._predict_labels(model, defense_input.train_data)
            train_labels = _to_numpy_1d(defense_input.train_labels)
            utility_metrics["train_accuracy"] = float(np.mean(train_preds == train_labels))

        if self.last_runtime_seconds is not None:
            efficiency_metrics["train_time"] = float(self.last_runtime_seconds)

        return DefenseEvaluationResult(
            utility_metrics=utility_metrics or None,
            privacy_metrics={
                "noise_multiplier": float(defense_input.defense_config.get("noise_multiplier", self.noise_multiplier)),
                "max_grad_norm": float(defense_input.defense_config.get("max_grad_norm", self.max_grad_norm)),
            },
            efficiency_metrics=efficiency_metrics or None,
            extra_metrics=None,
        )

    def _compute_clipped_grads(
        self,
        model: nn.Module,
        per_sample_losses: torch.Tensor,
        max_grad_norm: float,
    ) -> List[torch.Tensor]:
        params = [p for p in model.parameters() if p.requires_grad]
        accumulated_grads = [torch.zeros_like(param) for param in params]

        for loss in per_sample_losses:
            sample_grads = torch.autograd.grad(
                loss,
                params,
                retain_graph=True,
                allow_unused=False,
            )
            total_norm = torch.sqrt(
                sum(torch.sum(grad.detach() ** 2) for grad in sample_grads) + 1e-12 # type: ignore
            )
            clip_coef = min(1.0, max_grad_norm / float(total_norm.detach().cpu().item()))

            for i, grad in enumerate(sample_grads):
                accumulated_grads[i] += grad.detach() * clip_coef

        return accumulated_grads

    def _apply_private_update(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        clipped_grads: List[torch.Tensor],
        batch_size: int,
        noise_multiplier: float,
        max_grad_norm: float,
    ) -> None:
        optimizer.zero_grad()
        params = [p for p in model.parameters() if p.requires_grad]

        for param, grad_sum in zip(params, clipped_grads):
            noise = torch.normal(
                mean=0.0,
                std=noise_multiplier * max_grad_norm,
                size=grad_sum.shape,
                device=grad_sum.device,
            )
            private_grad = (grad_sum + noise) / float(batch_size)
            param.grad = private_grad

        optimizer.step()

    def _predict_labels(self, model: nn.Module, samples: Any) -> np.ndarray:
        model.eval()
        loader = _build_predict_loader(samples, batch_size=self.batch_size)
        all_preds: List[torch.Tensor] = []

        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                logits = model(batch_x)
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.detach().cpu())

        return torch.cat(all_preds, dim=0).numpy().astype(np.int64)


def _build_loader(
    samples: Any,
    labels: Any,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    x_tensor = _to_tensor_any(samples)
    y_tensor = _to_tensor_1d(labels, dtype=torch.long)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _build_predict_loader(samples: Any, batch_size: int) -> DataLoader:
    x_tensor = _to_tensor_any(samples)
    dataset = TensorDataset(x_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _to_tensor_any(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(torch.float32)
    return torch.as_tensor(value, dtype=torch.float32)


def _to_tensor_1d(value: Any, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    tensor = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    tensor = tensor.reshape(-1)
    if dtype is not None:
        tensor = tensor.to(dtype)
    return tensor


def _to_numpy_1d(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().reshape(-1)
    return np.asarray(value).reshape(-1)


__all__ = [
    "DPSGDDefense",
]
