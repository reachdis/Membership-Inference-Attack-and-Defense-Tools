"""
Base defense abstractions for MIA defense implementations.

This module turns the design in `DEFENSE_INTERFACE_DESIGN_ZH.md` into concrete
Python dataclasses and an abstract base class that other defenses can inherit.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class DefenseInput:
    """Standardized input for all defense methods."""

    target_model: Optional[Any] = None
    model_factory: Optional[Any] = None

    train_data: Optional[Any] = None
    train_labels: Optional[Any] = None
    val_data: Optional[Any] = None
    val_labels: Optional[Any] = None
    test_data: Optional[Any] = None
    test_labels: Optional[Any] = None

    samples: Optional[Any] = None
    labels: Optional[Any] = None

    auxiliary_data: Optional[Dict[str, Any]] = None
    signals: Optional[Dict[str, Any]] = None
    defense_config: Dict[str, Any] = field(default_factory=dict)
    eval_config: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DefenseEvaluationResult:
    """Unified evaluation structure for defense outputs."""

    utility_metrics: Optional[Dict[str, float]] = None
    privacy_metrics: Optional[Dict[str, float]] = None
    efficiency_metrics: Optional[Dict[str, float]] = None
    extra_metrics: Optional[Dict[str, Any]] = None


@dataclass
class DefenseOutput:
    """Standardized output for all defense methods."""

    defended_model: Optional[Any] = None
    protected_predictor: Optional[Any] = None
    protected_outputs: Optional[Any] = None
    transformed_data: Optional[Any] = None

    artifacts: Optional[Dict[str, Any]] = None
    intermediate_outputs: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    evaluation: Optional[DefenseEvaluationResult] = None


class BaseDefense(ABC):
    """
    Abstract base class for defense implementations.

    Subclasses are expected to implement `infer`. Training-based defenses will
    usually override `fit`; inference-only defenses can keep the default no-op.
    """

    name: str = "base_defense"
    defense_family: str = "generic"
    defense_mode: str = "hybrid"
    supported_model_types: list[str] = []
    required_input_keys: list[str] = []
    optional_input_keys: list[str] = []

    def fit(self, defense_input: DefenseInput) -> "BaseDefense":
        return self

    @abstractmethod
    def infer(self, defense_input: DefenseInput) -> DefenseOutput:
        raise NotImplementedError

    def evaluate(
        self,
        defense_output: DefenseOutput,
        defense_input: DefenseInput,
    ) -> DefenseEvaluationResult:
        """
        Default evaluation focused on task utility only.

        Subclasses can override this when they want to add privacy-risk or
        efficiency evaluation. The default implementation computes accuracy if
        both a defended model and test labels are available.
        """
        utility_metrics: Dict[str, float] = {}
        if defense_output.protected_outputs is not None and defense_input.labels is not None:
            preds = _to_numpy_1d(defense_output.protected_outputs)
            labels = _to_numpy_1d(defense_input.labels)
            if preds.shape == labels.shape:
                utility_metrics["accuracy"] = _accuracy_from_predictions(preds, labels)

        return DefenseEvaluationResult(
            utility_metrics=utility_metrics or None,
            privacy_metrics=None,
            efficiency_metrics=None,
            extra_metrics=None,
        )

    def run(self, defense_input: DefenseInput) -> DefenseOutput:
        self.fit(defense_input)
        output = self.infer(defense_input)
        if defense_input.eval_config is not None:
            output.evaluation = self.evaluate(output, defense_input)
        return output


def _to_numpy_1d(value: Any) -> np.ndarray:
    if value is None:
        raise ValueError("Expected non-empty value for conversion to numpy.")
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().reshape(-1)
    return np.asarray(value).reshape(-1)


def _accuracy_from_predictions(preds: np.ndarray, labels: np.ndarray) -> float:
    preds = preds.astype(np.int64)
    labels = labels.astype(np.int64)
    return float(np.mean(preds == labels))


__all__ = [
    "DefenseInput",
    "DefenseOutput",
    "DefenseEvaluationResult",
    "BaseDefense",
]
