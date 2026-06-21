"""
Utilities for RAPID reference-model training and difficulty calibration.

It reimplements the reference-model training pattern and the score computation
needed by RAPID.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


class RAPIDReferenceManager:
    """
    Train and manage reference models for RAPID.

    RAPID uses reference models to estimate a sample's difficulty:
        calibrated_score = original_score - mean_ref_score
    """

    def __init__(
        self,
        train_X: np.ndarray,
        train_y: np.ndarray,
        test_X: np.ndarray,
        test_y: np.ndarray,
        model_factory: Callable[[], nn.Module],
        train_fn: Optional[Callable[[nn.Module, DataLoader, torch.device, Dict[str, Any]], None]] = None,
        train_config: Optional[Dict[str, Any]] = None,
        batch_size: int = 128,
        device: Optional[str] = None,
    ) -> None:
        self.train_X = np.asarray(train_X)
        self.train_y = np.asarray(train_y)
        self.test_X = np.asarray(test_X)
        self.test_y = np.asarray(test_y)
        self.model_factory = model_factory
        self.train_fn = train_fn or self._default_train_fn
        self.train_config = train_config or {}
        self.batch_size = batch_size
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.reference_models: List[nn.Module] = []

    def train_reference_models(
        self,
        data_sizes: List[int],
        random_seed_num: int = 5,
        reference_model_number: int = 10,
    ) -> None:
        """
        Train reference models using the same overall pattern as the reference file:
        1. Sample from train/test pools.
        2. Merge sampled data.
        3. Randomly split into halves.
        4. Train two reciprocal models per split.
        """
        if reference_model_number < 2:
            raise ValueError("reference_model_number must be at least 2.")

        self.reference_models = []

        for data_size in data_sizes:
            for seed_offset in range(random_seed_num):
                seed = data_size + seed_offset
                self._set_seed(seed)

                train_sample_X, train_sample_y = self._sample_data(
                    self.train_X, self.train_y, data_size, seed
                )
                test_sample_X, test_sample_y = self._sample_data(
                    self.test_X, self.test_y, data_size, seed
                )

                merged_X = np.concatenate([train_sample_X, test_sample_X], axis=0)
                merged_y = np.concatenate([train_sample_y, test_sample_y], axis=0)

                for split_num in range(reference_model_number // 2):
                    split_seed = seed + split_num
                    x_a, x_b, y_a, y_b = train_test_split(
                        merged_X,
                        merged_y,
                        test_size=0.5,
                        random_state=split_seed,
                    )

                    self.reference_models.append(self._train_single_model(x_a, y_a))
                    self.reference_models.append(self._train_single_model(x_b, y_b))

    def compute_original_scores(
        self,
        model: nn.Module,
        samples: np.ndarray | torch.Tensor,
        labels: np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        """
        Compute the original membership score.

        Here we use the negative per-sample cross-entropy loss so that the global
        project convention is preserved:
            higher score -> more likely member
        """
        return compute_model_original_scores(
            model=model,
            samples=samples,
            labels=labels,
            batch_size=self.batch_size,
            device=self.device,
        )

    def compute_reference_mean_scores(
        self,
        samples: np.ndarray | torch.Tensor,
        labels: np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        """Average the original score over all trained reference models."""
        if not self.reference_models:
            raise RuntimeError("Reference models have not been trained yet.")

        all_scores = []
        for model in self.reference_models:
            scores = compute_model_original_scores(
                model=model,
                samples=samples,
                labels=labels,
                batch_size=self.batch_size,
                device=self.device,
            )
            all_scores.append(scores)

        stacked_scores = np.stack(all_scores, axis=0)
        return np.mean(stacked_scores, axis=0)

    def compute_calibrated_scores(
        self,
        original_scores: np.ndarray,
        samples: np.ndarray | torch.Tensor,
        labels: np.ndarray | torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute difficulty-calibrated scores for RAPID.

        Returns:
            calibrated_scores, reference_mean_scores
        """
        reference_mean_scores = self.compute_reference_mean_scores(samples=samples, labels=labels)
        calibrated_scores = np.asarray(original_scores) - reference_mean_scores
        return calibrated_scores, reference_mean_scores

    def _train_single_model(self, train_X: np.ndarray, train_y: np.ndarray) -> nn.Module:
        model = self.model_factory().to(self.device)
        train_loader = self._build_loader(train_X, train_y, shuffle=True)
        self.train_fn(model, train_loader, self.device, self.train_config)
        model.eval()
        return model

    def _sample_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        size: int,
        seed: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if size > len(X):
            raise ValueError(f"Requested sample size {size} exceeds dataset size {len(X)}.")
        if size == len(X):
            return X.copy(), y.copy()

        _, X_sample, _, y_sample = train_test_split(
            X,
            y,
            test_size=size,
            random_state=seed,
        )
        return X_sample, y_sample

    def _build_loader(
        self,
        samples: np.ndarray,
        labels: np.ndarray,
        shuffle: bool,
    ) -> DataLoader:
        x_tensor = torch.tensor(samples, dtype=torch.float32)
        y_tensor = torch.tensor(labels, dtype=torch.long)
        dataset = TensorDataset(x_tensor, y_tensor)
        return DataLoader(dataset, batch_size=min(self.batch_size, len(dataset)), shuffle=shuffle)

    def _set_seed(self, seed: int) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _default_train_fn(
        model: nn.Module,
        train_loader: DataLoader,
        device: torch.device,
        train_config: Dict[str, Any],
    ) -> None:
        epochs = int(train_config.get("epochs", 20))
        lr = float(train_config.get("lr", 1e-3))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for _ in range(epochs):
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()


def compute_model_original_scores(
    model: nn.Module,
    samples: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Compute negative per-sample cross-entropy loss."""
    loader = _build_loader(samples=samples, labels=labels, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss(reduction="none")
    all_scores = []

    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            losses = criterion(logits, batch_y)
            scores = (-losses).detach().cpu().numpy()
            all_scores.append(scores)

    return np.concatenate(all_scores, axis=0).astype(np.float64)


def _build_loader(
    samples: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    x_tensor = _to_tensor_any(samples)
    y_tensor = _to_tensor_1d(labels, dtype=torch.long)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _to_tensor_any(value: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(torch.float32)
    return torch.as_tensor(value, dtype=torch.float32)


def _to_tensor_1d(value: np.ndarray | torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    tensor = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    tensor = tensor.reshape(-1)
    if dtype is not None:
        tensor = tensor.to(dtype)
    return tensor
