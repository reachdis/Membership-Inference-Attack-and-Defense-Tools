"""
DL-MIA adapter compatible with the project's AttackInput/AttackOutput interface.

This wrapper captures the stable attack head used by DL-MIA:

1. Build three feature branches for each attacked sample:
   - vector branch   : original difference-vector style attack feature
   - semantic branch : recommender-invariant / disentangled semantic feature
   - syntax branch   : recommender-specific / disentangled syntax feature
2. Train one MLP per branch.
3. Fuse the three branch outputs with the paper-style weighted combination:
      alpha * semantic + beta * syntax + (1 - alpha - beta) * vector
4. Return the member probability as `membership_scores`.

This wrapper is intentionally feature-level. It does not re-run the full
joint-training pipeline from `DL-MIA-KDD-2022-main`. Instead, it exposes a
clean interface for plugging precomputed DL-MIA features into the unified
attack API.

Required training input
-----------------------
`attack_input.shadow_data` must provide one of:

1. Explicit member/non-member branch features:
   {
       "member_features": {
           "vector": ndarray or Tensor,    # [N_member, D1]
           "semantic": ndarray or Tensor,  # [N_member, D2]
           "syntax": ndarray or Tensor,    # [N_member, D3]
       },
       "nonmember_features": {
           "vector": ndarray or Tensor,    # [N_nonmember, D1]
           "semantic": ndarray or Tensor,  # [N_nonmember, D2]
           "syntax": ndarray or Tensor,    # [N_nonmember, D3]
       },
   }

2. A combined branch feature matrix with labels:
   {
       "features": {
           "vector": ndarray or Tensor,    # [N, D1]
           "semantic": ndarray or Tensor,  # [N, D2]
           "syntax": ndarray or Tensor,    # [N, D3]
       },
       "membership_labels": ndarray or Tensor,  # [N]
   }

3. Vector-only attack features:
   {
       "member_vectors": ndarray or Tensor,
       "nonmember_vectors": ndarray or Tensor,
   }
   or
   {
       "vectors": ndarray or Tensor,
       "membership_labels": ndarray or Tensor,
   }

4. Raw recommender inputs:
   {
       "member_interactions": {user_id: [item_id, ...], ...},
       "nonmember_interactions": {user_id: [item_id, ...], ...},
       "member_recommendations": {user_id: [item_id, ...], ...},
       "nonmember_recommendations": {user_id: [item_id, ...], ...},
       "item_embeddings": {item_id: embedding_vector, ...},
   }

Inference input
---------------
The same feature payload can be provided through `attack_input.signals` or
`attack_input.samples`.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from Attack.shadow_based import AttackInput, AttackOutput, BaseAttack


FeatureBundle = Dict[str, np.ndarray]


@dataclass
class _PreparedDLMIAQuery:
    features: FeatureBundle
    membership_labels: Optional[np.ndarray]
    has_explicit_partition: bool


class _DLMIABranchMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Tuple[int, int] = (32, 8)) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class _DLMIAEnsemble(nn.Module):
    def __init__(
        self,
        vector_dim: int,
        semantic_dim: int,
        syntax_dim: int,
        hidden_dims: Tuple[int, int],
        alpha: float,
        beta: float,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.vector_head = _DLMIABranchMLP(vector_dim, hidden_dims)
        self.semantic_head = _DLMIABranchMLP(semantic_dim, hidden_dims)
        self.syntax_head = _DLMIABranchMLP(syntax_dim, hidden_dims)

    def forward(
        self,
        vector_x: torch.Tensor,
        semantic_x: torch.Tensor,
        syntax_x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        vector_logits = self.vector_head(vector_x)
        semantic_logits = self.semantic_head(semantic_x)
        syntax_logits = self.syntax_head(syntax_x)
        fused_logits = (
            self.alpha * semantic_logits
            + self.beta * syntax_logits
            + (1.0 - self.alpha - self.beta) * vector_logits
        )
        return fused_logits, vector_logits, semantic_logits, syntax_logits


class DLMIAAttack(BaseAttack):
    """
    Interface-compatible wrapper for DL-MIA branch fusion.

    Required:
        - shadow_data with vector / semantic / syntax features

    Optional:
        - membership_labels for evaluation
        - signals containing branch features for inference
        - samples containing branch features for inference
        - config for runtime hyperparameters
    """

    def __init__(
        self,
        hidden_dims: Tuple[int, int] = (32, 8),
        batch_size: int = 256,
        lr: float = 1e-3,
        epochs: int = 50,
        alpha: float = 0.1,
        beta: float = 0.0,
        branch_loss_weights: Optional[Tuple[float, float, float]] = None,
        device: Optional[str] = None,
    ) -> None:
        self.hidden_dims = hidden_dims
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.alpha = alpha
        self.beta = beta
        self.branch_loss_weights = branch_loss_weights
        self.device = torch.device(device if device is not None else "cpu")
        self.threshold = 0.5

        self.model: Optional[_DLMIAEnsemble] = None
        self.feature_dims: Optional[Tuple[int, int, int]] = None
        self.runtime_config: Dict[str, Any] = {}
        self.vector_mean: Optional[np.ndarray] = None
        self.semantic_basis: Optional[np.ndarray] = None
        self.syntax_basis: Optional[np.ndarray] = None
        self.is_fitted = False

    def fit(self, attack_input: AttackInput) -> "DLMIAAttack":
        if attack_input.shadow_data is None:
            raise ValueError("shadow_data is required for DLMIAAttack.fit().")

        self.runtime_config = self._merge_config(attack_input.config)
        feature_bundle, y_train = self._resolve_train_data(
            attack_input.shadow_data,
            model=attack_input.shadow_data.get("shadow_model"),
        )
        vector_x = feature_bundle["vector"]
        semantic_x = feature_bundle["semantic"]
        syntax_x = feature_bundle["syntax"]
        self.feature_dims = (vector_x.shape[1], semantic_x.shape[1], syntax_x.shape[1])

        self.model = _DLMIAEnsemble(
            vector_dim=self.feature_dims[0],
            semantic_dim=self.feature_dims[1],
            syntax_dim=self.feature_dims[2],
            hidden_dims=tuple(self.runtime_config["hidden_dims"]),
            alpha=float(self.runtime_config["alpha"]),
            beta=float(self.runtime_config["beta"]),
        ).to(self.device)

        dataset = TensorDataset(
            torch.tensor(vector_x, dtype=torch.float32),
            torch.tensor(semantic_x, dtype=torch.float32),
            torch.tensor(syntax_x, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        loader = DataLoader(dataset, batch_size=self.runtime_config["batch_size"], shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.runtime_config["lr"])

        branch_loss_weights = self.runtime_config["branch_loss_weights"]
        if branch_loss_weights is None:
            branch_loss_weights = (
                1.0 - self.runtime_config["alpha"] - self.runtime_config["beta"],
                self.runtime_config["alpha"],
                self.runtime_config["beta"],
            )

        self.model.train()
        for _ in range(self.runtime_config["epochs"]):
            for batch_vector, batch_semantic, batch_syntax, batch_y in loader:
                batch_vector = batch_vector.to(self.device)
                batch_semantic = batch_semantic.to(self.device)
                batch_syntax = batch_syntax.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                fused_logits, vector_logits, semantic_logits, syntax_logits = self.model(
                    batch_vector,
                    batch_semantic,
                    batch_syntax,
                )
                fused_loss = criterion(fused_logits, batch_y)
                vector_loss = criterion(vector_logits, batch_y)
                semantic_loss = criterion(semantic_logits, batch_y)
                syntax_loss = criterion(syntax_logits, batch_y)

                total_loss = (
                    fused_loss
                    + branch_loss_weights[0] * vector_loss
                    + branch_loss_weights[1] * semantic_loss
                    + branch_loss_weights[2] * syntax_loss
                )
                total_loss.backward()
                optimizer.step()

        self.is_fitted = True
        return self

    def infer(self, attack_input: AttackInput) -> AttackOutput:
        if not self.is_fitted or self.model is None or self.feature_dims is None:
            raise RuntimeError("DLMIAAttack must be fitted before infer().")

        prepared = self._resolve_query_data(attack_input)
        self._check_feature_dims(prepared.features)

        with torch.no_grad():
            fused_logits, vector_logits, semantic_logits, syntax_logits = self.model(
                torch.tensor(prepared.features["vector"], dtype=torch.float32, device=self.device),
                torch.tensor(prepared.features["semantic"], dtype=torch.float32, device=self.device),
                torch.tensor(prepared.features["syntax"], dtype=torch.float32, device=self.device),
            )
            fused_probs = torch.softmax(fused_logits, dim=1).detach().cpu().numpy()
            vector_probs = torch.softmax(vector_logits, dim=1).detach().cpu().numpy()
            semantic_probs = torch.softmax(semantic_logits, dim=1).detach().cpu().numpy()
            syntax_probs = torch.softmax(syntax_logits, dim=1).detach().cpu().numpy()

        scores = fused_probs[:, 1].astype(np.float32)
        threshold = float(self.runtime_config.get("threshold", self.threshold))
        preds = (scores >= threshold).astype(np.int64)

        return AttackOutput(
            membership_scores=scores,
            membership_preds=preds,
            intermediate_outputs={
                "vector_branch_member_probs": vector_probs[:, 1].astype(np.float32),
                "semantic_branch_member_probs": semantic_probs[:, 1].astype(np.float32),
                "syntax_branch_member_probs": syntax_probs[:, 1].astype(np.float32),
                "fused_probabilities": fused_probs.astype(np.float32),
            },
            metadata={
                "attack_name": "dl_mia",
                "domain": "recommender_system",
                "score_direction": "higher_is_member",
                "alpha": self.runtime_config["alpha"],
                "beta": self.runtime_config["beta"],
                "threshold": threshold,
                "has_explicit_partition": prepared.has_explicit_partition,
            },
        )

    def _resolve_train_data(
        self,
        shadow_data: Dict[str, Any],
        model: Optional[Any] = None,
    ) -> Tuple[FeatureBundle, np.ndarray]:
        features, labels, _ = self._coerce_training_payload(
            shadow_data,
            context="shadow_data",
            model=model,
        )
        if labels is None:
            raise ValueError("Training labels are required for DL-MIA fit().")
        return features, labels

    def _merge_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        merged = {
            "hidden_dims": list(self.hidden_dims),
            "batch_size": self.batch_size,
            "lr": self.lr,
            "epochs": self.epochs,
            "alpha": self.alpha,
            "beta": self.beta,
            "branch_loss_weights": self.branch_loss_weights,
            "threshold": self.threshold,
            "semantic_dim": 50,
            "syntax_dim": 50,
        }
        merged.update(config)
        return merged

    def _resolve_query_data(self, attack_input: AttackInput) -> _PreparedDLMIAQuery:
        if attack_input.signals is not None and self._contains_supported_payload(attack_input.signals):
            features, labels, explicit = self._coerce_inference_payload(
                attack_input.signals,
                context="signals",
                allow_missing_labels=True,
                model=attack_input.target_model,
            )
            return _PreparedDLMIAQuery(features, labels, explicit)

        if isinstance(attack_input.samples, dict) and self._contains_supported_payload(attack_input.samples):
            features, labels, explicit = self._coerce_inference_payload(
                attack_input.samples,
                context="samples",
                allow_missing_labels=True,
                model=attack_input.target_model,
            )
            return _PreparedDLMIAQuery(features, labels, explicit)

        raise ValueError(
            "Unable to resolve DL-MIA query data. Provide vector/semantic/syntax "
            "features through signals or samples dict."
        )

    def _contains_feature_payload(self, payload: Dict[str, Any]) -> bool:
        return any(key in payload for key in ("member_features", "nonmember_features", "features"))

    def _contains_vector_payload(self, payload: Dict[str, Any]) -> bool:
        return any(key in payload for key in ("member_vectors", "nonmember_vectors", "vectors"))

    def _contains_raw_partition_payload(self, payload: Dict[str, Any]) -> bool:
        return all(
            key in payload
            for key in (
                "member_interactions",
                "nonmember_interactions",
            )
        )

    def _contains_raw_combined_payload(self, payload: Dict[str, Any]) -> bool:
        return "interactions" in payload

    def _contains_supported_payload(self, payload: Dict[str, Any]) -> bool:
        return (
            self._contains_feature_payload(payload)
            or self._contains_vector_payload(payload)
            or self._contains_raw_partition_payload(payload)
            or self._contains_raw_combined_payload(payload)
        )

    def _coerce_training_payload(
        self,
        payload: Dict[str, Any],
        context: str,
        model: Optional[Any] = None,
    ) -> Tuple[FeatureBundle, Optional[np.ndarray], bool]:
        if self._contains_feature_payload(payload):
            return self._coerce_feature_payload(payload, context=context)

        vectors, labels, explicit = self._coerce_vector_or_raw_payload(
            payload,
            context=context,
            allow_missing_labels=False,
            model=model,
        )
        features = self._materialize_feature_bundle(
            payload,
            vectors,
            context=context,
            model=model,
            fit_projector=True,
        )
        return features, labels, explicit

    def _coerce_inference_payload(
        self,
        payload: Dict[str, Any],
        context: str,
        allow_missing_labels: bool = False,
        model: Optional[Any] = None,
    ) -> Tuple[FeatureBundle, Optional[np.ndarray], bool]:
        if self._contains_feature_payload(payload):
            return self._coerce_feature_payload(
                payload,
                context=context,
                allow_missing_labels=allow_missing_labels,
            )

        vectors, labels, explicit = self._coerce_vector_or_raw_payload(
            payload,
            context=context,
            allow_missing_labels=allow_missing_labels,
            model=model,
        )
        features = self._materialize_feature_bundle(
            payload,
            vectors,
            context=context,
            model=model,
            fit_projector=False,
        )
        return features, labels, explicit

    def _coerce_feature_payload(
        self,
        payload: Dict[str, Any],
        context: str,
        allow_missing_labels: bool = False,
    ) -> Tuple[FeatureBundle, Optional[np.ndarray], bool]:
        if "member_features" in payload and "nonmember_features" in payload:
            member_features = self._normalize_feature_bundle(payload["member_features"], f"{context}['member_features']")
            nonmember_features = self._normalize_feature_bundle(payload["nonmember_features"], f"{context}['nonmember_features']")

            features = {
                name: np.concatenate([member_features[name], nonmember_features[name]], axis=0).astype(np.float32)
                for name in ("vector", "semantic", "syntax")
            }
            labels = np.concatenate(
                [
                    np.ones(member_features["vector"].shape[0], dtype=np.int64),
                    np.zeros(nonmember_features["vector"].shape[0], dtype=np.int64),
                ],
                axis=0,
            )
            return features, labels, True

        if "features" in payload:
            features = self._normalize_feature_bundle(payload["features"], f"{context}['features']")
            if "membership_labels" in payload:
                labels = self._to_numpy_1d(payload["membership_labels"]).astype(np.int64)
                self._validate_label_count(
                    labels,
                    features["vector"].shape[0],
                    f"{context}['membership_labels']",
                )
                return features, labels, False
            if allow_missing_labels:
                return features, None, False
            raise ValueError(f"{context}['membership_labels'] is required when using combined features.")

        raise ValueError(
            f"{context} must provide either member/nonmember branch features or "
            "combined branch features with membership_labels."
        )

    def _coerce_vector_or_raw_payload(
        self,
        payload: Dict[str, Any],
        context: str,
        allow_missing_labels: bool = False,
        model: Optional[Any] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], bool]:
        if "member_vectors" in payload and "nonmember_vectors" in payload:
            member_vectors = self._to_numpy_2d(payload["member_vectors"]).astype(np.float32)
            nonmember_vectors = self._to_numpy_2d(payload["nonmember_vectors"]).astype(np.float32)
            vectors = np.concatenate([member_vectors, nonmember_vectors], axis=0)
            labels = np.concatenate(
                [
                    np.ones(member_vectors.shape[0], dtype=np.int64),
                    np.zeros(nonmember_vectors.shape[0], dtype=np.int64),
                ],
                axis=0,
            )
            return vectors, labels, True

        if self._contains_raw_partition_payload(payload):
            item_embeddings = self._resolve_item_embeddings(payload, model, context)
            member_recommendations = self._resolve_partition_recommendations(
                payload,
                "member",
                model,
                context,
            )
            nonmember_recommendations = self._resolve_partition_recommendations(
                payload,
                "nonmember",
                model,
                context,
            )
            member_vectors = self._difference_vectors_from_partition(
                interactions=payload["member_interactions"],
                recommendations=member_recommendations,
                item_embeddings=item_embeddings,
                context=f"{context} member partition",
            )
            nonmember_vectors = self._difference_vectors_from_partition(
                interactions=payload["nonmember_interactions"],
                recommendations=nonmember_recommendations,
                item_embeddings=item_embeddings,
                context=f"{context} nonmember partition",
            )
            vectors = np.concatenate([member_vectors, nonmember_vectors], axis=0).astype(np.float32)
            labels = np.concatenate(
                [
                    np.ones(member_vectors.shape[0], dtype=np.int64),
                    np.zeros(nonmember_vectors.shape[0], dtype=np.int64),
                ],
                axis=0,
            )
            return vectors, labels, True

        if "vectors" in payload:
            vectors = self._to_numpy_2d(payload["vectors"]).astype(np.float32)
            if "membership_labels" in payload:
                labels = self._to_numpy_1d(payload["membership_labels"]).astype(np.int64)
                self._validate_label_count(labels, vectors.shape[0], f"{context}['membership_labels']")
                return vectors, labels, False
            if allow_missing_labels:
                return vectors, None, False
            raise ValueError(f"{context}['membership_labels'] is required when using combined vectors.")

        if self._contains_raw_combined_payload(payload):
            item_embeddings = self._resolve_item_embeddings(payload, model, context)
            recommendations = self._resolve_combined_recommendations(payload, model, context)
            vectors = self._difference_vectors_from_partition(
                interactions=payload["interactions"],
                recommendations=recommendations,
                item_embeddings=item_embeddings,
                context=f"{context} combined raw payload",
            )
            if "membership_labels" in payload:
                labels = self._coerce_raw_membership_labels(
                    payload["membership_labels"],
                    expected_size=vectors.shape[0],
                    sample_keys=self._raw_sequence_keys(payload["interactions"]),
                )
                return vectors, labels, False
            if allow_missing_labels:
                return vectors, None, False
            raise ValueError(f"{context}['membership_labels'] is required when using combined raw inputs.")

        raise ValueError(
            f"{context} must provide branch features, vectors, or raw recommender inputs."
        )

    def _materialize_feature_bundle(
        self,
        payload: Dict[str, Any],
        vectors: np.ndarray,
        context: str,
        model: Optional[Any],
        fit_projector: bool,
    ) -> FeatureBundle:
        callback = payload.get("feature_builder") or payload.get("feature_extractor")
        if callback is None:
            callback = self.runtime_config.get("feature_builder") or self.runtime_config.get("feature_extractor")
        if callback is not None:
            built = self._invoke_callback(
                callback,
                vectors=vectors,
                payload=payload,
                model=model,
                context=context,
                config=self.runtime_config,
                fit_projector=fit_projector,
                attack=self,
            )
            return self._normalize_feature_bundle(built, f"{context} callback features")

        return self._build_feature_bundle_from_vectors(vectors, fit_projector=fit_projector)

    def _resolve_partition_recommendations(
        self,
        payload: Dict[str, Any],
        split: str,
        model: Optional[Any],
        context: str,
    ) -> Any:
        key = f"{split}_recommendations"
        if key in payload:
            return payload[key]
        interaction_key = f"{split}_interactions"
        return self._resolve_recommendations_via_callback(
            interactions=payload[interaction_key],
            payload=payload,
            model=model,
            context=f"{context} {split} partition",
        )

    def _resolve_combined_recommendations(
        self,
        payload: Dict[str, Any],
        model: Optional[Any],
        context: str,
    ) -> Any:
        if "recommendations" in payload:
            return payload["recommendations"]
        return self._resolve_recommendations_via_callback(
            interactions=payload["interactions"],
            payload=payload,
            model=model,
            context=f"{context} combined payload",
        )

    def _resolve_recommendations_via_callback(
        self,
        interactions: Any,
        payload: Dict[str, Any],
        model: Optional[Any],
        context: str,
    ) -> Any:
        callback = payload.get("recommend_fn") or self.runtime_config.get("recommend_fn")
        if callback is None:
            raise ValueError(
                f"{context} is missing recommendations. Provide explicit recommendations or "
                "set config['recommend_fn'] / payload['recommend_fn'] to generate them."
            )
        return self._invoke_callback(
            callback,
            model=model,
            interactions=interactions,
            payload=payload,
            context=context,
            config=self.runtime_config,
            attack=self,
        )

    def _resolve_item_embeddings(
        self,
        payload: Dict[str, Any],
        model: Optional[Any],
        context: str,
    ) -> Dict[Any, np.ndarray]:
        if "item_embeddings" in payload:
            return self._normalize_item_embeddings(payload["item_embeddings"])

        callback = payload.get("item_embedding_fn") or self.runtime_config.get("item_embedding_fn")
        if callback is not None:
            resolved = self._invoke_callback(
                callback,
                model=model,
                payload=payload,
                context=context,
                config=self.runtime_config,
                attack=self,
            )
            return self._normalize_item_embeddings(resolved)

        if model is not None:
            resolved = self._extract_item_embeddings_from_model(model)
            if resolved is not None:
                return resolved

        raise ValueError(
            f"{context} is missing item embeddings. Provide payload['item_embeddings'], "
            "config['item_embedding_fn'], or a target/shadow model with accessible item embeddings."
        )

    def _build_feature_bundle_from_vectors(self, vectors: np.ndarray, fit_projector: bool) -> FeatureBundle:
        vectors = vectors.astype(np.float32)
        if fit_projector:
            self._fit_branch_projector(vectors)

        if self.vector_mean is None or self.semantic_basis is None or self.syntax_basis is None:
            raise RuntimeError(
                "DL-MIA branch projector is unavailable. Fit on raw/vector data first or provide "
                "explicit vector/semantic/syntax features."
            )

        centered = vectors - self.vector_mean
        semantic = centered @ self.semantic_basis.T
        residual = centered - semantic @ self.semantic_basis
        syntax = residual @ self.syntax_basis.T
        return {
            "vector": vectors.astype(np.float32),
            "semantic": semantic.astype(np.float32),
            "syntax": syntax.astype(np.float32),
        }

    def _fit_branch_projector(self, vectors: np.ndarray) -> None:
        vectors = vectors.astype(np.float32)
        vector_dim = vectors.shape[1]
        semantic_dim = min(int(self.runtime_config["semantic_dim"]), vector_dim)
        syntax_dim = min(int(self.runtime_config["syntax_dim"]), vector_dim)

        self.vector_mean = vectors.mean(axis=0, keepdims=True).astype(np.float32)
        centered = vectors - self.vector_mean

        self.semantic_basis = self._svd_basis(centered, semantic_dim)
        semantic = centered @ self.semantic_basis.T
        residual = centered - semantic @ self.semantic_basis
        self.syntax_basis = self._svd_basis(residual, syntax_dim)

    def _svd_basis(self, matrix: np.ndarray, n_components: int) -> np.ndarray:
        if matrix.ndim != 2:
            raise ValueError(f"Expected 2D matrix for SVD basis, got shape {matrix.shape}.")
        if n_components <= 0:
            raise ValueError("n_components must be positive.")

        if np.allclose(matrix, 0.0):
            dim = matrix.shape[1]
            basis = np.eye(dim, dtype=np.float32)[:n_components]
            return basis.astype(np.float32)

        _, _, vh = np.linalg.svd(matrix, full_matrices=False)
        basis = vh[:n_components]
        if basis.shape[0] < n_components:
            dim = matrix.shape[1]
            fallback = np.eye(dim, dtype=np.float32)
            needed = n_components - basis.shape[0]
            basis = np.concatenate([basis, fallback[:needed]], axis=0)
        return basis.astype(np.float32)

    def _normalize_item_embeddings(self, value: Any) -> Dict[Any, np.ndarray]:
        if isinstance(value, dict):
            raw = value
        elif isinstance(value, torch.Tensor):
            matrix = value.detach().cpu().numpy()
            if matrix.ndim != 2:
                raise TypeError("item_embeddings tensor must be 2D.")
            raw = {idx: matrix[idx] for idx in range(matrix.shape[0])}
        elif isinstance(value, np.ndarray):
            if value.ndim != 2:
                raise TypeError("item_embeddings array must be 2D.")
            raw = {idx: value[idx] for idx in range(value.shape[0])}
        else:
            raise TypeError("item_embeddings must be a dict of item_id -> embedding_vector.")
        normalized: Dict[Any, np.ndarray] = {}
        for key, vector in raw.items():
            array = np.asarray(vector, dtype=np.float32).reshape(-1)
            normalized[key] = array
        if not normalized:
            raise ValueError("item_embeddings cannot be empty.")
        return normalized

    def _extract_item_embeddings_from_model(self, model: Any) -> Optional[Dict[Any, np.ndarray]]:
        for method_name in ("get_item_embeddings", "export_item_embeddings"):
            if hasattr(model, method_name):
                resolved = getattr(model, method_name)()
                return self._normalize_item_embeddings(resolved)

        candidate_paths = (
            ("item_embeddings", "weight"),
            ("embeddings_item", "weight"),
            ("item_embedding", "weight"),
            ("embedding_item", "weight"),
            ("item_emb", "weight"),
        )
        for attr_name, weight_name in candidate_paths:
            if not hasattr(model, attr_name):
                continue
            attr = getattr(model, attr_name)
            weight = getattr(attr, weight_name, attr)
            if isinstance(weight, torch.Tensor):
                matrix = weight.detach().cpu().numpy()
            else:
                matrix = np.asarray(weight)
            if matrix.ndim == 2:
                return {
                    item_id: matrix[item_id].astype(np.float32).reshape(-1)
                    for item_id in range(matrix.shape[0])
                }
        return None

    def _difference_vectors_from_partition(
        self,
        interactions: Any,
        recommendations: Any,
        item_embeddings: Dict[Any, np.ndarray],
        context: str,
    ) -> np.ndarray:
        if isinstance(interactions, dict) and isinstance(recommendations, dict):
            missing_recommendations = [key for key in interactions if key not in recommendations]
            extra_recommendations = [key for key in recommendations if key not in interactions]
            if missing_recommendations or extra_recommendations:
                raise ValueError(
                    f"{context} user ids do not align between interactions and recommendations. "
                    f"Missing recommendations: {missing_recommendations[:5]}; "
                    f"extra recommendations: {extra_recommendations[:5]}."
                )
            interaction_entries = [(key, list(interactions[key])) for key in interactions]
            recommendation_entries = [(key, list(recommendations[key])) for key in interactions]
        else:
            interaction_entries = self._normalize_user_sequences(interactions, f"{context} interactions")
            recommendation_entries = self._normalize_user_sequences(recommendations, f"{context} recommendations")

        if len(interaction_entries) != len(recommendation_entries):
            raise ValueError(
                f"{context} has mismatched interaction/recommendation counts: "
                f"{len(interaction_entries)} vs {len(recommendation_entries)}."
            )

        embed_dim = len(next(iter(item_embeddings.values())))
        vectors = np.zeros((len(interaction_entries), embed_dim), dtype=np.float32)
        for idx, ((_, interaction_items), (_, recommendation_items)) in enumerate(
            zip(interaction_entries, recommendation_entries)
        ):
            interaction_vec = self._mean_embedding(interaction_items, item_embeddings, embed_dim)
            recommendation_vec = self._mean_embedding(recommendation_items, item_embeddings, embed_dim)
            vectors[idx] = interaction_vec - recommendation_vec
        return vectors

    def _normalize_user_sequences(self, value: Any, context: str) -> List[Tuple[Any, List[Any]]]:
        if isinstance(value, dict):
            return [(key, list(items)) for key, items in value.items()]
        if isinstance(value, (list, tuple)):
            return [(idx, list(items)) for idx, items in enumerate(value)]
        raise TypeError(f"{context} must be a dict or a sequence of item-id lists.")

    def _mean_embedding(
        self,
        items: Sequence[Any],
        item_embeddings: Dict[Any, np.ndarray],
        embed_dim: int,
    ) -> np.ndarray:
        collected: List[np.ndarray] = []
        for item_id in items:
            if item_id in item_embeddings:
                collected.append(item_embeddings[item_id])
        if not collected:
            return np.zeros(embed_dim, dtype=np.float32)
        return np.mean(np.stack(collected, axis=0), axis=0).astype(np.float32)

    def _raw_sequence_keys(self, value: Any) -> List[Any]:
        if isinstance(value, dict):
            return list(value.keys())
        if isinstance(value, (list, tuple)):
            return list(range(len(value)))
        raise TypeError("Raw sequence payload must be a dict or a sequence of item-id lists.")

    def _coerce_raw_membership_labels(
        self,
        value: Any,
        expected_size: int,
        sample_keys: Optional[Sequence[Any]] = None,
    ) -> np.ndarray:
        if isinstance(value, dict):
            if sample_keys is None:
                labels = np.asarray(list(value.values()), dtype=np.int64).reshape(-1)
            else:
                missing = [key for key in sample_keys if key not in value]
                if missing:
                    raise ValueError(
                        f"membership_labels is missing labels for sample keys: {missing[:5]}."
                    )
                labels = np.asarray([value[key] for key in sample_keys], dtype=np.int64).reshape(-1)
        else:
            labels = self._to_numpy_1d(value).astype(np.int64)
        self._validate_label_count(labels, expected_size, "membership_labels")
        return labels

    def _validate_label_count(self, labels: np.ndarray, expected_size: int, context: str) -> None:
        if labels.shape[0] != expected_size:
            raise ValueError(
                f"{context} size mismatch: expected {expected_size}, got {labels.shape[0]}."
            )

    def _normalize_feature_bundle(self, value: Any, context: str) -> FeatureBundle:
        if not isinstance(value, dict):
            raise TypeError(f"{context} must be a dict containing vector/semantic/syntax features.")

        aliases = {
            "vector": ("vector", "base", "difference", "x1", "branch1"),
            "semantic": ("semantic", "invariant", "debias_semantic", "x2", "branch2"),
            "syntax": ("syntax", "specific", "debias_syntax", "x3", "branch3"),
        }
        resolved: FeatureBundle = {}

        for canonical_name, candidates in aliases.items():
            for candidate in candidates:
                if candidate in value:
                    resolved[canonical_name] = self._to_numpy_2d(value[candidate]).astype(np.float32)
                    break
            else:
                raise ValueError(
                    f"{context} is missing '{canonical_name}' features. "
                    f"Accepted aliases: {candidates}."
                )

        sizes = {resolved[name].shape[0] for name in resolved}
        if len(sizes) != 1:
            shape_summary = {name: resolved[name].shape for name in resolved}
            raise ValueError(f"{context} branch sample counts do not match: {shape_summary}")
        return resolved

    def _check_feature_dims(self, features: FeatureBundle) -> None:
        expected = self.feature_dims
        assert expected is not None
        current = (
            int(features["vector"].shape[1]),
            int(features["semantic"].shape[1]),
            int(features["syntax"].shape[1]),
        )
        if current != expected:
            raise ValueError(f"DL-MIA query feature dims mismatch: expected {expected}, got {current}.")

    def _to_numpy_1d(self, value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy().reshape(-1)
        return np.asarray(value).reshape(-1)

    def _to_numpy_2d(self, value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            array = value.detach().cpu().numpy()
        else:
            array = np.asarray(value)
        if array.ndim != 2:
            raise ValueError(f"Expected 2D feature array, got shape {array.shape}.")
        return array

    def _invoke_callback(self, callback: Any, **kwargs: Any) -> Any:
        if not callable(callback):
            raise TypeError("Configured callback must be callable.")
        signature = inspect.signature(callback)
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
            return callback(**kwargs)

        supported_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in signature.parameters
        }
        return callback(**supported_kwargs)


__all__ = ["DLMIAAttack"]
