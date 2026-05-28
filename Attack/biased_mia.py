"""
Biased-MIA adapter compatible with the project's AttackInput/AttackOutput interface.

This wrapper captures the common core used by the biased-attack scripts in
`DL-MIA-KDD-2022-main`: a binary classifier trained on user-level difference
vectors between historical interactions and recommendation results.

Required inputs
---------------
Training (`attack_input.shadow_data`) supports:

1. Explicit member/non-member vectors:
   {
       "member_vectors": ndarray or Tensor,     # shape [N_member, D]
       "nonmember_vectors": ndarray or Tensor,  # shape [N_nonmember, D]
   }

2. A combined feature matrix with labels:
   {
       "vectors": ndarray or Tensor,            # shape [N, D]
       "membership_labels": ndarray or Tensor,  # shape [N]
   }

3. Raw recommender inputs:
   {
       "member_interactions": {user_id: [item_id, ...], ...},
       "nonmember_interactions": {user_id: [item_id, ...], ...},
       "member_recommendations": {user_id: [item_id, ...], ...},
       "nonmember_recommendations": {user_id: [item_id, ...], ...},
       "item_embeddings": {item_id: embedding_vector, ...},
   }

Inference supports the same formats through either `attack_input.signals`
or `attack_input.samples`.
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


@dataclass
class _PreparedBiasedQuery:
    vectors: np.ndarray
    membership_labels: Optional[np.ndarray]
    has_explicit_partition: bool


class _BiasedMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Tuple[int, int] = (32, 8)) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class BiasedMIAAttack(BaseAttack):
    """
    Interface-compatible wrapper for vector-based Biased-MIA.

    Required:
        - shadow_data with member/non-member vectors, raw interaction data,
          or vector matrix + labels

    Optional:
        - membership_labels for evaluation
        - signals containing vectors for inference
        - samples containing vectors for inference
        - config for runtime hyperparameters
    """

    def __init__(
        self,
        hidden_dims: Tuple[int, int] = (32, 8),
        batch_size: int = 256,
        lr: float = 1e-2,
        momentum: float = 0.7,
        epochs: int = 15,
        device: Optional[str] = None,
    ) -> None:
        self.hidden_dims = hidden_dims
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.epochs = epochs
        self.device = torch.device(device if device is not None else "cpu")
        self.threshold = 0.5

        self.model: Optional[_BiasedMLP] = None
        self.input_dim: Optional[int] = None
        self.runtime_config: Dict[str, Any] = {}
        self.is_fitted = False

    def fit(self, attack_input: AttackInput) -> "BiasedMIAAttack":
        if attack_input.shadow_data is None:
            raise ValueError("shadow_data is required for BiasedMIAAttack.fit().")

        self.runtime_config = self._merge_config(attack_input.config)
        x_train, y_train = self._resolve_train_data(
            attack_input.shadow_data,
            model=attack_input.shadow_data.get("shadow_model"),
        )
        self.input_dim = int(x_train.shape[1])
        self.model = _BiasedMLP(
            self.input_dim,
            tuple(self.runtime_config["hidden_dims"]),
        ).to(self.device)

        dataset = TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        loader = DataLoader(dataset, batch_size=self.runtime_config["batch_size"], shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.runtime_config["lr"],
            momentum=self.runtime_config["momentum"],
        )

        self.model.train()
        for _ in range(self.runtime_config["epochs"]):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

        self.is_fitted = True
        return self

    def infer(self, attack_input: AttackInput) -> AttackOutput:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("BiasedMIAAttack must be fitted before infer().")

        prepared = self._resolve_query_data(attack_input)
        x_query = prepared.vectors
        if x_query.ndim != 2:
            raise ValueError(f"Expected 2D query vectors, got shape {x_query.shape}.")
        if self.input_dim is not None and x_query.shape[1] != self.input_dim:
            raise ValueError(
                f"Biased-MIA query feature dimension mismatch: expected {self.input_dim}, "
                f"got {x_query.shape[1]}."
            )

        with torch.no_grad():
            logits = self.model(torch.tensor(x_query, dtype=torch.float32, device=self.device))
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy().astype(np.float32)

        threshold = float(self.runtime_config.get("threshold", self.threshold))
        preds = (probs >= threshold).astype(np.int64)
        metadata = {
            "attack_name": "biased_mia",
            "domain": "recommender_system",
            "score_direction": "higher_is_member",
            "has_explicit_partition": prepared.has_explicit_partition,
            "threshold": threshold,
        }

        return AttackOutput(
            membership_scores=probs,
            membership_preds=preds,
            intermediate_outputs={"difference_vectors": x_query},
            metadata=metadata,
        )

    def _resolve_train_data(
        self,
        shadow_data: Dict[str, Any],
        model: Optional[Any] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        x_train, y_train, _ = self._coerce_vectors_and_labels(
            shadow_data,
            context="shadow_data",
            model=model,
        )
        if y_train is None:
            raise ValueError("Training labels are required for Biased-MIA fit().")
        return x_train, y_train

    def _merge_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        merged = {
            "hidden_dims": list(self.hidden_dims),
            "batch_size": self.batch_size,
            "lr": self.lr,
            "momentum": self.momentum,
            "epochs": self.epochs,
            "threshold": self.threshold,
        }
        merged.update(config)
        return merged

    def _resolve_query_data(self, attack_input: AttackInput) -> _PreparedBiasedQuery:
        if attack_input.signals is not None:
            if self._contains_supported_payload(attack_input.signals):
                x_query, y_query, explicit = self._coerce_vectors_and_labels(
                    attack_input.signals,
                    context="signals",
                    allow_missing_labels=True,
                    model=attack_input.target_model,
                )
                return _PreparedBiasedQuery(x_query, y_query, explicit)

        if isinstance(attack_input.samples, dict) and self._contains_supported_payload(attack_input.samples):
            x_query, y_query, explicit = self._coerce_vectors_and_labels(
                attack_input.samples,
                context="samples",
                allow_missing_labels=True,
                model=attack_input.target_model,
            )
            return _PreparedBiasedQuery(x_query, y_query, explicit)

        if attack_input.samples is not None:
            vectors = self._to_numpy_2d(attack_input.samples)
            return _PreparedBiasedQuery(vectors, None, False)

        raise ValueError(
            "Unable to resolve Biased-MIA query data. Provide vectors through "
            "signals, samples dict, or samples matrix."
        )

    def _contains_vector_payload(self, payload: Dict[str, Any]) -> bool:
        return any(
            key in payload
            for key in ("member_vectors", "nonmember_vectors", "vectors")
        )

    def _contains_supported_payload(self, payload: Dict[str, Any]) -> bool:
        return (
            self._contains_vector_payload(payload)
            or self._contains_raw_partition_payload(payload)
            or self._contains_raw_combined_payload(payload)
        )

    def _coerce_vectors_and_labels(
        self,
        payload: Dict[str, Any],
        context: str,
        allow_missing_labels: bool = False,
        model: Optional[Any] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], bool]:
        if "member_vectors" in payload and "nonmember_vectors" in payload:
            member_vectors = self._to_numpy_2d(payload["member_vectors"])
            nonmember_vectors = self._to_numpy_2d(payload["nonmember_vectors"])
            x = np.concatenate([member_vectors, nonmember_vectors], axis=0).astype(np.float32)
            y = np.concatenate(
                [
                    np.ones(member_vectors.shape[0], dtype=np.int64),
                    np.zeros(nonmember_vectors.shape[0], dtype=np.int64),
                ],
                axis=0,
            )
            return x, y, True

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
            x = np.concatenate([member_vectors, nonmember_vectors], axis=0).astype(np.float32)
            y = np.concatenate(
                [
                    np.ones(member_vectors.shape[0], dtype=np.int64),
                    np.zeros(nonmember_vectors.shape[0], dtype=np.int64),
                ],
                axis=0,
            )
            return x, y, True

        if "vectors" in payload:
            x = self._to_numpy_2d(payload["vectors"]).astype(np.float32)
            if "membership_labels" in payload:
                y = self._to_numpy_1d(payload["membership_labels"]).astype(np.int64)
                self._validate_label_count(y, x.shape[0], f"{context}['membership_labels']")
                return x, y, False
            if allow_missing_labels:
                return x, None, False
            raise ValueError(f"{context}['membership_labels'] is required when using combined vectors.")

        if self._contains_raw_combined_payload(payload):
            item_embeddings = self._resolve_item_embeddings(payload, model, context)
            recommendations = self._resolve_combined_recommendations(payload, model, context)
            x = self._difference_vectors_from_partition(
                interactions=payload["interactions"],
                recommendations=recommendations,
                item_embeddings=item_embeddings,
                context=f"{context} combined raw payload",
            )
            if "membership_labels" in payload:
                y = self._coerce_raw_membership_labels(
                    payload["membership_labels"],
                    expected_size=x.shape[0],
                    sample_keys=self._raw_sequence_keys(payload["interactions"]),
                )
                return x, y, False
            if allow_missing_labels:
                return x, None, False
            raise ValueError(f"{context}['membership_labels'] is required when using combined raw inputs.")

        raise ValueError(
            f"{context} must provide vectors, raw recommender inputs, or a combined "
            "feature matrix with membership_labels."
        )

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

    def _normalize_item_embeddings(self, value: Any) -> Dict[Any, np.ndarray]:
        if isinstance(value, dict):
            embeddings = value
        elif isinstance(value, torch.Tensor):
            matrix = value.detach().cpu().numpy()
            if matrix.ndim != 2:
                raise TypeError("item_embeddings tensor must be 2D.")
            embeddings = {idx: matrix[idx] for idx in range(matrix.shape[0])}
        elif isinstance(value, np.ndarray):
            if value.ndim != 2:
                raise TypeError("item_embeddings array must be 2D.")
            embeddings = {idx: value[idx] for idx in range(value.shape[0])}
        else:
            raise TypeError("item_embeddings must be a dict of item_id -> embedding_vector.")

        normalized: Dict[Any, np.ndarray] = {}
        for key, vector in embeddings.items():
            array = np.asarray(vector, dtype=np.float32).reshape(-1)
            if array.ndim != 1:
                raise ValueError("Each item embedding must be one-dimensional.")
            normalized[key] = array
        if not normalized:
            raise ValueError("item_embeddings cannot be empty.")
        return normalized

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
            raise ValueError(f"Expected 2D array, got shape {array.shape}.")
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


__all__ = ["BiasedMIAAttack"]
