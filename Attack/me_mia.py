"""
ME-MIA adapter compatible with the project's AttackInput/AttackOutput interface.

This wrapper targets recommendation-system membership inference. It keeps the
original ME-MIA workflow lightweight:

1. Obtain per-user score dictionaries from the target/shadow recommender.
2. Aggregate each user's sequence-level features into one attack feature.
3. Train a binary MIA classifier on shadow member/non-member users.
4. Infer membership scores for target users.

Accepted inputs
---------------
Training data can be provided in `attack_input.shadow_data` using either:

1. Precomputed score dictionaries:
   {
       "member_scores": ...,
       "nonmember_scores": ...,
   }

2. A shadow recommender model plus RecStudio datasets:
   {
       "shadow_model": shadow_model,
       "member_dataset": member_shadow_datasets,
       "nonmember_dataset": nonmember_shadow_datasets,
       "score_split": "train",   # optional: train / val / test / 0 / 1 / 2
   }

Inference data can be provided in one of these forms:

1. `attack_input.signals["score_dict"]`:
   a single unlabeled score dictionary to attack.

2. `attack_input.signals["member_scores"]` and
   `attack_input.signals["nonmember_scores"]`:
   useful when evaluation labels are already known and callers want the wrapper
   to concatenate member samples first and non-member samples second.

3. `attack_input.samples` as datasets together with `attack_input.target_model`:
   {
       "dataset": target_datasets,
       "score_split": "train",
   }
   or
   {
       "member_dataset": member_target_datasets,
       "nonmember_dataset": nonmember_target_datasets,
       "score_split": "train",
   }
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import copy
import importlib.util
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler

from Attack.shadow_based import AttackInput, AttackOutput, BaseAttack


ScoreDict = Dict[Any, Dict[str, Any]]


class _DataSampler(Sampler[torch.Tensor]):
    def __init__(
        self,
        data_source: Sequence[Any],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.generator = generator

    def __iter__(self) -> Iterable[torch.Tensor]:
        n = len(self.data_source)
        generator = self.generator or torch.Generator()
        if self.generator is None:
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))

        if self.shuffle:
            batches = torch.randperm(n, generator=generator).split(self.batch_size)
        else:
            batches = torch.arange(n).split(self.batch_size)
        if self.drop_last and len(batches[-1]) < self.batch_size:
            yield from batches[:-1]
        else:
            yield from batches

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        return (len(self.data_source) + self.batch_size - 1) // self.batch_size


class _SortedDataSampler(Sampler[torch.Tensor]):
    def __init__(
        self,
        data_source: "_MEMIADataset",
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.generator = generator

    def __iter__(self) -> Iterable[torch.Tensor]:
        n = len(self.data_source)
        if self.shuffle:
            sort_keys = torch.div(torch.randperm(n), self.batch_size * 10, rounding_mode="floor")
            sort_keys = self.data_source.sample_length + sort_keys * (
                int(self.data_source.sample_length.max().item()) + 1
            )
        else:
            sort_keys = self.data_source.sample_length
        batches = torch.sort(sort_keys).indices.split(self.batch_size)
        if self.drop_last and len(batches[-1]) < self.batch_size:
            yield from batches[:-1]
        else:
            yield from batches

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        return (len(self.data_source) + self.batch_size - 1) // self.batch_size


class _MEMIADataset(Dataset[Dict[str, torch.Tensor]]):
    def __init__(
        self,
        member_scores: Optional[ScoreDict],
        nonmember_scores: Optional[ScoreDict],
        mia_data_mode: str = "mean",
        gen_fea: bool = True,
    ) -> None:
        super().__init__()
        self.member_scores = member_scores or {}
        self.nonmember_scores = nonmember_scores or {}
        self.mode = mia_data_mode
        self.features: Any = None
        self.labels = torch.zeros(0, dtype=torch.long)
        self.sample_len = torch.zeros(0, dtype=torch.long)
        self.user_ids = torch.zeros(0, dtype=torch.long)

        if gen_fea:
            self._gen_fea()
            self._transform_fea()

    def _gen_fea(self) -> None:
        num_members = len(self.member_scores)
        num_nonmembers = len(self.nonmember_scores)
        total = num_members + num_nonmembers

        self.user_ids = torch.zeros(total, dtype=torch.long)
        self.labels = torch.zeros(total, dtype=torch.long)
        self.labels[:num_members] = 1
        self.sample_len = torch.zeros(total, dtype=torch.long)
        self.features = {}

        member_keys = list(self.member_scores.keys())
        nonmember_keys = list(self.nonmember_scores.keys())

        for idx in range(total):
            if idx < num_members:
                user_key = member_keys[idx]
                score_entry = self.member_scores[user_key]
            else:
                user_key = nonmember_keys[idx - num_members]
                score_entry = self.nonmember_scores[user_key]

            feature_seq = torch.as_tensor(score_entry["features"], dtype=torch.float32)
            end_order = np.argsort(np.asarray(score_entry["end"]))
            seq_len = int(feature_seq.shape[0])

            self.user_ids[idx] = int(user_key)
            self.sample_len[idx] = seq_len

            ordered_features = torch.zeros_like(feature_seq)
            for pos, original_idx in enumerate(end_order):
                ordered_features[pos] = feature_seq[int(original_idx)]

            if self.mode == "mean":
                self.features[idx] = torch.mean(ordered_features, dim=0)
            elif self.mode == "max":
                self.features[idx] = torch.max(ordered_features, dim=0).values
            elif self.mode == "min":
                self.features[idx] = torch.min(ordered_features, dim=0).values
            elif self.mode == "sum":
                self.features[idx] = torch.sum(ordered_features, dim=0)
            elif self.mode == "ccs":
                self.features[idx] = ordered_features[-1, :]
            else:
                self.features[idx] = ordered_features

    def _transform_fea(self) -> None:
        if self.mode == "all":
            return
        feature_matrix = torch.zeros((len(self.features), len(self.features[0])), dtype=torch.float32)
        for idx in range(len(self.features)):
            feature_matrix[idx] = self.features[idx]
        normalizer = torch.nn.BatchNorm1d(
            num_features=feature_matrix.shape[1],
            eps=1e-8,
            affine=False,
            track_running_stats=False,
        )
        self.features = normalizer(feature_matrix)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: Any) -> Dict[str, torch.Tensor]:
        if isinstance(index, int):
            index = np.asarray([index])
        elif isinstance(index, torch.Tensor):
            index = index.detach().cpu().numpy()

        batch: Dict[str, torch.Tensor] = {}
        if self.mode == "all":
            batch["input"] = pad_sequence(
                tuple(self.features[item_id] for item_id in index),
                batch_first=True,
            )
        else:
            batch["input"] = torch.vstack(tuple(self.features[item_id] for item_id in index))
        batch["label"] = self.labels[index]
        batch["user_id"] = self.user_ids[index]
        return batch

    @property
    def sample_length(self) -> torch.Tensor:
        return self.sample_len

    def loader(
        self,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0,
        drop_last: bool = False,
    ) -> DataLoader:
        if self.mode == "all":
            sampler = _SortedDataSampler(self, batch_size, shuffle, drop_last)
        else:
            sampler = _DataSampler(self, batch_size, shuffle, drop_last)
        return DataLoader(self, sampler=sampler, batch_size=None, shuffle=False, num_workers=num_workers)

    def build(self, split_ratio: Tuple[float, float] = (0.8, 0.2)) -> Tuple["_MEMIADataset", "_MEMIADataset"]:
        train_dataset = _MEMIADataset(None, None, mia_data_mode=self.mode, gen_fea=False)
        val_dataset = _MEMIADataset(None, None, mia_data_mode=self.mode, gen_fea=False)

        num_train = int(split_ratio[0] * len(self))
        train_index = np.random.choice(np.arange(len(self)), num_train, replace=False)
        val_index = np.setdiff1d(np.arange(len(self)), train_index)

        train_dataset.features = self.features[train_index]
        train_dataset.labels = self.labels[train_index]
        train_dataset.sample_len = self.sample_len[train_index]
        train_dataset.user_ids = self.user_ids[train_index]

        val_dataset.features = self.features[val_index]
        val_dataset.labels = self.labels[val_index]
        val_dataset.sample_len = self.sample_len[val_index]
        val_dataset.user_ids = self.user_ids[val_index]
        return train_dataset, val_dataset


class _MEMIAClassifier(torch.nn.Module):
    def __init__(
        self,
        num_fea: int,
        avg_mode: str = "mean",
        hidden_dims: Optional[List[int]] = None,
        rnn_layers: int = 2,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [64, 32, 8]
        assert avg_mode in ["mean", "max", "min", "sum", "LSTM", "GRU", "ccs", "all"]
        self.avg_mode = avg_mode
        self.num_fea = num_fea
        self.hidden_dims = hidden_dims
        self.rnn_layers = rnn_layers
        self.layers = torch.nn.ModuleList()

        if avg_mode == "LSTM":
            self.layers.append(torch.nn.LSTM(self.num_fea, self.hidden_dims[0], self.rnn_layers))
        elif avg_mode == "GRU":
            self.layers.append(torch.nn.GRU(self.num_fea, self.hidden_dims[0], self.rnn_layers))
        else:
            self.layers.append(torch.nn.Linear(self.num_fea, hidden_dims[0]))

        self.layers.append(torch.nn.ReLU())
        for idx in range(1, len(hidden_dims)):
            self.layers.append(torch.nn.Linear(hidden_dims[idx - 1], hidden_dims[idx]))
            self.layers.append(torch.nn.ReLU())
        self.layers.append(torch.nn.Linear(hidden_dims[-1], 2))

        self.start = 0
        self.end = num_fea

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        if self.avg_mode == "LSTM":
            hidden = self.layers[0](batch)[0][:, -1, :]
        elif self.avg_mode == "GRU":
            hidden = self.layers[0](batch)[0][:, -1, :]
        else:
            hidden = self.layers[0](batch)

        for idx in range(1, len(self.layers)):
            hidden = self.layers[idx](hidden)
        return hidden

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        early_stop_patience: int = 50,
        epochs: int = 300,
        optim_name: str = "Adam",
        lr: float = 1e-3,
        val_check: bool = True,
        start: int = 0,
        end: int = -1,
    ) -> None:
        self.start = start
        self.end = end
        best_parameters = copy.deepcopy(self.state_dict())
        best_val_auc = 0.0
        early_stop_track = 0
        loss_fn = torch.nn.CrossEntropyLoss()
        if optim_name == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        else:
            optimizer = torch.optim.SGD(self.parameters(), lr=lr)

        for _ in range(epochs):
            tik = time.time()
            self.train()
            train_loss: List[float] = []
            pre: Optional[np.ndarray] = None
            target: Optional[np.ndarray] = None

            for batch_data in train_loader:
                if self.avg_mode in ["LSTM", "GRU"]:
                    batch_x = batch_data["input"][:, :, start:end]
                else:
                    batch_x = batch_data["input"][:, start:end]
                labels = batch_data["label"]
                output = self(batch_x)
                loss = loss_fn(output, labels)

                batch_logits = output.detach().cpu().numpy()
                batch_labels = labels.detach().cpu().numpy()
                pre = batch_logits if pre is None else np.vstack((pre, batch_logits))
                target = batch_labels if target is None else np.hstack((target, batch_labels))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss.extend([loss.item()] * batch_x.shape[0])

            current_val_auc = 0.0
            if target is not None and pre is not None:
                current_train_auc = roc_auc_score(np.eye(2, dtype=int)[target], pre)
            else:
                current_train_auc = 0.0

            if val_loader is not None:
                self.eval()
                pre = None
                target = None
                with torch.no_grad():
                    for batch_data in val_loader:
                        if self.avg_mode in ["LSTM", "GRU"]:
                            batch_x = batch_data["input"][:, :, start:end]
                        else:
                            batch_x = batch_data["input"][:, start:end]
                        labels = batch_data["label"]
                        output = self(batch_x)

                        batch_logits = output.detach().cpu().numpy()
                        batch_labels = labels.detach().cpu().numpy()
                        pre = batch_logits if pre is None else np.vstack((pre, batch_logits))
                        target = batch_labels if target is None else np.hstack((target, batch_labels))

                if target is not None and pre is not None:
                    current_val_auc = roc_auc_score(np.eye(2, dtype=int)[target], pre)
                if val_check:
                    if current_val_auc <= best_val_auc:
                        early_stop_track += 1
                    else:
                        early_stop_track = 0
                        best_val_auc = current_val_auc
                        best_parameters = copy.deepcopy(self.state_dict())
                else:
                    best_parameters = copy.deepcopy(self.state_dict())
            else:
                best_parameters = copy.deepcopy(self.state_dict())

            _ = tik
            _ = current_train_auc
            _ = np.mean(np.asarray(train_loss)) if train_loss else 0.0
            if early_stop_track > early_stop_patience and val_check:
                self.load_state_dict(best_parameters)
                break

        self.load_state_dict(best_parameters)

    def predict_scores(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.eval()
        logits_list: List[np.ndarray] = []
        labels_list: List[np.ndarray] = []
        user_ids_list: List[np.ndarray] = []

        with torch.no_grad():
            for batch_data in dataloader:
                if self.avg_mode in ["LSTM", "GRU"]:
                    batch_x = batch_data["input"][:, :, self.start:self.end]
                else:
                    batch_x = batch_data["input"][:, self.start:self.end]
                logits = self(batch_x)
                probs = torch.softmax(logits, dim=1)[:, 1]

                logits_list.append(probs.detach().cpu().numpy())
                labels_list.append(batch_data["label"].detach().cpu().numpy())
                user_ids_list.append(batch_data["user_id"].detach().cpu().numpy())

        scores = np.concatenate(logits_list, axis=0) if logits_list else np.zeros(0, dtype=np.float32)
        labels = np.concatenate(labels_list, axis=0) if labels_list else np.zeros(0, dtype=np.int64)
        user_ids = np.concatenate(user_ids_list, axis=0) if user_ids_list else np.zeros(0, dtype=np.int64)
        return scores.astype(np.float32), labels.astype(np.int64), user_ids.astype(np.int64)


@dataclass
class _PreparedQuery:
    member_scores: Optional[ScoreDict]
    nonmember_scores: Optional[ScoreDict]
    dataset: _MEMIADataset
    has_explicit_partition: bool


class MEMIAAttack(BaseAttack):
    """
    Attack-interface wrapper for ME-MIA.

    Required:
        - shadow_data
        - signals or samples that can be resolved into ME-MIA score dictionaries

    Optional:
        - target_model when scores are computed online from datasets
        - membership_labels for evaluation
        - config for runtime hyperparameters

    Expected `config` keys
    ----------------------
    - `mia_data_mode`: feature aggregation mode. Default `mean`.
    - `classifier_mode`: classifier head mode. Default `mean`.
    - `hidden_dims`: MLP hidden dimensions. Default `[64, 32, 8]`.
    - `rnn_layers`: recurrent layers when classifier_mode is `LSTM` or `GRU`.
    - `batch_size`: default `1024`.
    - `epochs`: default `300`.
    - `lr`: default `1e-3`.
    - `feature_start`: default `64`.
    - `feature_end`: default `None`, meaning use all remaining features.
    - `val_split`: default `0.2`.
    - `val_check`: default `True`.
    - `early_stop_patience`: default `50`.
    - `optim`: default `Adam`.
    - `score_split`: default `train`.
    - `threshold`: default `0.5`.
    """

    def __init__(self) -> None:
        self.attack_model: Optional[_MEMIAClassifier] = None
        self.is_fitted = False
        self.config: Dict[str, Any] = {}

    def fit(self, attack_input: AttackInput) -> "MEMIAAttack":
        self.config = self._merge_config(attack_input.config)
        shadow_data = attack_input.shadow_data
        if shadow_data is None:
            raise ValueError("shadow_data is required for MEMIAAttack.fit().")

        member_scores, nonmember_scores = self._resolve_train_scores(shadow_data)
        train_val_dataset = _MEMIADataset(
            member_scores=member_scores,
            nonmember_scores=nonmember_scores,
            mia_data_mode=self.config["mia_data_mode"],
        )
        train_dataset, val_dataset = train_val_dataset.build(
            split_ratio=(1.0 - self.config["val_split"], self.config["val_split"])
        )
        train_loader = train_dataset.loader(
            batch_size=self.config["batch_size"],
            shuffle=True,
            drop_last=False,
        )
        val_loader = val_dataset.loader(
            batch_size=self.config["batch_size"],
            shuffle=False,
            drop_last=False,
        )

        inferred_dim = self._infer_feature_dim(member_scores, nonmember_scores)
        feature_end = self.config["feature_end"]
        if feature_end is None or feature_end > inferred_dim:
            feature_end = inferred_dim
        feature_start = min(self.config["feature_start"], feature_end)

        self.attack_model = _MEMIAClassifier(
            num_fea=feature_end - feature_start,
            avg_mode=self.config["classifier_mode"],
            hidden_dims=self.config["hidden_dims"],
            rnn_layers=self.config["rnn_layers"],
        )
        self.attack_model.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            early_stop_patience=self.config["early_stop_patience"],
            epochs=self.config["epochs"],
            optim_name=self.config["optim"],
            lr=self.config["lr"],
            val_check=self.config["val_check"],
            start=feature_start,
            end=feature_end,
        )
        self.is_fitted = True
        return self

    def infer(self, attack_input: AttackInput) -> AttackOutput:
        if not self.is_fitted or self.attack_model is None:
            raise RuntimeError("MEMIAAttack must be fitted before infer().")

        prepared_query = self._resolve_query_dataset(attack_input)
        query_loader = prepared_query.dataset.loader(
            batch_size=self.config["batch_size"],
            shuffle=False,
            drop_last=False,
        )
        scores, _, user_ids = self.attack_model.predict_scores(query_loader)
        threshold = float(self.config["threshold"])
        preds = (scores >= threshold).astype(np.int64)

        metadata = {
            "attack_name": "me_mia",
            "domain": "recommender_system",
            "has_explicit_partition": prepared_query.has_explicit_partition,
            "score_direction": "higher_is_member",
            "threshold": threshold,
        }
        if prepared_query.has_explicit_partition:
            num_members = len(prepared_query.member_scores or {})
            metadata["member_first_count"] = num_members

        return AttackOutput(
            membership_scores=scores,
            membership_preds=preds,
            intermediate_outputs={
                "user_ids": user_ids,
                "member_scores": prepared_query.member_scores,
                "nonmember_scores": prepared_query.nonmember_scores,
            },
            metadata=metadata,
        )

    def _merge_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        merged = {
            "mia_data_mode": "mean",
            "classifier_mode": "mean",
            "hidden_dims": [64, 32, 8],
            "rnn_layers": 2,
            "batch_size": 1024,
            "epochs": 300,
            "lr": 1e-3,
            "feature_start": 64,
            "feature_end": None,
            "val_split": 0.2,
            "val_check": True,
            "early_stop_patience": 50,
            "optim": "Adam",
            "score_split": "train",
            "threshold": 0.5,
        }
        merged.update(config)
        return merged

    def _resolve_train_scores(self, shadow_data: Dict[str, Any]) -> Tuple[ScoreDict, ScoreDict]:
        if "member_scores" in shadow_data and "nonmember_scores" in shadow_data:
            return (
                self._select_score_split(shadow_data["member_scores"], shadow_data.get("score_split")),
                self._select_score_split(shadow_data["nonmember_scores"], shadow_data.get("score_split")),
            )

        if "shadow_model" in shadow_data and "member_dataset" in shadow_data and "nonmember_dataset" in shadow_data:
            shadow_model = shadow_data["shadow_model"]
            score_split = shadow_data.get("score_split", self.config["score_split"])
            member_scores = self._compute_scores_from_dataset(shadow_model, shadow_data["member_dataset"], score_split)
            nonmember_scores = self._compute_scores_from_dataset(
                shadow_model,
                shadow_data["nonmember_dataset"],
                score_split,
            )
            return member_scores, nonmember_scores

        raise ValueError(
            "shadow_data must provide either member/nonmember score dictionaries "
            "or a shadow_model with member_dataset/nonmember_dataset."
        )

    def _resolve_query_dataset(self, attack_input: AttackInput) -> _PreparedQuery:
        signals = attack_input.signals or {}

        if "score_dict" in signals:
            score_dict = self._select_score_split(signals["score_dict"], signals.get("score_split"))
            dataset = _MEMIADataset(score_dict, None, mia_data_mode=self.config["mia_data_mode"])
            return _PreparedQuery(score_dict, None, dataset, has_explicit_partition=False)

        if "member_scores" in signals and "nonmember_scores" in signals:
            member_scores = self._select_score_split(signals["member_scores"], signals.get("score_split"))
            nonmember_scores = self._select_score_split(signals["nonmember_scores"], signals.get("score_split"))
            dataset = _MEMIADataset(member_scores, nonmember_scores, mia_data_mode=self.config["mia_data_mode"])
            return _PreparedQuery(member_scores, nonmember_scores, dataset, has_explicit_partition=True)

        if isinstance(attack_input.samples, dict):
            sample_dict = attack_input.samples
            score_split = sample_dict.get("score_split", self.config["score_split"])

            if "score_dict" in sample_dict:
                score_dict = self._select_score_split(sample_dict["score_dict"], score_split)
                dataset = _MEMIADataset(score_dict, None, mia_data_mode=self.config["mia_data_mode"])
                return _PreparedQuery(score_dict, None, dataset, has_explicit_partition=False)

            if "member_scores" in sample_dict and "nonmember_scores" in sample_dict:
                member_scores = self._select_score_split(sample_dict["member_scores"], score_split)
                nonmember_scores = self._select_score_split(sample_dict["nonmember_scores"], score_split)
                dataset = _MEMIADataset(member_scores, nonmember_scores, mia_data_mode=self.config["mia_data_mode"])
                return _PreparedQuery(member_scores, nonmember_scores, dataset, has_explicit_partition=True)

            if "dataset" in sample_dict:
                if attack_input.target_model is None:
                    raise ValueError("target_model is required when samples['dataset'] is provided.")
                score_dict = self._compute_scores_from_dataset(
                    attack_input.target_model,
                    sample_dict["dataset"],
                    score_split,
                )
                dataset = _MEMIADataset(score_dict, None, mia_data_mode=self.config["mia_data_mode"])
                return _PreparedQuery(score_dict, None, dataset, has_explicit_partition=False)

            if "member_dataset" in sample_dict and "nonmember_dataset" in sample_dict:
                if attack_input.target_model is None:
                    raise ValueError(
                        "target_model is required when samples['member_dataset'] and "
                        "samples['nonmember_dataset'] are provided."
                    )
                member_scores = self._compute_scores_from_dataset(
                    attack_input.target_model,
                    sample_dict["member_dataset"],
                    score_split,
                )
                nonmember_scores = self._compute_scores_from_dataset(
                    attack_input.target_model,
                    sample_dict["nonmember_dataset"],
                    score_split,
                )
                dataset = _MEMIADataset(member_scores, nonmember_scores, mia_data_mode=self.config["mia_data_mode"])
                return _PreparedQuery(member_scores, nonmember_scores, dataset, has_explicit_partition=True)

        raise ValueError(
            "Unable to resolve ME-MIA query data. Provide signals['score_dict'], "
            "signals['member_scores']/['nonmember_scores'], or dataset-based samples."
        )

    def _infer_feature_dim(self, member_scores: ScoreDict, nonmember_scores: ScoreDict) -> int:
        for score_dict in (member_scores, nonmember_scores):
            if score_dict:
                first_user = next(iter(score_dict.values()))
                return int(np.asarray(first_user["features"]).shape[-1])
        raise ValueError("Cannot infer ME-MIA feature dimension from empty score dictionaries.")

    def _select_score_split(self, score_value: Any, split: Optional[Any]) -> ScoreDict:
        if isinstance(score_value, dict):
            return score_value
        if isinstance(score_value, (list, tuple)):
            split_index = self._normalize_split_index(split if split is not None else self.config["score_split"])
            if split_index >= len(score_value):
                raise IndexError(
                    f"Requested score split {split_index}, but only {len(score_value)} splits are available."
                )
            selected = score_value[split_index]
            if not isinstance(selected, dict):
                raise TypeError("Selected ME-MIA score split must be a score dictionary.")
            return selected
        raise TypeError("Unsupported ME-MIA score container; expected dict, list, or tuple.")

    def _normalize_split_index(self, split: Any) -> int:
        if isinstance(split, int):
            return split
        mapping = {"train": 0, "val": 1, "valid": 1, "test": 2}
        split_key = str(split).lower()
        if split_key not in mapping:
            raise ValueError(f"Unsupported score_split '{split}'. Use train/val/test or 0/1/2.")
        return mapping[split_key]

    def _compute_scores_from_dataset(self, model: Any, datasets: Any, score_split: Any) -> ScoreDict:
        score_fn = _load_memia_score_function()
        score_result = score_fn(model, datasets)
        return self._select_score_split(score_result, score_split)


@lru_cache(maxsize=1)
def _load_memia_score_function():
    recstudio_root = Path(__file__).resolve().parent / "memia" / "RecStudio"
    if not recstudio_root.exists():
        raise FileNotFoundError(f"ME-MIA RecStudio directory not found: {recstudio_root}")

    recstudio_root_str = str(recstudio_root)
    if recstudio_root_str not in sys.path:
        sys.path.insert(0, recstudio_root_str)

    module_path = recstudio_root / "utils.py"
    spec = importlib.util.spec_from_file_location("_memia_recstudio_utils", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load ME-MIA utils module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "score"):
        raise AttributeError("ME-MIA utils module does not expose score().")
    return module.score


__all__ = ["MEMIAAttack"]
