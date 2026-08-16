from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import torch

from tiny_mistral.device import synchronize

from ..config import ExperimentConfig
from ..data.manifest import file_sha256, verify_artifact
from ..data.packed_dataset import PackedTokenDataset, StatefulBlockSampler
from ..evaluation.nll import evaluate_nll
from ..variants.base import ExperimentalVariant
from .checkpoint import TrainState, load_checkpoint, save_checkpoint
from .phases import configure_phase
from .schedule import cosine_lr_multiplier


def _append_jsonl(path: Path, item: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, sort_keys=True) + "\n")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(
        self,
        *,
        model: ExperimentalVariant,
        config: ExperimentConfig,
        train_data: PackedTokenDataset,
        validation_data: PackedTokenDataset,
        device: torch.device,
    ):
        config.validate()
        if train_data.manifest != validation_data.manifest:
            raise ValueError("train and validation datasets must come from the same artifact")
        self.model = model
        self.config = config
        self.train_data = train_data
        self.validation_data = validation_data
        self.device = device
        self.run_dir = Path(config.output_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        verify_artifact(config.data_dir)
        self.manifest_path = Path(config.data_dir) / "manifest.json"
        self.manifest_sha256 = file_sha256(self.manifest_path)
        _set_seed(config.seed)
        self.sampler = StatefulBlockSampler(len(train_data), seed=config.seed + 1)

        # Vanilla has no Phase-A parameters, so the bootstrap starts directly
        # in Phase B. The phase machinery is already present for future variants.
        trainable = configure_phase(model, "B")
        if trainable == 0:
            raise RuntimeError("Phase B has no trainable parameters")
        self.optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            foreach=False,
        )
        self.state = TrainState(phase="B")

        run_info = {
            "config": config.to_dict(),
            "data_manifest_sha256": self.manifest_sha256,
            "sequence_length": train_data.sequence_length,
            "train_blocks": len(train_data),
            "validation_blocks": len(validation_data),
            "trainable_parameters": trainable,
        }
        (self.run_dir / "run.json").write_text(
            json.dumps(run_info, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if config.resume_from:
            self.state, sampler_state = load_checkpoint(
                config.resume_from,
                model=self.model,
                optimizer=self.optimizer,
                expected_manifest_sha256=self.manifest_sha256,
                expected_experiment_config=config.to_dict(),
            )
            self.sampler.load_state_dict(sampler_state)

    def _set_lr(self) -> float:
        multiplier = cosine_lr_multiplier(
            self.state.unique_tokens_seen,
            total_tokens=self.config.max_unique_tokens,
            warmup_tokens=self.config.warmup_tokens,
            min_lr_ratio=self.config.min_lr_ratio,
        )
        lr = self.config.learning_rate * multiplier
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def _checkpoint(self) -> Path:
        return save_checkpoint(
            self.run_dir / "latest.pt",
            model=self.model,
            optimizer=self.optimizer,
            sampler_state=self.sampler.state_dict(),
            train_state=self.state,
            experiment_config=self.config.to_dict(),
            data_manifest_sha256=self.manifest_sha256,
        )

    def _evaluate(self) -> dict:
        result = evaluate_nll(
            self.model,
            self.validation_data,
            device=self.device,
            max_blocks=self.config.eval_batches or None,
        )
        record = {
            "event": "validation",
            "optimizer_steps": self.state.optimizer_steps,
            "unique_tokens_seen": self.state.unique_tokens_seen,
            "token_equivalent_compute": self.state.token_equivalent_compute,
            "nll": result.nll,
            "perplexity": result.perplexity,
            "nll_by_source": result.nll_by_source,
            "validation_blocks": result.blocks,
        }
        _append_jsonl(self.metrics_path, record)
        return record

    def train(self, *, until_unique_tokens: int | None = None) -> TrainState:
        cfg = self.config
        tokens_per_micro = cfg.batch_size * self.train_data.sequence_length
        target_tokens = cfg.max_unique_tokens if until_unique_tokens is None else int(until_unique_tokens)
        if not self.state.unique_tokens_seen <= target_tokens <= cfg.max_unique_tokens:
            raise ValueError("until_unique_tokens must lie between current progress and max_unique_tokens")
        if target_tokens % tokens_per_micro:
            raise ValueError(
                "token budget must be divisible by batch_size * sequence_length so the run ends exactly"
            )
        next_eval = (
            ((self.state.unique_tokens_seen // cfg.eval_every_tokens) + 1) * cfg.eval_every_tokens
            if cfg.eval_every_tokens else None
        )
        next_checkpoint = (
            ((self.state.unique_tokens_seen // cfg.checkpoint_every_tokens) + 1) * cfg.checkpoint_every_tokens
            if cfg.checkpoint_every_tokens else None
        )
        self.model.train()
        while self.state.unique_tokens_seen < target_tokens:
            start = time.perf_counter()
            self.optimizer.zero_grad(set_to_none=True)
            update_loss = 0.0
            update_passes = 0
            remaining_micro = (target_tokens - self.state.unique_tokens_seen) // tokens_per_micro
            accumulation_steps = min(cfg.grad_accum_steps, remaining_micro)
            if accumulation_steps <= 0:
                raise RuntimeError("invalid zero-length optimizer update")
            for _ in range(accumulation_steps):
                indices = self.sampler.next_indices(cfg.batch_size)
                ids = self.train_data.batch(indices, device=self.device)
                output = self.model.compute_loss(ids, phase="B", passes=1)
                if not bool(torch.isfinite(output.loss).item()):
                    raise RuntimeError("non-finite training loss")
                (output.loss / accumulation_steps).backward()
                update_loss += float(output.loss.detach().cpu())
                update_passes += output.effective_passes
                self.state.micro_steps += 1
                self.state.unique_tokens_seen += int(ids.numel())
                self.state.token_equivalent_compute += int(ids.numel()) * output.effective_passes
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            if not bool(torch.isfinite(grad_norm).item()):
                raise RuntimeError("non-finite gradient norm")
            lr = self._set_lr()
            self.optimizer.step()
            synchronize(self.device)
            self.state.optimizer_steps += 1
            elapsed = max(time.perf_counter() - start, 1e-9)
            record = {
                "event": "train",
                "optimizer_steps": self.state.optimizer_steps,
                "micro_steps": self.state.micro_steps,
                "unique_tokens_seen": self.state.unique_tokens_seen,
                "token_equivalent_compute": self.state.token_equivalent_compute,
                "loss": update_loss / accumulation_steps,
                "grad_norm": float(grad_norm.detach().cpu()),
                "lr": lr,
                "tokens_per_second": (tokens_per_micro * accumulation_steps) / elapsed,
                "mean_passes": update_passes / accumulation_steps,
            }
            _append_jsonl(self.metrics_path, record)

            if next_eval is not None and self.state.unique_tokens_seen >= next_eval:
                self._evaluate()
                while next_eval <= self.state.unique_tokens_seen:
                    next_eval += cfg.eval_every_tokens
            if next_checkpoint is not None and self.state.unique_tokens_seen >= next_checkpoint:
                self._checkpoint()
                while next_checkpoint <= self.state.unique_tokens_seen:
                    next_checkpoint += cfg.checkpoint_every_tokens

        self._checkpoint()
        if cfg.eval_batches:
            self._evaluate()
        return self.state
