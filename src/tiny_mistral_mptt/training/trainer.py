from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import random
import subprocess
import time

import torch

from tiny_mistral.device import synchronize

from ..config import ExperimentConfig
from ..data.manifest import file_sha256, verify_artifact
from ..data.packed_dataset import PackedTokenDataset, StatefulBlockSampler
from ..evaluation.nll import evaluate_nll
from ..evaluation.pass_depth import evaluate_pass_depth
from ..variants.base import ExperimentalVariant
from ..variants.multipass import MultiPassVariant
from .checkpoint import TrainState, load_checkpoint, load_model_weights, save_checkpoint
from .pass_schedule import PassScheduler
from .phases import configure_phase
from .schedule import lr_multiplier


def _append_jsonl(path: Path, item: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, sort_keys=True) + "\n")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parameter_count(parameters) -> int:
    return sum(parameter.numel() for parameter in parameters)


def _source_provenance() -> dict[str, bool | str | None]:
    """Return the source revision used to create a run manifest."""
    repository = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(repository), "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit, "git_dirty": dirty}


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
        self.pass_scheduler = PassScheduler(config.pass_schedule, seed=config.seed + 2)

        initialization_provenance = None
        if config.init_from:
            initialization_provenance = load_model_weights(config.init_from, model=self.model)

        trainable = configure_phase(model, config.phase)
        if trainable == 0:
            raise RuntimeError(f"Phase {config.phase} has no trainable parameters")
        self.optimizer = self._build_optimizer()
        self.state = TrainState(phase=config.phase)

        added_ids = {id(parameter) for parameter in model.added_parameters()}
        trainable_pretrained = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) not in added_ids
        ]
        trainable_added = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) in added_ids
        ]
        run_info = {
            "source": _source_provenance(),
            "config": config.to_dict(),
            "data_manifest_sha256": self.manifest_sha256,
            "sequence_length": train_data.sequence_length,
            "train_blocks": len(train_data),
            "validation_blocks": len(validation_data),
            "trainable_parameters": trainable,
            "trainable_pretrained_parameters": _parameter_count(trainable_pretrained),
            "trainable_added_parameters": _parameter_count(trainable_added),
            "added_parameters_total": _parameter_count(model.added_parameters()),
            "initialization_provenance": initialization_provenance,
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
                pass_scheduler=self.pass_scheduler,
            )
            self._repair_optimizer_group_metadata()
            if self.state.phase != config.phase:
                raise ValueError(
                    f"checkpoint phase {self.state.phase!r} does not match requested phase {config.phase!r}"
                )
            self.sampler.load_state_dict(sampler_state)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        added_ids = {id(parameter) for parameter in self.model.added_parameters()}
        pretrained = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad and id(parameter) not in added_ids
        ]
        added = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad and id(parameter) in added_ids
        ]
        groups: list[dict] = []
        if pretrained:
            groups.append(
                {
                    "params": pretrained,
                    "lr": self.config.pretrained_lr,
                    "base_lr": self.config.pretrained_lr,
                    "group_name": "pretrained",
                    "weight_decay": self.config.weight_decay,
                }
            )
        if added:
            groups.append(
                {
                    "params": added,
                    "lr": self.config.added_lr,
                    "base_lr": self.config.added_lr,
                    "group_name": "added",
                    "weight_decay": self.config.weight_decay,
                }
            )
        if not groups:
            raise RuntimeError("no optimizer parameters")
        return torch.optim.AdamW(groups, foreach=False)

    def _repair_optimizer_group_metadata(self) -> None:
        """Restore research-only optimizer metadata absent from v1 checkpoints."""
        added_ids = {id(parameter) for parameter in self.model.added_parameters()}
        for group in self.optimizer.param_groups:
            flags = {id(parameter) in added_ids for parameter in group["params"]}
            if len(flags) != 1:
                raise RuntimeError("optimizer group mixes pretrained and added parameters")
            is_added = next(iter(flags))
            name = "added" if is_added else "pretrained"
            group.setdefault("group_name", name)
            group.setdefault("base_lr", self.config.added_lr if is_added else self.config.pretrained_lr)

    def _set_lr(self) -> dict[str, float]:
        multiplier = lr_multiplier(
            self.state.unique_tokens_seen,
            total_tokens=self.config.max_unique_tokens,
            schedule=self.config.lr_schedule,
            legacy_warmup_tokens=self.config.warmup_tokens,
            legacy_min_lr_ratio=self.config.min_lr_ratio,
        )
        result = {"lr_multiplier": float(multiplier)}
        for group in self.optimizer.param_groups:
            base_lr = float(group["base_lr"])
            group["lr"] = base_lr * multiplier
            result[f"lr_{group['group_name']}"] = float(group["lr"])
        # Backwards-compatible scalar for existing plotting scripts.
        if "lr_pretrained" in result:
            result["lr"] = result["lr_pretrained"]
        elif "lr_added" in result:
            result["lr"] = result["lr_added"]
        return result

    def _checkpoint(self) -> Path:
        return save_checkpoint(
            self.run_dir / "latest.pt",
            model=self.model,
            optimizer=self.optimizer,
            sampler_state=self.sampler.state_dict(),
            pass_scheduler_state=self.pass_scheduler.state_dict(),
            train_state=self.state,
            experiment_config=self.config.to_dict(),
            data_manifest_sha256=self.manifest_sha256,
        )

    def _pass_schedule_metrics(self) -> dict[str, object]:
        return {
            "pass_samples": self.pass_scheduler.samples,
            "pass_histogram": {
                str(passes): count
                for passes, count in sorted(self.pass_scheduler.histogram.items())
            },
        }

    def _evaluate(self) -> dict:
        if self.config.eval_passes > 1:
            if not isinstance(self.model, MultiPassVariant):
                raise ValueError("eval_passes>1 requires a multipass variant")
            result = evaluate_pass_depth(
                self.model,
                self.validation_data,
                device=self.device,
                passes=self.config.eval_passes,
                max_blocks=self.config.eval_batches or None,
            )
            record = {
                "event": "validation",
                "optimizer_steps": self.state.optimizer_steps,
                "unique_tokens_seen": self.state.unique_tokens_seen,
                "token_equivalent_compute": self.state.token_equivalent_compute,
                "nll": result.final_nll,
                "perplexity": result.final_perplexity,
                "nll_by_pass": list(result.nll_by_pass),
                "perplexity_by_pass": list(result.perplexity_by_pass),
                "hidden_delta_rms": list(result.hidden_delta_rms),
                "nll_by_source_by_pass": list(result.nll_by_source_by_pass),
                "validation_blocks": result.blocks,
                "eval_passes": result.passes,
                **self._pass_schedule_metrics(),
            }
        else:
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
                "eval_passes": 1,
                **self._pass_schedule_metrics(),
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
            update_metrics: dict[str, float] = defaultdict(float)
            remaining_micro = (target_tokens - self.state.unique_tokens_seen) // tokens_per_micro
            accumulation_steps = min(cfg.grad_accum_steps, remaining_micro)
            if accumulation_steps <= 0:
                raise RuntimeError("invalid zero-length optimizer update")
            for _ in range(accumulation_steps):
                indices = self.sampler.next_indices(cfg.batch_size)
                ids = self.train_data.batch(indices, device=self.device)
                passes = self.pass_scheduler.sample(self.state.unique_tokens_seen)
                output = self.model.compute_loss(
                    ids,
                    phase=cfg.phase,
                    passes=passes,
                    loss_weights=cfg.loss_weights_for_passes(passes),
                )
                if not bool(torch.isfinite(output.loss).item()):
                    raise RuntimeError("non-finite training loss")
                (output.loss / accumulation_steps).backward()
                update_loss += float(output.loss.detach().cpu())
                update_passes += output.effective_passes
                for key, value in output.metrics.items():
                    update_metrics[key] += float(value)
                self.state.micro_steps += 1
                self.state.unique_tokens_seen += int(ids.numel())
                self.state.token_equivalent_compute += int(ids.numel()) * output.effective_passes
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            if not bool(torch.isfinite(grad_norm).item()):
                raise RuntimeError("non-finite gradient norm")
            lr_record = self._set_lr()
            self.optimizer.step()
            synchronize(self.device)
            self.state.optimizer_steps += 1
            elapsed = max(time.perf_counter() - start, 1e-9)
            record = {
                "event": "train",
                "phase": cfg.phase,
                "optimizer_steps": self.state.optimizer_steps,
                "micro_steps": self.state.micro_steps,
                "unique_tokens_seen": self.state.unique_tokens_seen,
                "token_equivalent_compute": self.state.token_equivalent_compute,
                "loss": update_loss / accumulation_steps,
                "grad_norm": float(grad_norm.detach().cpu()),
                "tokens_per_second": (tokens_per_micro * accumulation_steps) / elapsed,
                "mean_passes": update_passes / accumulation_steps,
                **self._pass_schedule_metrics(),
                **lr_record,
            }
            record.update(
                {key: value / accumulation_steps for key, value in sorted(update_metrics.items())}
            )
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
