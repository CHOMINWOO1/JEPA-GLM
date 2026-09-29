from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils.data import DataLoader

from jepa_glm.data.masking import contiguous_span_mask, multi_region_span_mask
from jepa_glm.data.tokenizer import SequenceTokenizer
from jepa_glm.losses.jepa_loss import jepa_loss
from jepa_glm.losses.mlm_loss import mlm_loss
from jepa_glm.losses.regularization import covariance_regularization, embedding_std, variance_regularization
from jepa_glm.training.optimizer import optimizer_learning_rates
from jepa_glm.models.jepa_glm import JepaGlmModel
from jepa_glm.training.checkpoint import capture_rng_state, restore_rng_state
from jepa_glm.training.ema import max_parameter_delta, update_ema


@dataclass
class TrainSummary:
    final_loss: float
    final_jepa_loss: float
    final_mlm_loss: float
    embedding_std: float
    min_embedding_std: float
    final_ema_delta: float
    final_variance_reg: float = 0.0
    final_covariance_reg: float = 0.0
    regularization_queue_fill: int = 0
    validation_loss: float | None = None
    best_validation_loss: float | None = None
    start_step: int = 0
    completed_steps: int = 0
    start_optimizer_step: int = 0
    completed_optimizer_steps: int = 0
    step_metrics: list[dict[str, float]] = field(default_factory=list)
    scaler_state: dict[str, Any] = field(default_factory=dict)


def train_steps(
    model: JepaGlmModel,
    dataloader: DataLoader,
    config: dict[str, Any],
    tokenizer: SequenceTokenizer,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    start_step: int = 0,
    start_optimizer_step: int = 0,
    validation_dataloader: DataLoader | None = None,
    scaler_state: dict[str, Any] | None = None,
) -> TrainSummary:
    model.to(device)
    model.train()
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=float(config["training"].get("learning_rate", 5e-4)),
            weight_decay=float(config["training"].get("weight_decay", 0.01)),
        )
    steps = int(config["training"].get("steps", 10))
    grad_accum = int(config["training"].get("gradient_accumulation_steps", 1))
    max_grad_norm = float(config["training"].get("max_grad_norm", 0.0))
    use_amp = bool(config["training"].get("mixed_precision", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if scaler_state:
        scaler.load_state_dict(scaler_state)
    last = {"loss": 0.0, "jepa": 0.0, "mlm": 0.0, "std": 0.0, "variance": 0.0, "covariance": 0.0}
    step_metrics: list[dict[str, float]] = []
    iterator = iter(dataloader)
    optimizer.zero_grad(set_to_none=True)
    optimizer_step_count = int(start_optimizer_step)
    for local_step in range(1, steps + 1):
        global_step = start_step + local_step
        try:
            input_ids = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            input_ids = next(iterator)
        input_ids = input_ids.to(device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            batch = compute_batch_losses(model, input_ids, config, tokenizer)
            accumulation_start = ((local_step - 1) // grad_accum) * grad_accum + 1
            accumulation_end = min(accumulation_start + grad_accum - 1, steps)
            accumulation_size = accumulation_end - accumulation_start + 1
            loss = batch["loss"] / accumulation_size
        scaler.scale(loss).backward()
        did_optimizer_step = local_step == accumulation_end
        grad_norm = 0.0
        if did_optimizer_step:
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm).item())
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            ema_momentum = _ema_momentum(
                config.get("training", {}),
                optimizer_step_count + 1,
            )
            update_ema(
                model.context_encoder,
                model.target_encoder.encoder,
                ema_momentum,
            )
            optimizer_step_count += 1
        else:
            ema_momentum = _ema_momentum(config.get("training", {}), optimizer_step_count)
        ema_delta = max_parameter_delta(model.context_encoder, model.target_encoder.encoder)
        last = {
            "loss": float(batch["loss"].item()),
            "jepa": float(batch["jepa_loss"].item()),
            "mlm": float(batch["mlm_loss"].item()),
            "std": embedding_std(batch["statistics_embedding"]),
            "ema_delta": float(ema_delta),
            "variance": float(batch["variance_reg"].item()),
            "covariance": float(batch["covariance_reg"].item()),
        }
        step_metrics.append(
            {
                "step": float(local_step),
                "global_step": float(global_step),
                "global_optimizer_step": float(optimizer_step_count),
                "loss": last["loss"],
                "jepa_loss": last["jepa"],
                "mlm_loss": last["mlm"],
                "variance_reg": float(batch["variance_reg"].item()),
                "covariance_reg": float(batch["covariance_reg"].item()),
                "regularization_batch_size": float(batch["regularization_batch_size"]),
                "regularization_gradient_scale": float(batch["regularization_gradient_scale"]),
                "embedding_std": last["std"],
                "ema_delta": last["ema_delta"],
                "ema_momentum": float(ema_momentum),
                "mask_fraction": float(batch["target_mask"].float().mean().item()),
                "mask_fraction_eligible": float(batch["mask_fraction_eligible"].item()),
                "target_tokens": float(batch["target_mask"].sum().item()),
                "prediction_count": float(batch["predicted_embedding"].shape[0]),
                "optimizer_step": float(1 if did_optimizer_step else 0),
                "grad_norm": grad_norm,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "backbone_learning_rate": optimizer_learning_rates(optimizer).get(
                    "backbone", float(optimizer.param_groups[0]["lr"])
                ),
                "predictor_learning_rate": optimizer_learning_rates(optimizer).get(
                    "predictor", float(optimizer.param_groups[0]["lr"])
                ),
                "mlm_head_learning_rate": optimizer_learning_rates(optimizer).get(
                    "mlm_head", float(optimizer.param_groups[0]["lr"])
                ),
            }
        )
        if local_step % int(config["training"].get("log_every", 1)) == 0:
            print(
                f"micro_step={global_step} optimizer_step={optimizer_step_count} loss={last['loss']:.4f} jepa={last['jepa']:.4f} "
                f"mlm={last['mlm']:.4f} std={last['std']:.4f} ema_delta={last['ema_delta']:.6f}"
            )
    min_std = min(row["embedding_std"] for row in step_metrics) if step_metrics else 0.0
    validation_rng_state = capture_rng_state()
    try:
        validation_loss = evaluate_loss(model, validation_dataloader, config, tokenizer, device) if validation_dataloader is not None else None
    finally:
        restore_rng_state(validation_rng_state)
    return TrainSummary(
        final_loss=last["loss"],
        final_jepa_loss=last["jepa"],
        final_mlm_loss=last["mlm"],
        embedding_std=last["std"],
        min_embedding_std=min_std,
        final_ema_delta=last.get("ema_delta", 0.0),
        final_variance_reg=last.get("variance", 0.0),
        final_covariance_reg=last.get("covariance", 0.0),
        regularization_queue_fill=int(model.regularization_queue_count.item()),
        validation_loss=validation_loss,
        best_validation_loss=validation_loss,
        start_step=start_step,
        completed_steps=start_step + steps,
        start_optimizer_step=start_optimizer_step,
        completed_optimizer_steps=optimizer_step_count,
        step_metrics=step_metrics,
        scaler_state=scaler.state_dict(),
    )


def compute_batch_losses(
    model: JepaGlmModel,
    input_ids: torch.Tensor,
    config: dict[str, Any],
    tokenizer: SequenceTokenizer,
) -> dict[str, torch.Tensor]:
    loss_cfg = config.get("loss", {})
    data_cfg = config.get("data", {})
    context_ids, target_mask, mlm_labels = _mask_batch(input_ids, data_cfg, tokenizer)
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    outputs = model(context_ids, input_ids, attention_mask=attention_mask, target_mask=target_mask)
    jepa = jepa_loss(
        outputs["predicted_embedding"],
        outputs["target_embedding"],
        loss_type=str(loss_cfg.get("jepa_loss", "cosine")),
    )
    masked_lm = mlm_loss(outputs["mlm_logits"], mlm_labels)
    if model.jepa_target_mode == "cls":
        context_statistics, prediction_statistics, gradient_scale = model.regularization_batches(
            outputs["context_embedding"],
            outputs["predicted_embedding"],
        )
        reg = 0.5 * (
            variance_regularization(context_statistics)
            + variance_regularization(prediction_statistics)
        )
        covariance = 0.5 * (
            covariance_regularization(context_statistics)
            + covariance_regularization(prediction_statistics)
        )
        statistics_embedding = prediction_statistics
    else:
        reg = variance_regularization(outputs["predicted_embedding"])
        covariance = outputs["predicted_embedding"].sum() * 0.0
        statistics_embedding = outputs["predicted_embedding"]
        gradient_scale = 1.0
    if not bool(loss_cfg.get("regularization_queue_gradient_rescale", True)):
        gradient_scale = 1.0
    lambda_variance = float(loss_cfg.get("lambda_var", loss_cfg.get("lambda_reg", 0.0)))
    lambda_covariance = float(loss_cfg.get("lambda_cov", 0.0))
    regularization_term = lambda_variance * reg + lambda_covariance * covariance
    if gradient_scale != 1.0:
        regularization_term = regularization_term.detach() + gradient_scale * (
            regularization_term - regularization_term.detach()
        )
    loss = (
        float(loss_cfg.get("lambda_jepa", 1.0)) * jepa
        + float(loss_cfg.get("lambda_mlm", 0.0)) * masked_lm
        + regularization_term
    )
    eligible = attention_mask.clone()
    for token_id in getattr(tokenizer, "special_token_ids", ()):
        eligible &= input_ids.ne(int(token_id))
    mask_fraction_eligible = target_mask.sum().float() / eligible.sum().clamp_min(1).float()
    return {
        "loss": loss,
        "jepa_loss": jepa,
        "mlm_loss": masked_lm,
        "variance_reg": reg,
        "covariance_reg": covariance,
        "predicted_embedding": outputs["predicted_embedding"],
        "statistics_embedding": statistics_embedding,
        "regularization_batch_size": int(statistics_embedding.shape[0]),
        "regularization_gradient_scale": gradient_scale,
        "target_mask": target_mask,
        "mask_fraction_eligible": mask_fraction_eligible,
    }


@torch.no_grad()
def evaluate_loss(
    model: JepaGlmModel,
    dataloader: DataLoader | None,
    config: dict[str, Any],
    tokenizer: SequenceTokenizer,
    device: torch.device,
    max_batches: int | None = None,
) -> float | None:
    if dataloader is None:
        return None
    was_training = model.training
    model.eval()
    losses: list[float] = []
    for batch_index, input_ids in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break
        input_ids = input_ids.to(device)
        losses.append(float(compute_batch_losses(model, input_ids, config, tokenizer)["loss"].item()))
    if was_training:
        model.train()
    return sum(losses) / len(losses) if losses else None


def _mask_batch(input_ids: torch.Tensor, data_cfg: dict[str, Any], tokenizer: SequenceTokenizer):
    common = {
        "mask_token_id": tokenizer.mask_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "excluded_token_ids": getattr(tokenizer, "special_token_ids", ()),
    }
    strategy = str(data_cfg.get("mask_strategy", data_cfg.get("mask_type", "contiguous_span"))).lower()
    if strategy in {"multi_region", "multi_region_span"}:
        return multi_region_span_mask(
            input_ids,
            min_mask_ratio=float(data_cfg.get("min_mask_ratio", 0.2)),
            max_mask_ratio=float(data_cfg.get("max_mask_ratio", 0.4)),
            min_regions=int(data_cfg.get("min_mask_regions", 1)),
            max_regions=int(data_cfg.get("max_mask_regions", 3)),
            **common,
        )
    return contiguous_span_mask(
        input_ids,
        mask_ratio=float(data_cfg.get("mask_ratio", 0.3)),
        span_length=int(data_cfg.get("mask_span_length", 12)),
        **common,
    )


def _ema_momentum(training: dict[str, Any], optimizer_step: int) -> float:
    if training.get("ema_momentum_start") is None and training.get("ema_momentum_end") is None:
        return float(training.get("ema_momentum", 0.996))
    start = float(training.get("ema_momentum_start", training.get("ema_momentum", 0.996)))
    end = float(training.get("ema_momentum_end", start))
    total = max(1, int(training.get("optimizer_steps", training.get("steps", 1))))
    progress = min(max(optimizer_step - 1, 0), total - 1) / max(1, total - 1)
    return start + (end - start) * progress
