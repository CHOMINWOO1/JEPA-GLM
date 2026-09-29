from __future__ import annotations

import torch
from torch import nn

from jepa_glm.models.backbone import build_backbone
from jepa_glm.models.context_encoder import ContextEncoder
from jepa_glm.models.pooling import pool_hidden
from jepa_glm.models.predictor import AttentionPredictor, Predictor
from jepa_glm.models.target_encoder import TargetEncoder


class JepaGlmModel(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        backbone = build_backbone(config)
        hidden_size = int(getattr(backbone, "hidden_size"))
        pretrained_mlm_head = (
            backbone.take_pretrained_mlm_head()
            if hasattr(backbone, "take_pretrained_mlm_head")
            else None
        )
        self.context_encoder = ContextEncoder(backbone)
        self.target_encoder = TargetEncoder(self.context_encoder)
        self.target_encoder_eval_mode = bool(config.get("target_encoder_eval_mode", False))
        self.pooling = str(config.get("pooling", "mean"))
        self.jepa_target_mode = str(config.get("jepa_target_mode", "pooled")).lower()
        if self.jepa_target_mode not in {"pooled", "masked_tokens", "cls"}:
            raise ValueError("jepa_target_mode must be 'pooled', 'masked_tokens', or 'cls'")
        predictor_size = hidden_size * 2 if self.jepa_target_mode == "pooled" and self.pooling == "mean_max" else hidden_size
        if self.jepa_target_mode == "cls":
            self.predictor = AttentionPredictor(
                hidden_size,
                embed_dim=int(config.get("predictor_embed_dim", 384)),
                layers=int(config.get("predictor_layers", 4)),
                heads=int(config.get("predictor_heads", 2)),
                max_length=int(config.get("max_length", 512)),
                dropout=float(config.get("predictor_dropout", 0.0)),
            )
        else:
            self.predictor = Predictor(predictor_size, int(config.get("predictor_layers", 2)))
        self.mlm_head = pretrained_mlm_head or nn.Linear(hidden_size, int(config.get("vocab_size", 8)))
        self.mlm_head_source = "pretrained" if pretrained_mlm_head is not None else "random_linear"
        self.regularization_queue_size = int(config.get("regularization_queue_size", 0))
        if self.regularization_queue_size < 0:
            raise ValueError("regularization_queue_size must be >= 0")
        self.register_buffer(
            "context_regularization_queue",
            torch.zeros(self.regularization_queue_size, hidden_size),
            persistent=self.regularization_queue_size > 0,
        )
        self.register_buffer(
            "prediction_regularization_queue",
            torch.zeros(self.regularization_queue_size, hidden_size),
            persistent=self.regularization_queue_size > 0,
        )
        self.register_buffer(
            "regularization_queue_position",
            torch.zeros((), dtype=torch.long),
            persistent=self.regularization_queue_size > 0,
        )
        self.register_buffer(
            "regularization_queue_count",
            torch.zeros((), dtype=torch.long),
            persistent=self.regularization_queue_size > 0,
        )

    def forward(
        self,
        context_ids: torch.Tensor,
        target_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if attention_mask is None:
            attention_mask = context_ids.ne(0)
        context_hidden = self.context_encoder(context_ids, attention_mask)
        with torch.no_grad():
            if self.target_encoder_eval_mode:
                self.target_encoder.eval()
            target_hidden = self.target_encoder(target_ids, attention_mask)
        if target_mask is None:
            target_pool_mask = attention_mask
        else:
            target_pool_mask = target_mask & attention_mask.bool()
        if self.jepa_target_mode == "masked_tokens":
            if not bool(target_pool_mask.any().item()):
                raise ValueError("masked_tokens JEPA requires at least one eligible target token")
            predicted_embedding = self.predictor(context_hidden[target_pool_mask])
            target_embedding = target_hidden[target_pool_mask].detach()
            context_embedding = context_hidden[target_pool_mask]
        elif self.jepa_target_mode == "cls":
            predicted_embedding = self.predictor(
                context_hidden,
                attention_mask=attention_mask,
                target_mask=target_pool_mask,
            )
            target_embedding = target_hidden[:, 0].detach()
            context_embedding = context_hidden[:, 0]
        else:
            pooled_context = pool_hidden(context_hidden, attention_mask, self.pooling)
            predicted_embedding = self.predictor(pooled_context)
            target_embedding = pool_hidden(target_hidden, target_pool_mask, self.pooling).detach()
            context_embedding = pooled_context
        return {
            "predicted_embedding": predicted_embedding,
            "target_embedding": target_embedding,
            "context_embedding": context_embedding,
            "context_hidden": context_hidden,
            "mlm_logits": self.mlm_head(context_hidden),
        }

    def regularization_batches(
        self,
        context_embedding: torch.Tensor,
        predicted_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Combine current global embeddings with a checkpointed detached history."""

        if context_embedding.ndim != 2 or predicted_embedding.ndim != 2:
            raise ValueError("Regularization embeddings must have shape [batch, hidden]")
        if context_embedding.shape != predicted_embedding.shape:
            raise ValueError("Context and predicted regularization embeddings must match")
        count = int(self.regularization_queue_count.item())
        context_history = self.context_regularization_queue[:count].detach()
        prediction_history = self.prediction_regularization_queue[:count].detach()
        context_batch = torch.cat([context_embedding, context_history], dim=0)
        prediction_batch = torch.cat([predicted_embedding, prediction_history], dim=0)
        gradient_rescale = float(context_batch.shape[0] / max(1, context_embedding.shape[0]))
        if self.training and self.regularization_queue_size > 0:
            self._enqueue_regularization_embeddings(context_embedding.detach(), predicted_embedding.detach())
        return context_batch, prediction_batch, gradient_rescale

    @torch.no_grad()
    def _enqueue_regularization_embeddings(
        self,
        context_embedding: torch.Tensor,
        predicted_embedding: torch.Tensor,
    ) -> None:
        for row in range(context_embedding.shape[0]):
            position = int(self.regularization_queue_position.item())
            self.context_regularization_queue[position].copy_(context_embedding[row])
            self.prediction_regularization_queue[position].copy_(predicted_embedding[row])
            self.regularization_queue_position.fill_((position + 1) % self.regularization_queue_size)
            self.regularization_queue_count.fill_(
                min(self.regularization_queue_size, int(self.regularization_queue_count.item()) + 1)
            )
