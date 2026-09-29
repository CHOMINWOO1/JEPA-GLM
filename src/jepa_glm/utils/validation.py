from __future__ import annotations


def validate_config(config: dict) -> None:
    training_cfg = config.get("training", {})
    if training_cfg.get("optimizer_steps") is not None and training_cfg.get("steps") is not None:
        raise ValueError("training.optimizer_steps and training.steps are mutually exclusive")
    if training_cfg.get("optimizer_steps") is not None and int(training_cfg["optimizer_steps"]) <= 0:
        raise ValueError("training.optimizer_steps must be positive")
    for key in ["learning_rate", "backbone_learning_rate", "predictor_learning_rate", "mlm_head_learning_rate"]:
        if training_cfg.get(key) is not None and float(training_cfg[key]) <= 0.0:
            raise ValueError(f"training.{key} must be positive")
    optimizer_name = str(training_cfg.get("optimizer", "adamw")).lower()
    if optimizer_name not in {"adamw", "sgd"}:
        raise ValueError("training.optimizer must be 'adamw' or 'sgd'")
    scheduler_name = str(training_cfg.get("scheduler", "")).lower()
    if scheduler_name not in {"", "none", "cosine", "warmup_cosine"}:
        raise ValueError("training.scheduler must be 'cosine' or 'warmup_cosine'")
    if scheduler_name == "warmup_cosine":
        total_steps = int(training_cfg.get("optimizer_steps", training_cfg.get("steps", 0)))
        warmup_steps = int(training_cfg.get("predictor_warmup_steps", 0)) + int(
            training_cfg.get("joint_warmup_steps", 0)
        )
        if total_steps <= warmup_steps:
            raise ValueError("warmup_cosine requires optimizer steps after warmup")
    rc_probability = float(config.get("data", {}).get("reverse_complement_probability", 0.0))
    if not 0.0 <= rc_probability <= 1.0:
        raise ValueError("data.reverse_complement_probability must be in [0, 1]")
    model_cfg = config.get("model", config)
    tokenizer_cfg = model_cfg.get("tokenizer", config.get("tokenizer", {}))
    backbone_name = model_cfg.get("backbone_name")
    tokenizer_type = str(tokenizer_cfg.get("type", "dna")).lower()
    if backbone_name and tokenizer_type == "dna":
        raise ValueError("A Hugging Face backbone requires a matching Hugging Face tokenizer config")
    if tokenizer_type in {"hf", "huggingface"} and not tokenizer_cfg.get("name"):
        raise ValueError("Hugging Face tokenizer config requires model.tokenizer.name")
    freeze_policy = str(model_cfg.get("freeze_policy", "none")).lower()
    allowed = {"none", "full_unfreeze", "unfrozen", "all", "frozen", "freeze", "embeddings", "freeze_embeddings", "encoder_last_n", "last_n"}
    if freeze_policy not in allowed:
        raise ValueError(f"Unknown freeze_policy: {freeze_policy}")
    if freeze_policy in {"encoder_last_n", "last_n"} and int(model_cfg.get("trainable_last_n_layers", 0)) <= 0:
        raise ValueError("encoder_last_n freeze policy requires trainable_last_n_layers > 0")
    target_mode = str(model_cfg.get("jepa_target_mode", "pooled")).lower()
    if target_mode not in {"pooled", "masked_tokens", "cls"}:
        raise ValueError("model.jepa_target_mode must be 'pooled', 'masked_tokens', or 'cls'")
    if target_mode == "cls":
        predictor_dim = int(model_cfg.get("predictor_embed_dim", 384))
        predictor_heads = int(model_cfg.get("predictor_heads", 2))
        if predictor_dim <= 0 or predictor_heads <= 0 or predictor_dim % predictor_heads != 0:
            raise ValueError("CLS predictor_embed_dim must be positive and divisible by predictor_heads")
        if int(model_cfg.get("regularization_queue_size", 0)) < 2:
            raise ValueError("CLS mode requires regularization_queue_size >= 2")
    data_cfg = config.get("data", {})
    mask_strategy = str(data_cfg.get("mask_strategy", data_cfg.get("mask_type", "contiguous_span"))).lower()
    if mask_strategy in {"multi_region", "multi_region_span"}:
        min_ratio = float(data_cfg.get("min_mask_ratio", 0.2))
        max_ratio = float(data_cfg.get("max_mask_ratio", 0.4))
        min_regions = int(data_cfg.get("min_mask_regions", 1))
        max_regions = int(data_cfg.get("max_mask_regions", 3))
        if not 0.0 < min_ratio <= max_ratio <= 1.0:
            raise ValueError("multi-region mask ratios must satisfy 0 < min <= max <= 1")
        if min_regions < 1 or max_regions < min_regions:
            raise ValueError("multi-region counts must satisfy 1 <= min <= max")
    if bool(model_cfg.get("use_pretrained_mlm_head", False)) and not backbone_name:
        raise ValueError("model.use_pretrained_mlm_head requires a Hugging Face backbone")
    loss_type = str(config.get("loss", {}).get("jepa_loss", "cosine")).lower()
    if loss_type not in {"cosine", "mse"}:
        raise ValueError("loss.jepa_loss must be 'cosine' or 'mse'")
