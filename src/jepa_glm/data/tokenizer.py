from __future__ import annotations

from typing import Protocol

import torch


class SequenceTokenizer(Protocol):
    pad_token_id: int
    mask_token_id: int

    @property
    def vocab_size(self) -> int:
        ...

    @property
    def special_token_ids(self) -> tuple[int, ...]:
        ...

    def encode(self, sequence: str, max_length: int | None = None) -> torch.Tensor:
        ...


class DnaTokenizer:
    pad_token_id = 0
    mask_token_id = 1
    unk_token_id = 2

    def __init__(self) -> None:
        self.vocab = {"[PAD]": 0, "[MASK]": 1, "N": 2, "A": 3, "C": 4, "G": 5, "T": 6}
        self.id_to_token = {value: key for key, value in self.vocab.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def special_token_ids(self) -> tuple[int, ...]:
        return (self.pad_token_id, self.mask_token_id)

    def encode(self, sequence: str, max_length: int | None = None) -> torch.Tensor:
        tokens = [self.vocab.get(base.upper(), self.unk_token_id) for base in sequence]
        if max_length is not None:
            tokens = tokens[:max_length]
            tokens += [self.pad_token_id] * max(0, max_length - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)

    def decode(self, ids: torch.Tensor) -> str:
        return "".join(self.id_to_token.get(int(idx), "N") for idx in ids if int(idx) > self.unk_token_id)


class HuggingFaceDnaTokenizer:
    def __init__(self, tokenizer_name: str, trust_remote_code: bool = True) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install the 'hf' extra to use Hugging Face tokenizers.") from exc
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
        if self.tokenizer.mask_token_id is None:
            raise ValueError(f"Tokenizer '{tokenizer_name}' does not define a mask token")

    @property
    def pad_token_id(self) -> int:
        return int(self.tokenizer.pad_token_id)

    @property
    def mask_token_id(self) -> int:
        return int(self.tokenizer.mask_token_id)

    @property
    def vocab_size(self) -> int:
        return int(len(self.tokenizer))

    @property
    def special_token_ids(self) -> tuple[int, ...]:
        return tuple(int(token_id) for token_id in self.tokenizer.all_special_ids)

    def encode(self, sequence: str, max_length: int | None = None) -> torch.Tensor:
        encoded = self.tokenizer(
            sequence,
            add_special_tokens=True,
            truncation=max_length is not None,
            max_length=max_length,
            padding="max_length" if max_length is not None else False,
            return_tensors="pt",
        )
        return encoded["input_ids"].squeeze(0).long()


def build_tokenizer(config: dict | None = None) -> SequenceTokenizer:
    config = config or {}
    if "tokenizer" in config:
        tokenizer_cfg = config["tokenizer"]
    elif "model" in config and isinstance(config["model"], dict) and "tokenizer" in config["model"]:
        tokenizer_cfg = config["model"]["tokenizer"]
    else:
        tokenizer_cfg = config
    tokenizer_name = tokenizer_cfg.get("name") or tokenizer_cfg.get("tokenizer_name")
    trust_remote_code = bool(tokenizer_cfg.get("trust_remote_code", True))
    if "model" in config and isinstance(config["model"], dict):
        model_cfg = config["model"]
        if model_cfg.get("local_model_dir"):
            tokenizer_name = model_cfg["local_model_dir"]
        trust_remote_code = bool(tokenizer_cfg.get("trust_remote_code", model_cfg.get("trust_remote_code", True)))
    tokenizer_type = str(tokenizer_cfg.get("type", "dna")).lower()
    if tokenizer_name or tokenizer_type in {"hf", "huggingface"}:
        if not tokenizer_name:
            raise ValueError("Hugging Face tokenizer config requires 'name'")
        return HuggingFaceDnaTokenizer(str(tokenizer_name), trust_remote_code=trust_remote_code)
    return DnaTokenizer()
