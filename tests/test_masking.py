from __future__ import annotations

import torch

from jepa_glm.data.masking import contiguous_span_mask, multi_region_span_mask
from jepa_glm.data.tokenizer import DnaTokenizer


def test_contiguous_span_mask_shapes_and_labels() -> None:
    tokenizer = DnaTokenizer()
    input_ids = torch.tensor([[3, 4, 5, 6, 3, 4, 5, 6]])
    context, mask, labels = contiguous_span_mask(input_ids, 0.25, 2, tokenizer.mask_token_id)
    assert context.shape == input_ids.shape
    assert mask.sum().item() == 2
    assert torch.all(context[mask] == tokenizer.mask_token_id)
    assert torch.all(labels[~mask] == -100)


def test_contiguous_span_mask_hits_exact_ratio_without_overlap() -> None:
    input_ids = torch.tensor([[3, 4, 5, 6, 3, 4, 5, 6]])

    _, mask, _ = contiguous_span_mask(input_ids, 0.75, 4, mask_token_id=1)

    assert mask.sum().item() == 6


def test_contiguous_span_mask_excludes_padding_and_special_tokens() -> None:
    input_ids = torch.tensor([[101, 3, 4, 5, 6, 3, 4, 102, 0, 0]])

    context, mask, labels = contiguous_span_mask(
        input_ids,
        0.5,
        3,
        mask_token_id=1,
        pad_token_id=0,
        excluded_token_ids=(0, 101, 102),
    )

    assert mask.sum().item() == 3
    assert not mask[0, 0]
    assert not mask[0, 7]
    assert not mask[0, 8:].any()
    assert torch.equal(context[~mask], input_ids[~mask])
    assert torch.equal(labels[mask], input_ids[mask])


def test_multi_region_mask_samples_bounded_fraction_and_regions() -> None:
    torch.manual_seed(7)
    input_ids = torch.tensor([[101, *([3, 4, 5, 6] * 16), 102, 0, 0]])

    context, mask, labels = multi_region_span_mask(
        input_ids,
        min_mask_ratio=0.2,
        max_mask_ratio=0.4,
        min_regions=1,
        max_regions=3,
        mask_token_id=1,
        pad_token_id=0,
        excluded_token_ids=(0, 101, 102),
    )

    eligible_count = 64
    masked_count = int(mask.sum().item())
    transitions = mask[0].int()[1:] - mask[0].int()[:-1]
    region_count = int((transitions == 1).sum().item())
    assert round(eligible_count * 0.2) <= masked_count <= round(eligible_count * 0.4)
    assert 1 <= region_count <= 3
    assert torch.all(context[mask] == 1)
    assert torch.equal(labels[mask], input_ids[mask])
    assert not mask[0, 0] and not mask[0, -3:].any()
