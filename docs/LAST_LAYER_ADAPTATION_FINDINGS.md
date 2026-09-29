# Last-layer adaptation findings

This note records the current evidence for moving beyond the frozen-backbone
main experiment.

## Motivation

The completed 12-run main experiment used `freeze_policy: all`, so DNABERT-2
backbone embeddings were not updated. The downstream promoter/splice and
ClinVar zero-shot variant evaluations therefore produced the same embedding
metrics across `pretrained_glm`, `mlm`, `jepa`, and `mlm_jepa`. Those results
are useful as pipeline validation, but they do not support a claim that JEPA
continual pretraining improved biological transfer.
The current artifact validator confirms that all 12 frozen-backbone main runs
exist for seeds 7, 13, and 23, and the pretraining-objective loss comparison is
statistically interpretable. This is recorded as training-protocol evidence,
not as biological representation-improvement evidence.

## Adaptation run

A last-encoder-layer adaptation ablation was added:

- Configs:
  - `configs/pretrain_dnabert2_grch38_mlm_last1.yaml`
  - `configs/pretrain_dnabert2_grch38_jepa_last1.yaml`
  - `configs/pretrain_dnabert2_grch38_last1_pilot.yaml`
- Conditions:
  - `mlm_last1`
  - `jepa_last1`
  - `mlm_jepa_last1`
- Freeze policy: `encoder_last_n`
- Trainable encoder layers: 1
- Seeds: 7, 13, 23
- Target steps: 1000
- Device: local CUDA GPU

All nine last-layer runs completed 1000 steps and passed training-log audit.
The recorded local environment uses CUDA-enabled PyTorch on an 8 GiB NVIDIA
GPU. The 8GB smoke/prototype compute profile and DNABERT-2 CUDA backward smoke
validation pass; this supports local reproducibility of the current focused
experiments, but should not be described as capacity for large-scale full
DNABERT-2 fine-tuning.

An exploratory two-axis smoke ablation over mask ratio (`0.15`, `0.30`) and
predictor depth (`1`, `2`) also passes validation with two seeds per condition
and non-collapsed embedding statistics. This is useful as a software and
sensitivity sanity check, but it is not the preregistered full ablation grid
needed for manuscript-level ablation conclusions over pooling mode and JEPA
loss weight.

## Current evaluation result

Combined downstream comparison:

- `outputs/main_experiment_last1_pilot/downstream_comparison_combined.csv`

Combined variant comparison:

- `outputs/main_experiment_last1_pilot/variant_comparison_combined.csv`

Validation outputs:

- `outputs/main_experiment_last1_pilot/downstream_comparison.validation.json`
- `outputs/main_experiment_last1_pilot/variant_comparison.validation.json`
- `outputs/main_experiment_last1_pilot/figure_readiness.json`
- `outputs/main_experiment_large_eval/downstream_comparison.validation.json`
- `outputs/main_experiment_large_eval/variant_comparison.validation.json`
- `outputs/clinvar_sequences_2k_allele_swap.validation.json`
- `outputs/main_experiment_large_eval/variant_comparison_allele_swap.validation.json`
- `outputs/main_experiment_large_eval/variant_score_control_summary.json`
- `outputs/main_experiment_large_eval/variant_score_controls/*/seed_*.json`
- `outputs/main_experiment_chr20_22_eval/variant_comparison.validation.json`
- `outputs/main_experiment_chr20_22_eval/variant_score_audits/*/seed_*.json`

At 1000 steps, `mlm_last1`, `jepa_last1`, and `mlm_jepa_last1` do not show a
robust downstream or zero-shot variant gain over the frozen-backbone controls.
Promoter AUROC is essentially unchanged, splice AUROC is slightly lower, and
ClinVar AUROC changes are very small.

## Larger-sample follow-up

The initial last-layer evaluation used 200 examples per downstream split and
400 balanced ClinVar variants. A follow-up evaluation increased the downstream
sample to 1000 examples per split across seeds 7, 13, and 23, and increased
ClinVar to 2000 balanced variants across the same three seeds.

Larger-sample outputs:

- `outputs/main_experiment_large_eval/downstream_comparison.csv`
- `outputs/main_experiment_large_eval/variant_comparison.csv`
- `outputs/main_experiment_large_eval/variant_comparison_allele_swap.csv`
- `outputs/main_experiment_large_eval/variant_effect_summary.csv`
- `outputs/clinvar_sequences_2k.csv`
- `outputs/clinvar_sequences_2k_allele_swap.csv`
- `outputs/clinvar_chr20_22_holdout.csv`
- `outputs/clinvar_chr20_22_holdout_sequences.csv`
- `outputs/main_experiment_chr20_22_eval/variant_comparison.csv`
- `outputs/main_experiment_chr20_22_eval/variant_effect_summary.csv`
- `outputs/main_experiment_variant_evidence.csv`
- `outputs/main_experiment_variant_evidence.md`
- `outputs/main_experiment_last_layer_manuscript_results.md`
- `outputs/main_experiment_variant_figures/variant_evidence_delta.png`
- `outputs/main_experiment_variant_figures/figure_readiness.json`
- `docs/LAST_LAYER_VARIANT_REPRODUCTION.md`
- `outputs/main_experiment_last_layer_variant_packet/packet_manifest.json`
- `outputs/main_experiment_last_layer_variant_packet.zip`
- `outputs/main_experiment_last_layer_variant_packet.verification.json`
- `outputs/main_experiment_last_layer_variant_packet.semantic_validation.json`
- `outputs/manuscript_gap_audit.json`
- `outputs/manuscript_gap_audit.md`

The larger downstream evaluation remains neutral: promoter and splice AUROC are
nearly unchanged across pretrained, MLM-last1, JEPA-last1, and MLM+JEPA-last1
representations. The 3-seed downstream comparison validator passes, but the
downstream effect-signal validator records `no_claim_signal` because no
candidate condition reaches the preregistered absolute AUROC delta threshold
of `0.01` on either task. A shuffled-label promoter negative control using the
same pretrained seed-7 probe setup yields near-random AUROC (`0.5045`) and
passes the negative-control validator, supporting the evaluation pipeline while
still not supporting a positive downstream-transfer claim.

The larger ClinVar zero-shot evaluation shows a small replicated cosine-score
signal. Across three seeds on 2000 balanced variants:

- pretrained cosine AUROC: 0.5288
- MLM-last1 cosine AUROC: 0.5312
- JEPA-last1 cosine AUROC: 0.5395
- MLM+JEPA-last1 cosine AUROC: 0.5390

The same 2000-variant panel now has an allele-swap control in which
`ref_sequence` and `alt_sequence` are exchanged while labels are preserved.
The swapped sequence manifest and validator pass with 2000 mirrored rows. All
12 original/swapped score pairs across four conditions and three seeds pass
the score-symmetry control at tolerance `1e-6`, and the aggregated swapped
comparison matches the original comparison. This rules out a simple ref/alt
ordering artifact in the current zero-shot score implementation.

This remains a candidate positive signal for variant sensitivity, not yet a
strong broad biological claim. It should be treated as hypothesis-generating
until confirmed with a larger variant set and preferably a held-out ClinVar
date or chromosome split.

## Chromosome held-out ClinVar follow-up

The local ClinVar CSV contains `chrom`, `pos`, `ref`, `alt`, and `label`
columns but does not contain release-date or review-date metadata. Therefore a
date-based held-out split cannot be produced from the current local asset
without downloading or reconstructing richer ClinVar metadata. As a local
no-download alternative, a balanced 2000-variant chromosome holdout was created
from `chr20`, `chr21`, and `chr22`.

Holdout outputs:

- `outputs/clinvar_chr20_22_holdout_manifest.json`
- `outputs/clinvar_chr20_22_holdout_sequences.summary.json`
- `outputs/main_experiment_chr20_22_eval/variant_comparison.csv`
- `outputs/main_experiment_chr20_22_eval/variant_comparison.validation.json`
- `outputs/main_experiment_chr20_22_eval/variant_score_audits/*/seed_*.json`

The holdout manifest records 2000 variants with balanced labels
(`1000` benign/likely benign and `1000` pathogenic/likely pathogenic). Across
three seeds on this chromosome holdout:

- pretrained cosine AUROC: 0.5082
- MLM-last1 cosine AUROC: 0.5096
- JEPA-last1 cosine AUROC: 0.5133
- MLM+JEPA-last1 cosine AUROC: 0.5131

The direction of the cosine-score signal is consistent with the broader
ClinVar 2k evaluation, but the magnitude is smaller: JEPA-last1 improves by
about `+0.0052` AUROC over pretrained on this holdout. L2 remains near or
below chance and should not be used as positive evidence.

## Manuscript wording guard

The consolidated variant evidence table is written to:

- `outputs/main_experiment_variant_evidence.csv`
- `outputs/main_experiment_variant_evidence.json`
- `outputs/main_experiment_variant_evidence.md`
- `outputs/main_experiment_last_layer_manuscript_results.md`
- `outputs/main_experiment_variant_figures/variant_evidence_delta.png`
- `outputs/main_experiment_variant_figures/figure_readiness.json`
- `docs/LAST_LAYER_VARIANT_REPRODUCTION.md`
- `outputs/main_experiment_last_layer_variant_packet.zip`
- `outputs/manuscript_gap_audit.md`

This table labels JEPA-last1 and MLM+JEPA-last1 cosine results on the broader
ClinVar 2k panel as `directional_candidate_only`, and labels the JEPA-last1
cosine result on the chr20-22 holdout as `directional_candidate_only`.
Everything else is `neutral_or_tiny`. The intended manuscript wording is
therefore a small candidate cosine-score signal, not zero-shot variant-effect
prediction and not robust pathogenicity discrimination.

## Paper-positioning implication

The current evidence supports a conservative methods/result statement:

> We validated the full local GRCh38/DNABERT-2 JEPA-GLM pipeline, including
> multi-seed frozen-backbone controls and a local-GPU last-layer adaptation
> ablation across MLM-only, JEPA-only, and MLM+JEPA objectives. Under the
> current 1000-step last-layer adaptation setting, promoter/splice transfer
> gains were not observed, while a small replicated ClinVar cosine-score signal
> emerged in a larger 2000-variant follow-up and passed same-panel allele-swap
> score-symmetry controls. A chromosome-held-out ClinVar follow-up on
> chr20-22 preserved the same direction with a smaller effect.

It does not yet support a broad positive biological representation-improvement
claim. The next paper-grade experiment should extend the larger ClinVar
evaluation, add a date-based ClinVar holdout or independent MPRA/eQTL variant
benchmark, complete preregistered ablations, and pre-register a selection rule
based on held-out downstream and variant metrics.
