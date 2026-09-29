# Last-Layer ClinVar Evidence Reproduction Runbook

This runbook reproduces the focused last-layer ClinVar evidence packet. It is
scoped to the current manuscript-safe result:

> The exploratory ClinVar cosine-score candidate did not replicate on the
> preregistered, disjoint 10,000-variant panel and is not an allowed positive
> claim.

It does not reproduce or validate a broad claim that JEPA improves genomic
representation learning, promoter/splice transfer, or zero-shot pathogenicity
prediction.

## Local Inputs

Required local files:

- `data/clinvar.csv`
- `data/GRCh38.fa`
- `models/dnabert2-117m`
- `configs/clinvar_replication_10k.json`
- `outputs/main_experiment/pretrained_glm/seed_{7,13,23}/checkpoint.pt`
- `outputs/main_experiment_last1_pilot/{mlm_last1,jepa_last1,mlm_jepa_last1}/seed_{7,13,23}/checkpoint.pt`

No network download is required for the commands below.

The exact bytes of the four-condition, three-seed checkpoint set are recorded
in `main_experiment_reported_checkpoint_lock.json` with SHA-256 values and run
manifest metadata. Regenerate that lock after replacing any checkpoint.

This runbook has two distinct modes:

- Packet/manuscript regeneration: rebuilds summaries, figures, audits, the focused
  packet, semantic validation, and zip verification from already staged local
  assets. This is the expected reviewer reproduction path and requires no new
  downloads.
- Full data/model restaging or new training: may require the recorded acquisition
  commands, source-manifest updates, and additional compute. Do not describe this
  runbook as proving that a fresh machine can train the full JEPA-GLM experiment
  without downloads.

Local GPU use is optional for packet/manuscript regeneration. The authoritative
evaluation records include 24 fresh representation cells and 12 fresh ClinVar
replication cells on the RTX 5060 Ti 8GB. For training, the recorded 8GB profile
supports only short-sequence CUDA smoke tests with memory-saving settings; it
does not establish full-scale pretraining or paper-grade ablation capacity.

## Recommended Reviewer Reproduction Order

1. Confirm the local inputs above and inspect `source_manifest.validation.json`,
   `asset_lock.validation.json`, `environment_check.json`, and
   `dnabert2_model_smoke.validation.json`.
2. Recreate ClinVar panels only if the local `outputs/clinvar_*` tables need to
   be regenerated.
3. Re-run variant scoring only if the checkpoint-derived metric JSON files are
   being regenerated; otherwise use the staged metric roots.
4. Validate the repeated-seed promoter/splice representation diagnostics; rerun
   embedding extraction only if checkpoints or benchmark rows changed.
5. Rebuild manuscript artifacts with the commands in this runbook.
6. Build the focused packet, then run `verify_packet` and
   `validate_last_layer_variant_packet`.
7. Treat `downstream_effect.level: fail` with `no_claim_signal` and
   `representation_suite.level: fail` with `no_representation_claim_signal` as negative
   claim boundary, not as a failed reproduction.
8. Confirm `replication_suite.level: pass` with `no_replication_signal`; this is
   the claim-determining valid negative replication, not a pipeline failure.
9. Run `audit_variant_replication_robustness` on the locked score arrays and
   confirm `meaningful_threshold_excluded`; keep this result explicitly post hoc.
10. Do not promote claims unless the corresponding semantic gate explicitly
   passes in `main_experiment_last_layer_variant_packet.semantic_validation.json`.

## Recreate The ClinVar Panels

```powershell
.\.venv\Scripts\python.exe -m jepa_glm.cli.convert_variants --input-csv data\clinvar.csv --fasta data\GRCh38.fa --flank 64 --max-rows 2000 --balanced-labels --output-csv outputs\clinvar_sequences_2k.csv --summary-json outputs\clinvar_sequences_2k.summary.json
.\.venv\Scripts\python.exe -m jepa_glm.cli.make_variant_allele_swap_control --input-csv outputs\clinvar_sequences_2k.csv --output-csv outputs\clinvar_sequences_2k_allele_swap.csv --manifest-json outputs\clinvar_sequences_2k_allele_swap_manifest.json
.\.venv\Scripts\python.exe -m jepa_glm.cli.validate_variant_allele_swap_control --source-csv outputs\clinvar_sequences_2k.csv --swapped-csv outputs\clinvar_sequences_2k_allele_swap.csv --manifest-json outputs\clinvar_sequences_2k_allele_swap_manifest.json --output-json outputs\clinvar_sequences_2k_allele_swap.validation.json
.\.venv\Scripts\python.exe -m jepa_glm.cli.make_variant_chromosome_holdout --input-csv data\clinvar.csv --output-csv outputs\clinvar_chr20_22_holdout.csv --manifest-json outputs\clinvar_chr20_22_holdout_manifest.json --holdout-chromosomes chr20,chr21,chr22 --max-rows 2000 --balanced-labels
.\.venv\Scripts\python.exe -m jepa_glm.cli.convert_variants --input-csv outputs\clinvar_chr20_22_holdout.csv --fasta data\GRCh38.fa --flank 64 --output-csv outputs\clinvar_chr20_22_holdout_sequences.csv --summary-json outputs\clinvar_chr20_22_holdout_sequences.summary.json
```

## Evaluate Variant Scores

Run `jepa_glm.cli.evaluate_variants` for each condition and seed. Use:

- original ClinVar 2k input: `outputs\clinvar_sequences_2k.csv`
- allele-swap input: `outputs\clinvar_sequences_2k_allele_swap.csv`
- chromosome holdout input: `outputs\clinvar_chr20_22_holdout_sequences.csv`

The checkpoint mapping is:

- `pretrained_glm`: `outputs\main_experiment\pretrained_glm\seed_<SEED>\checkpoint.pt`
- other last-layer conditions: `outputs\main_experiment_last1_pilot\<CONDITION>\seed_<SEED>\checkpoint.pt`

Use the same evaluation settings for all runs:

```powershell
--local-model-dir models\dnabert2-117m --freeze-policy all --trust-remote-code --local-files-only --max-length 256 --batch-size 16 --bootstrap-samples 100 --seed <SEED>
```

## Reproduce The Disjoint 10k Replication

The protocol is locked by `configs\clinvar_replication_10k.json`. Panel creation
uses deterministic hash ranking, balances 5,000 benign and 5,000 pathogenic
rows, excludes the discovery-panel prefixes and chr20/21/22 holdout, and checks
for zero sequence overlap. All inputs are already local.

```powershell
.\.venv\Scripts\python.exe -m jepa_glm.cli.make_variant_replication_panel --protocol-json configs\clinvar_replication_10k.json --raw-output-csv outputs\clinvar_replication_10k_raw.csv --sequence-output-csv outputs\clinvar_replication_10k_sequences.csv --manifest-json outputs\clinvar_replication_10k_manifest.json
.\.venv\Scripts\python.exe -m jepa_glm.cli.run_variant_replication_suite --protocol-json configs\clinvar_replication_10k.json --panel-manifest-json outputs\clinvar_replication_10k_manifest.json --input-csv outputs\clinvar_replication_10k_sequences.csv --output-root outputs\main_experiment_variant_replication_10k --device cuda --batch-size 32
.\.venv\Scripts\python.exe -m jepa_glm.cli.validate_variant_replication_suite --protocol-json configs\clinvar_replication_10k.json --panel-manifest-json outputs\clinvar_replication_10k_manifest.json --output-root outputs\main_experiment_variant_replication_10k
.\.venv\Scripts\python.exe -m jepa_glm.cli.audit_variant_replication_robustness --bootstrap-samples 2000 --permutations 2000 --output-root outputs\main_experiment_variant_replication_10k_robustness
```

For the faster no-GPU reviewer path, skip panel creation and scoring and run only
the validator against the staged local outputs. The authoritative fresh CUDA run
completed all 12 condition/seed cells. JEPA cosine AUROC was 0.512858 and
MLM+JEPA cosine AUROC was 0.512685, both below the fixed 0.55 gate. The result is
`no_replication_signal`; positive paired deltas do not override that absolute
performance gate. L2 AUROCs remained within the preregistered 0.47-0.53
specificity range.

The robustness command uses the staged `scores.npz` arrays and runs on CPU; it
does not download assets or refit a model. It is a post hoc analysis of locked
preregistered outputs, so it does not change the primary promotion gate. With
2,000 hierarchical bootstrap samples, the candidate rank-ensemble AUROCs were
0.512860 and 0.512686 and their upper 95% limits were 0.524360 and 0.524221,
excluding 0.55. Variant-type-conditioned AUROCs were 0.489746 and 0.489590;
within-type label permutations were not significant after Holm adjustment.
The validation level is `meaningful_threshold_excluded`. These findings support
the non-replication and identify cross-variant-type composition as a confounding
contribution; they do not support a positive or equivalence claim.

## Aggregate And Validate

```powershell
.\.venv\Scripts\python.exe -m jepa_glm.cli.make_downstream_comparison --metric-dir promoter=outputs\main_experiment_large_eval\downstream\promoter --metric-dir splice=outputs\main_experiment_large_eval\downstream\splice --n-seeds 3 --output-csv outputs\main_experiment_large_eval\downstream_comparison.csv
.\.venv\Scripts\python.exe -m jepa_glm.cli.validate_downstream_comparison --comparison-csv outputs\main_experiment_large_eval\downstream_comparison.csv --required-tasks promoter,splice --required-conditions pretrained_glm,mlm_last1,jepa_last1,mlm_jepa_last1 --min-seeds 3 --output-json outputs\main_experiment_large_eval\downstream_comparison.validation.json
.\.venv\Scripts\python.exe -m jepa_glm.cli.make_shuffled_label_control --input-csv outputs\promoter_split.csv --output-csv outputs\promoter_split_shuffled_labels.csv --manifest-json outputs\promoter_split_shuffled_labels_manifest.json --seed 7
.\.venv\Scripts\python.exe -m jepa_glm.cli.run_linear_probe --config configs\downstream_promoter_dnabert2_smoke.yaml --input-csv outputs\promoter_split_shuffled_labels.csv --checkpoint outputs\main_experiment\pretrained_glm\seed_7\checkpoint.pt --seed 7 --bootstrap-samples 100 --max-examples-per-split 1000 --output-json outputs\promoter_probe_shuffled_labels.json
.\.venv\Scripts\python.exe -m jepa_glm.cli.validate_negative_control --metric-json outputs\main_experiment_large_eval\downstream\promoter\pretrained_glm\seed_7.json --control-json outputs\promoter_probe_shuffled_labels.json --metric auroc --max-control-metric 0.65 --min-metric-gap 0.05 --output-json outputs\negative_control.validation.json
.\.venv\Scripts\python.exe -m jepa_glm.cli.make_variant_comparison --metric-dir outputs\main_experiment_large_eval\variants --output-csv outputs\main_experiment_large_eval\variant_comparison.csv
.\.venv\Scripts\python.exe -m jepa_glm.cli.validate_variant_comparison --comparison-csv outputs\main_experiment_large_eval\variant_comparison.csv --required-scores l2,cosine --required-conditions pretrained_glm,mlm_last1,jepa_last1,mlm_jepa_last1 --min-seeds 3 --output-json outputs\main_experiment_large_eval\variant_comparison.validation.json
.\.venv\Scripts\python.exe -m jepa_glm.cli.make_variant_comparison --metric-dir outputs\main_experiment_large_eval\variants_allele_swap --output-csv outputs\main_experiment_large_eval\variant_comparison_allele_swap.csv
.\.venv\Scripts\python.exe -m jepa_glm.cli.validate_variant_comparison --comparison-csv outputs\main_experiment_large_eval\variant_comparison_allele_swap.csv --required-scores l2,cosine --required-conditions pretrained_glm,mlm_last1,jepa_last1,mlm_jepa_last1 --min-seeds 3 --output-json outputs\main_experiment_large_eval\variant_comparison_allele_swap.validation.json
.\.venv\Scripts\python.exe -m jepa_glm.cli.make_variant_comparison --metric-dir outputs\main_experiment_chr20_22_eval\variants --output-csv outputs\main_experiment_chr20_22_eval\variant_comparison.csv
.\.venv\Scripts\python.exe -m jepa_glm.cli.validate_variant_comparison --comparison-csv outputs\main_experiment_chr20_22_eval\variant_comparison.csv --required-scores l2,cosine --required-conditions pretrained_glm,mlm_last1,jepa_last1,mlm_jepa_last1 --min-seeds 3 --output-json outputs\main_experiment_chr20_22_eval\variant_comparison.validation.json
```

The downstream effect-signal validator is expected to exit nonzero for the
current evidence because the 3-seed downstream comparison is neutral. Preserve
the JSON as a claim boundary artifact:

```powershell
.\.venv\Scripts\python.exe -m jepa_glm.cli.validate_downstream_effect_signal --comparison-csv outputs\main_experiment_large_eval\downstream_comparison.csv --baseline pretrained_glm --reference-condition mlm_last1 --candidate jepa_last1 --candidate mlm_jepa_last1 --metric auroc --min-abs-delta 0.01 --min-tasks-with-signal 1 --output-json outputs\main_experiment_large_eval\downstream_effect_signal.json
```

The final row should be `downstream_effect.level: fail` with
`no_claim_signal`; this is a negative claim gate, not a failed reproduction.
Promotion to a downstream-transfer claim requires a positive delta against
both the pretrained GLM and MLM comparator.

Validate all 12 original/swap score-control pairs, then summarize them:

```powershell
.\.venv\Scripts\python.exe -m jepa_glm.cli.make_control_summary --input-file outputs\main_experiment_large_eval\variant_score_controls\pretrained_glm\seed_7.json --input-file outputs\main_experiment_large_eval\variant_score_controls\pretrained_glm\seed_13.json --input-file outputs\main_experiment_large_eval\variant_score_controls\pretrained_glm\seed_23.json --input-file outputs\main_experiment_large_eval\variant_score_controls\mlm_last1\seed_7.json --input-file outputs\main_experiment_large_eval\variant_score_controls\mlm_last1\seed_13.json --input-file outputs\main_experiment_large_eval\variant_score_controls\mlm_last1\seed_23.json --input-file outputs\main_experiment_large_eval\variant_score_controls\jepa_last1\seed_7.json --input-file outputs\main_experiment_large_eval\variant_score_controls\jepa_last1\seed_13.json --input-file outputs\main_experiment_large_eval\variant_score_controls\jepa_last1\seed_23.json --input-file outputs\main_experiment_large_eval\variant_score_controls\mlm_jepa_last1\seed_7.json --input-file outputs\main_experiment_large_eval\variant_score_controls\mlm_jepa_last1\seed_13.json --input-file outputs\main_experiment_large_eval\variant_score_controls\mlm_jepa_last1\seed_23.json --required-count 12 --output-json outputs\main_experiment_large_eval\variant_score_control_summary.json
```

## Evaluate Representation Diagnostics

The reported representation suite uses the local GPU for embedding extraction
but requires no network download. It evaluates promoter and splice splits for
four conditions and three seeds, with at most 1,000 examples per split:

```powershell
.\.venv\Scripts\python.exe -m jepa_glm.cli.run_representation_suite --conditions "pretrained_glm,mlm_last1,jepa_last1,mlm_jepa_last1" --seeds "7,13,23" --max-examples-per-split 1000 --batch-size 32 --device cuda --output-root outputs\main_experiment_representation
.\.venv\Scripts\python.exe -m jepa_glm.cli.validate_representation_suite --output-root outputs\main_experiment_representation
```

The current RTX 5060 Ti 8GB run completed 24 task/condition/seed cells. The
integrity checks passed for coverage, finite metrics, non-collapse, and CKA
bounds. The claim level is `no_representation_claim_signal`: JEPA promoter k-NN
AUROC was 0.811585 with delta +0.000843 (95% paired t CI -0.006499 to
+0.008185), while splice k-NN AUROC was 0.678297 with delta -0.004109 (95%
paired t CI -0.012624 to +0.004406). Linear CKA to pretrained was 0.994370 for
promoter and 0.994866 for splice. These values are negative-boundary evidence,
not support for broad representation improvement.

`representation_run_manifest.json` records the resolved CUDA device, GPU name,
PyTorch/CUDA versions, timing, all 24 freshly extracted cells, and 3,000
examples per cell. Use `--resume` only for an incremental local check; omit it
when regenerating the authoritative extraction-provenance manifest.

## Build Manuscript Artifacts

```powershell
.\.venv\Scripts\python.exe -m jepa_glm.cli.make_reported_checkpoint_lock --output-json outputs\main_experiment_reported_checkpoint_lock.json --output-md outputs\main_experiment_reported_checkpoint_lock.md
.\.venv\Scripts\python.exe -m jepa_glm.cli.make_variant_evidence_summary --comparison clinvar_2k=outputs\main_experiment_large_eval\variant_comparison.csv --comparison clinvar_chr20_22=outputs\main_experiment_chr20_22_eval\variant_comparison.csv --seed-metric-root clinvar_2k=outputs\main_experiment_large_eval --seed-metric-root clinvar_chr20_22=outputs\main_experiment_chr20_22_eval --output-csv outputs\main_experiment_variant_evidence.csv --output-json outputs\main_experiment_variant_evidence.json --output-md outputs\main_experiment_variant_evidence.md
.\.venv\Scripts\python.exe -m jepa_glm.cli.make_claim_matrix --output-json outputs\claim_evidence_matrix.json --output-md outputs\claim_evidence_matrix.md
.\.venv\Scripts\python.exe -m jepa_glm.cli.make_claim_wording_guard --matrix-json outputs\claim_evidence_matrix.json --output-json outputs\claim_wording_guard.json --output-md outputs\claim_wording_guard.md
.\.venv\Scripts\python.exe -m jepa_glm.cli.make_manuscript_scope_decision --claim-wording-json outputs\claim_wording_guard.json --variant-evidence-json outputs\main_experiment_variant_evidence.json --downstream-effect-json outputs\main_experiment_large_eval\downstream_effect_signal.json --downstream-effect-json outputs\paper_grade_8gb_pilot\downstream_effect_signal_mask50_last2_1000.json --downstream-timecourse-json outputs\paper_grade_8gb_pilot\downstream_timecourse_mask50_last2.json --output-json outputs\manuscript_scope_decision.json --output-md outputs\manuscript_scope_decision.md --checks-json outputs\manuscript_scope_decision.validation.json
.\.venv\Scripts\python.exe -m jepa_glm.cli.make_figures --variant-evidence-csv outputs\main_experiment_variant_evidence.csv --output-dir outputs\main_experiment_variant_figures --manifest-json outputs\main_experiment_variant_figures\figure_manifest.json
.\.venv\Scripts\python.exe -m jepa_glm.cli.make_last_layer_manuscript_results --output-md outputs\main_experiment_last_layer_manuscript_results.md
.\.venv\Scripts\python.exe -m jepa_glm.cli.validate_figure_readiness --figure-manifest-json outputs\main_experiment_variant_figures\figure_manifest.json --paper-report-md outputs\main_experiment_last_layer_manuscript_results.md --required-figure variant_evidence_delta.png --required-table variant --required-table clinvar --output-json outputs\main_experiment_variant_figures\figure_readiness.json
```

`make_variant_evidence_summary` computes the primary delta confidence interval
from paired seed-level differences against `pretrained_glm` when
`--seed-metric-root` is supplied. The comparison-summary CI is retained as a
secondary conservative audit column.

`make_manuscript_scope_decision` records the current manuscript boundary: the
exploratory ClinVar cosine-score candidate is blocked after failed replication,
and downstream-transfer and broad representation-learning wording also remain
blocked under the recorded evidence.

The focused packet includes `manuscript_scope_decision.json`,
`manuscript_scope_decision.md`, and
`manuscript_scope_decision.validation.json`. The current packet manifest
records the authoritative file count and SHA-256 values; do not rely on a
manually copied narrative count. The semantic gate checks that the scope
decision blocks the exploratory variant candidate after replication, together
with downstream-transfer and broad representation-learning wording.

## Reproduction Completion Checklist

|check|expected evidence|meaning|
|---|---|---|
|No-network packet path|`LAST_LAYER_VARIANT_REPRODUCTION.md` states that no network download is required for the commands below.|Reviewer can rebuild the current focused packet from staged local assets.|
|Source and asset locks|`source_manifest.validation.json` and `asset_lock.validation.json` pass.|Reported sources and local staged files are traceable.|
|GPU boundary|`representation_run_manifest.json` records 24 fresh CUDA cells and `replication_run_manifest.json` records 12 fresh CUDA cells on the RTX 5060 Ti 8GB; `compute_readiness_8gb_smoke.json` retains the training boundary.|Local repeated-seed evaluation is reproducible, but it does not become a full-pretraining or paper-grade ablation-training claim.|
|Checkpoint identity|`main_experiment_reported_checkpoint_lock.json` contains 12 passing condition/seed rows with checkpoint SHA-256 values.|The reported local checkpoints are byte-identifiable without redistributing them.|
|Paired uncertainty|`main_experiment_variant_evidence.csv` includes paired-seed delta CI columns generated with `--seed-metric-root`.|Candidate deltas are traceable to seed-level paired differences.|
|Disjoint replication boundary|`replication_validation.json` passes panel integrity, zero overlap, CUDA freshness, L2 specificity, and valid-negative gates with `no_replication_signal`.|The discovery-panel candidate is exploratory only and is blocked as a positive manuscript claim.|
|Locked-score robustness boundary|`robustness_validation.json` records `meaningful_threshold_excluded`; candidate upper confidence limits are below 0.55 and variant-type-conditioned AUROCs are near chance.|This post hoc audit strengthens the negative boundary and exposes cross-variant-type composition without changing the preregistered endpoint.|
|Negative downstream boundary|`downstream_effect.level` records `fail` with `no_claim_signal`.|Neutral downstream transfer blocks broad transfer claims.|
|Representation boundary|`representation_suite.level` records `fail` with `no_representation_claim_signal`, while coverage, finite-metric, non-collapse, CKA, and GPU-provenance checks pass.|Repeated-seed diagnostics do not support broad representation improvement.|
|Packet integrity|`verify_packet` passes against `packet_manifest.json` and the zip file.|The packet contents match manifest hashes and zip contents.|
|Semantic claim gate|`focused_packet.level` reports `last_layer_variant_packet_ready`.|Audited non-replication wording is allowed; the exploratory candidate and broad claims are blocked.|

## Build And Validate The Focused Packet

```powershell
.\.venv\Scripts\python.exe -m jepa_glm.cli.make_last_layer_variant_packet --output-dir outputs\main_experiment_last_layer_variant_packet --zip-path outputs\main_experiment_last_layer_variant_packet.zip
.\.venv\Scripts\python.exe -m jepa_glm.cli.verify_packet --packet-dir outputs\main_experiment_last_layer_variant_packet --zip-path outputs\main_experiment_last_layer_variant_packet.zip --output-json outputs\main_experiment_last_layer_variant_packet.verification.json
.\.venv\Scripts\python.exe -m jepa_glm.cli.validate_last_layer_variant_packet --packet-dir outputs\main_experiment_last_layer_variant_packet --output-json outputs\main_experiment_last_layer_variant_packet.semantic_validation.json
.\.venv\Scripts\python.exe -m jepa_glm.cli.make_manuscript_gap_audit --claim-wording-json outputs\claim_wording_guard.json --focused-packet-validation-json outputs\main_experiment_last_layer_variant_packet.semantic_validation.json --output-json outputs\manuscript_gap_audit.json --output-md outputs\manuscript_gap_audit.md
```

The final semantic gate must report:

```text
focused_packet.level: pass
last_layer_variant_packet_ready
```
