# JEPA-GLM Experiment Protocol

This protocol defines the minimum artifact trail required before a JEPA-GLM run should be treated as paper-reportable.

## 1. Pretraining Data

Check that the local environment can run the intended model before downloading large data or launching pretraining:

```bash
jepa-glm-check-environment --config configs/pretrain_dnabert2_hf.yaml --require-cuda --require-hf --output-json outputs/environment_check.json
jepa-glm-check-compute --config configs/pretrain_dnabert2_hf.yaml --output-json outputs/compute_readiness.json
jepa-glm-validate-environment-transition --before-json outputs/environment_check.json --after-json outputs/environment_check.after.json --output-json outputs/environment_transition.json --output-md outputs/environment_transition.md
jepa-glm-validate-source-manifest --manifest configs/paper_sources.yaml --output-json outputs/source_manifest.validation.json
jepa-glm-make-source-citation-table --source-manifest configs/paper_sources.yaml --output-json outputs/source_citation_table.json --output-md outputs/source_citation_table.md
jepa-glm-make-asset-lock --source-manifest configs/paper_sources.yaml --output-json outputs/asset_lock.json
jepa-glm-validate-asset-lock --asset-lock-json outputs/asset_lock.json --output-json outputs/asset_lock.validation.json
jepa-glm-make-asset-staging-runbook --source-manifest configs/paper_sources.yaml --output-md outputs/asset_staging_runbook.md --output-ps1 outputs/asset_staging_runbook.ps1
jepa-glm-make-asset-bootstrap --source-manifest configs/paper_sources.yaml --output-ps1 outputs/asset_bootstrap.ps1 --output-md outputs/asset_bootstrap.md
jepa-glm-validate-asset-execution-log --log-path outputs/asset_bootstrap.execution.jsonl --expected-source-manifest configs/paper_sources.yaml --expected-mode bootstrap --output-json outputs/asset_bootstrap.execution.validation.json --output-md outputs/asset_bootstrap.execution.validation.md
jepa-glm-validate-asset-execution-log --log-path outputs/asset_staging.execution.jsonl --expected-source-manifest configs/paper_sources.yaml --expected-mode staging --output-json outputs/asset_staging.execution.validation.json --output-md outputs/asset_staging.execution.validation.md
jepa-glm-check-asset-resource-budget --source-manifest configs/paper_sources.yaml --workspace . --output-json outputs/asset_resource_budget.json --output-md outputs/asset_resource_budget.md
jepa-glm-audit-model-cache --model-dir models/dnabert2-117m --output-json outputs/model_cache_audit.json --output-md outputs/model_cache_audit.md
jepa-glm-make-acquisition-plan --source-manifest configs/paper_sources.yaml --output-md outputs/acquisition_plan.md
jepa-glm-make-acquisition-ledger --source-manifest configs/paper_sources.yaml --output-json outputs/acquisition_ledger.json --output-md outputs/acquisition_ledger.md
jepa-glm-validate-acquisition-ledger --ledger-json outputs/acquisition_ledger.json --output-json outputs/acquisition_ledger.validation.json
jepa-glm-check-assets --config outputs/configs/dnabert2_grch38_train.yaml --require-local-model --output-json outputs/asset_check.json
jepa-glm-make-environment-runbook --environment-json outputs/environment_check.json --output-md outputs/environment_runbook.md
jepa-glm-make-local-gpu-runbook --environment-json outputs/environment_check.json --compute-json outputs/compute_readiness.json --output-md outputs/local_gpu_runbook.md
jepa-glm-check-launch-preflight --config configs/pretrain_dnabert2_hf.yaml --output-json outputs/launch_preflight.json --output-md outputs/launch_preflight.md
jepa-glm-validate-staging-completion --output-json outputs/staging_completion.json --output-md outputs/staging_completion.md
```

For DNABERT-2 experiments, `transformers` and `einops` must be installed and PyTorch must be a CUDA-enabled build if local GPU training is expected. The compute report records visible GPU suitability and whether the initial local config uses conservative settings such as small micro-batches, mixed precision, checkpointing, accumulation, and partial freezing. If CUDA checks fail, the local GPU runbook records the setup and verification sequence. After installation, rerun the environment check to `outputs/environment_check.after.json` and validate the environment transition so the packet records exactly what changed. The launch preflight is the final go/no-go gate before model smoke and repeated-seed training, and the staging completion gate is the handoff between asset staging and model smoke; retain the environment, compute, environment transition, launch preflight, staging completion, and runbook reports with the experiment packet.
The source manifest records the intended reference genome, model, downstream benchmark, and variant benchmark sources before results are generated. The source citation table converts that manifest into the citation/license checklist used for Methods, Data Availability, Model Availability, and supplement source documentation. The asset staging runbook turns those entries into reviewable local PowerShell commands. The asset resource budget estimates required local disk space from manifest `expected_size_gb` values or conservative role defaults. The model cache audit checks for local weights, tokenizer files, config metadata, and a revision record before offline DNABERT-2 model smoke. The asset bootstrap script is safe by default and prints network/download steps unless called with `-ExecuteNetwork`; use it as the operator-facing entrypoint after reviewing disk space and source versions. Both staging PowerShell scripts write JSONL execution logs under `outputs/` so dry-run, skipped, passing, and failing steps can be retained with the packet after real acquisition. The acquisition plan turns the manifest into a staged checklist, and the acquisition ledger records the observed local/raw paths, sizes, sha256 hashes, and complete/incomplete state after every staging event. The asset lock records the local files or model cache directories with sizes, sha256 values, and missing-path markers; the acquisition ledger and asset lock validations must pass before any biological claim is treated as paper-ready. These files should be updated with exact versions, licenses, URLs, and local paths as assets are downloaded.

After the environment report passes, run a one-batch model integration smoke test:

```bash
jepa-glm-check-training-sanity --config configs/pretrain_dnabert2_hf.yaml --output-json outputs/training_sanity.json
jepa-glm-model-smoke --config configs/pretrain_dnabert2_8gb_smoke.yaml --device cuda --max-length 64 --backward --output-json outputs/dnabert2_model_smoke.json
```

The training sanity report checks effective batch size, mask settings, validation fraction, mixed precision, gradient checkpointing, and freeze policy. The model smoke report verifies tokenizer loading, backbone loading, JEPA-GLM forward shapes, finite outputs, trainable parameter counts, and the actual device used before long-running training is launched.

Create chromosome-disjoint GRCh38 windows:

```bash
python scripts/prepare_grch38.py --config configs/pretrain_dnabert2_hf.yaml --fasta data/GRCh38.fa --output outputs/grch38_windows.csv
jepa-glm-validate-manifest --manifest-csv outputs/grch38_windows.csv --max-non-acgt-fraction 0.5 --output-json outputs/grch38_windows.validation.json
jepa-glm-materialize-config --base-config configs/pretrain_dnabert2_hf.yaml --fasta data/GRCh38.fa --manifest-csv outputs/grch38_windows.csv --split train --local-model-dir models/dnabert2-117m --output-config outputs/configs/dnabert2_grch38_train.yaml
jepa-glm-make-experiment-plan --base-config outputs/configs/dnabert2_grch38_train.yaml --condition-config-dir outputs/configs/main_conditions --seeds 7,11,13 --output-root outputs/main_experiment --output-json outputs/experiment_plan.json --output-md outputs/experiment_plan.md
jepa-glm-validate-experiment-plan --plan-json outputs/experiment_plan.json --output-json outputs/experiment_plan.validation.json
```

Required artifacts:

- `outputs/grch38_windows.csv`
- `outputs/grch38_windows.summary.json`
- `outputs/grch38_windows.validation.json`
- `outputs/environment_check.json`
- `outputs/environment_runbook.md`
- `outputs/asset_lock.json`
- `outputs/asset_lock.validation.json`
- `outputs/acquisition_ledger.json`
- `outputs/acquisition_ledger.md`
- `outputs/acquisition_ledger.validation.json`
- `outputs/asset_staging_runbook.md`
- `outputs/asset_staging_runbook.ps1`
- `outputs/training_sanity.json`
- `outputs/dnabert2_model_smoke.json`
- materialized config file with `data.fasta_path`, `data.manifest_csv`, and `data.split`
- `outputs/experiment_plan.json` and `outputs/experiment_plan.md`
- `outputs/experiment_plan.validation.json`

## 2. Continual Pretraining

Run pretrained GLM baseline, MLM, JEPA, and MLM+JEPA conditions with identical data splits and seeds:

```bash
jepa-glm-smoke-comparison --config outputs/configs/dnabert2_grch38_train.yaml --seeds 7,11,13 --output-dir outputs/main_comparison
```

Required artifacts:

- seed-level JSON under each condition directory
- `comparison.csv` with mean, standard deviation, and 95% CI
- per-run `manifest.json`, `metrics.csv`, and `summary.json`

Validate ablation coverage before treating sensitivity claims as reportable. Paper ablations must cover mask ratio, predictor depth, pooling, and JEPA loss weight with at least three seeds.

```bash
jepa-glm-validate-ablation-readiness --grid-config configs/ablations_smoke.yaml --summary-csv outputs/ablations/ablation_summary.csv --run-dir outputs/ablations --output-json outputs/ablation_readiness.json
```

## 3. Downstream Evaluation

Prepare fixed splits:

```bash
jepa-glm-convert-bed-pair --positive-bed data/promoter_positive.bed --negative-bed data/promoter_negative.bed --fasta data/GRCh38.fa --window-size 512 --output-csv data/promoter_raw.csv --summary-json outputs/promoter_benchmark_conversion.summary.json
jepa-glm-audit-sequence-source --input-csv data/promoter_raw.csv --output-json outputs/promoter_source_audit.json
jepa-glm-prepare-benchmark --input-csv data/promoter_raw.csv --output-csv outputs/promoter_split.csv --manifest-json outputs/promoter_split_manifest.json
jepa-glm-validate-benchmark --input-csv outputs/promoter_split.csv --output-json outputs/promoter_split.validation.json
jepa-glm-validate-benchmark-manifest --manifest-json outputs/promoter_split_manifest.json --output-json outputs/promoter_split_manifest.validation.json
```

The BED-pair conversion fails by default on positive/negative coordinate overlaps and records cross-label overlap and duplicate-sequence counts in the summary JSON. Do not use `--allow-overlap` for paper benchmark generation.
The converted benchmark CSV retains `chrom,start,end`; split preparation uses chromosome-disjoint train/validation/test assignment when these columns are present. Retain the split manifest because it records the split strategy, source hash, split-level chromosome lists, label counts, and dropped rows.

Run linear probing:

```bash
jepa-glm-linear-probe --config configs/downstream_promoter.yaml --input-csv outputs/promoter_split.csv --bootstrap-samples 1000 --output-json outputs/promoter_probe.json
jepa-glm-validate-downstream-task-registry --registry-yaml configs/downstream_tasks.yaml --min-tasks 2 --output-json outputs/downstream_task_registry.validation.json
jepa-glm-validate-downstream-comparison --comparison-csv outputs/downstream_comparison.csv --required-tasks promoter,splice --required-conditions pretrained_glm,mlm,jepa,mlm_jepa --min-seeds 3 --output-json outputs/downstream_comparison.validation.json
jepa-glm-make-shuffled-label-control --input-csv outputs/promoter_split.csv --output-csv outputs/promoter_split_shuffled_labels.csv --manifest-json outputs/promoter_split_shuffled_labels_manifest.json
jepa-glm-linear-probe --config configs/downstream_promoter.yaml --input-csv outputs/promoter_split_shuffled_labels.csv --bootstrap-samples 1000 --output-json outputs/promoter_probe_shuffled_labels.json
jepa-glm-validate-negative-control --metric-json outputs/promoter_probe.json --control-json outputs/promoter_probe_shuffled_labels.json --output-json outputs/negative_control.validation.json
```

Required metrics:

- `outputs/promoter_split.validation.json` passing split, label coverage, and leakage checks
- `outputs/promoter_source_audit.json` passing raw sequence, label, duplicate, and alphabet checks
- `outputs/promoter_split_shuffled_labels_manifest.json` recording the shuffled-label control seed and source hash
- `outputs/negative_control.validation.json` passing shuffled-label or random-control sanity checks
- AUROC
- AUPRC
- MCC
- F1
- Accuracy

## 4. Zero-Shot Variant Evaluation

Create reference/alternate sequence pairs:

```bash
jepa-glm-convert-clinvar-vcf --input-vcf data/clinvar.vcf.gz --output-csv data/clinvar.csv --summary-json outputs/clinvar_conversion.summary.json
jepa-glm-audit-variant-source --input-csv data/clinvar.csv --output-json outputs/variant_source_audit.json
jepa-glm-convert-variants --input-csv data/clinvar.csv --fasta data/GRCh38.fa --flank 128 --output-csv outputs/clinvar_sequences.csv --summary-json outputs/clinvar_sequences.summary.json
jepa-glm-make-variant-allele-swap-control --input-csv outputs/clinvar_sequences.csv --output-csv outputs/clinvar_sequences_allele_swap.csv --manifest-json outputs/clinvar_sequences_allele_swap_manifest.json
jepa-glm-validate-variant-allele-swap-control --source-csv outputs/clinvar_sequences.csv --swapped-csv outputs/clinvar_sequences_allele_swap.csv --manifest-json outputs/clinvar_sequences_allele_swap_manifest.json --output-json outputs/clinvar_sequences_allele_swap.validation.json
```

Evaluate scores:

```bash
jepa-glm-evaluate-variants --checkpoint outputs/best_checkpoint.pt --input-csv outputs/clinvar_sequences.csv --bootstrap-samples 1000 --scores-csv outputs/variant_scores.csv --metrics-json outputs/variant_metrics.json
jepa-glm-evaluate-variants --checkpoint outputs/best_checkpoint.pt --input-csv outputs/clinvar_sequences_allele_swap.csv --bootstrap-samples 1000 --scores-csv outputs/variant_scores_allele_swap.csv --metrics-json outputs/variant_metrics_allele_swap.json
jepa-glm-validate-variant-score-control --original-scores-csv outputs/variant_scores.csv --swapped-scores-csv outputs/variant_scores_allele_swap.csv --output-json outputs/variant_score_control.validation.json
jepa-glm-validate-variant-comparison --comparison-csv outputs/variant_comparison.csv --required-scores l2,cosine --required-conditions pretrained_glm,mlm,jepa,mlm_jepa --min-seeds 3 --output-json outputs/variant_comparison.validation.json
```

## 5. Quality Gates

Every reportable training run must pass:

```bash
jepa-glm-check-quality --run-dir outputs/main_run --output-json outputs/quality_gates.json
```

Minimum gates:

- finite final loss
- finite validation loss
- embedding standard deviation above collapse threshold
- mask fraction inside expected bounds
- finite step-wise losses

## 6. Paper Packet

Generate figures, audit, paper report, and packet:

```bash
jepa-glm-make-methods --manifest-json outputs/main_run/manifest.json --summary-json outputs/main_run/summary.json --config configs/pretrain_dnabert2_hf.yaml --output-md outputs/methods_draft.md --allow-missing
jepa-glm-make-figures --metrics-csv outputs/main_run/metrics.csv --comparison-csv outputs/main_comparison/comparison.csv --quality-json outputs/quality_gates.json --output-dir outputs/figures
jepa-glm-validate-figure-readiness --figure-manifest-json outputs/figures/figure_manifest.json --paper-report-md outputs/paper_report.md --output-json outputs/figure_readiness.json
jepa-glm-audit-experiment --run-dir outputs/main_run --comparison-csv outputs/main_comparison/comparison.csv --linear-probe-json outputs/promoter_probe.json --variant-json outputs/variant_metrics.json --quality-json outputs/quality_gates.json --manifest-validation-json outputs/grch38_windows.validation.json --benchmark-validation-json outputs/promoter_split.validation.json --benchmark-manifest-validation-json outputs/promoter_split_manifest.validation.json --output-json outputs/audit.json
jepa-glm-validate-evaluation-readiness --benchmark-validation-json outputs/promoter_split.validation.json --linear-probe-json outputs/promoter_probe.json --downstream-metric-json splice=outputs/splice_probe.json --min-downstream-tasks 2 --variant-metrics-json outputs/variant_metrics.json --variant-scores-csv outputs/variant_scores.csv --output-json outputs/evaluation_readiness.json
jepa-glm-validate-effects --comparison-csv outputs/main_comparison/comparison.csv --baseline mlm --candidate jepa --candidate mlm_jepa --output-json outputs/effect_interpretation.json
jepa-glm-validate-statistical-evidence --comparison-csv outputs/main_comparison/comparison.csv --baseline mlm --candidate jepa --candidate mlm_jepa --output-json outputs/statistical_evidence.json
jepa-glm-validate-negative-control --metric-json outputs/promoter_probe.json --control-json outputs/promoter_probe_shuffled_labels.json --output-json outputs/negative_control.validation.json
jepa-glm-audit-training-log --metrics-csv outputs/main_run/metrics.csv --min-rows 10 --output-json outputs/training_log_audit.json --output-md outputs/training_log_audit.md
jepa-glm-validate-claims --run-manifest outputs/main_run/manifest.json --audit-json outputs/audit.json --comparison-csv outputs/main_comparison/comparison.csv --linear-probe-json outputs/promoter_probe.json --variant-json outputs/variant_metrics.json --source-validation-json outputs/source_manifest.validation.json --output-json outputs/claim_validation.json
jepa-glm-make-report --paper-report --audit-json outputs/audit.json --quality-json outputs/quality_gates.json --comparison-csv outputs/main_comparison/comparison.csv --linear-probe-json outputs/promoter_probe.json --variant-json outputs/variant_metrics.json --output-md outputs/paper_report.md
jepa-glm-make-packet --paper-defaults --output-root outputs --output-dir outputs/repro_packet --zip-path outputs/repro_packet.zip
jepa-glm-verify-packet --packet-dir outputs/repro_packet --zip-path outputs/repro_packet.zip --output-json outputs/repro_packet.validation.json
jepa-glm-validate-submission-packet --packet-dir outputs/repro_packet --output-json outputs/submission_packet_readiness.json
```

The Methods draft is generated directly from the run manifest and summary so the model architecture, data source, masking policy, optimization settings, and quality signals stay synchronized with the reported experiment. Synthetic runs must remain labeled as software validation rather than biological evidence.
The statistical evidence gate checks minimum seed count, relative effect size, and CI width before a comparison is treated as interpretable.

## 7. Reporting Rule

Synthetic smoke results may validate software behavior but must not be reported as biological evidence. Paper claims require real reference-genome pretraining data, fixed benchmark splits, and held-out downstream or variant labels.

Before drafting claims, generate the current readiness gap report:

```bash
jepa-glm-make-readiness-refresh --output-ps1 outputs/readiness_refresh.ps1 --output-md outputs/readiness_refresh.md
jepa-glm-check-readiness --output-json outputs/readiness_check.json
jepa-glm-make-readiness-report --readiness-json outputs/readiness_check.json --output-md outputs/readiness_gap_report.md
jepa-glm-make-readiness-action-plan --readiness-json outputs/readiness_check.json --output-md outputs/readiness_action_plan.md
jepa-glm-make-readiness-dashboard --readiness-json outputs/readiness_check.json --output-json outputs/readiness_dashboard.json --output-md outputs/readiness_dashboard.md
jepa-glm-make-claim-matrix --output-json outputs/claim_evidence_matrix.json --output-md outputs/claim_evidence_matrix.md
jepa-glm-make-claim-wording-guard --matrix-json outputs/claim_evidence_matrix.json --output-json outputs/claim_wording_guard.json --output-md outputs/claim_wording_guard.md
jepa-glm-make-paper-status-draft --claim-wording-json outputs/claim_wording_guard.json --readiness-json outputs/readiness_check.json --source-citation-json outputs/source_citation_table.json --output-md outputs/paper_status_draft.md
```

Use `outputs/readiness_refresh.ps1` as the fixed-order local refresh script for paper-readiness evidence. It is intentionally non-destructive and records expected failures while CUDA, GRCh38, model weights, downstream labels, and completed main runs are still absent. Before drafting manuscript claims, inspect `outputs/claim_wording_guard.md` and `outputs/paper_status_draft.md`; any claim with unresolved evidence must be written as an implemented evaluation target rather than as a demonstrated biological result.
Use `outputs/claim_evidence_matrix.md` before drafting the Results and Discussion sections; claims whose evidence rows are missing or failed should remain out of the manuscript.
