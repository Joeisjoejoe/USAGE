# Experiment Protocol

This project now writes structured run artifacts for each training run so the
results can be traced back to a specific configuration, checkpoint, and log.

## Output layout

Each run writes to:

- `./roberta-hg-<dataset>-logs/<run_id>/`
- `./roberta-hg-<dataset>-checkpoints/<run_id>/`

Run ids include the experiment label for easier manual inspection:

- `<model>-<dataset>-<experiment_name>-seed<seed>-<YYYYMMDD-HHMMSS>`
- Example:
  `roberta-hg-esconv-preprocessed-baseline-seed114514-20260510-165949`

Every run directory contains:

- `run_config.json`: runtime metadata, raw CLI arguments, package versions,
  dataset sizes, git commit, and file paths.
- `train_history.jsonl`: one JSON line per validation event.
- `summary.json`: final run summary with best checkpoint, best validation
  metrics, selected test checkpoint, and training resource stats.
- `result.json`: final test metrics.
- `analysis.json`: confidence stats, top confusions, error counts, and label
  supports.
- `cases.json`: test-time cases for qualitative analysis.
- `train.log`: console-style validation log.

## Recommended naming

Use the following CLI fields for paper-facing runs:

- `--experiment_name`: high-level run label such as
  `baseline`, `v2-position-readout-granule`, or `our-final-method-run`.
  This value is included in the run directory name, so keep it short and
  human-readable.
- `--theory_variant`: method family such as
  `baseline`, `uncertainty-gating`, or `three-way-decision`
- `--run_notes`: short free-form note for hardware or special settings

These values are recorded in `run_config.json`.

## Phase 1 baseline reproduction

For reproduction runs:

- keep one seed per setting
- use `esconv-preprocessed` first
- keep the command line fixed for the entire comparison slice
- compare the final `summary.json` and `result.json` against the paper

Key fields to report:

- `summary.best_checkpoint`
- `summary.best_valid_metrics`
- `summary.test_metrics`
- `summary.training_stats.total_training_time_sec`
- `summary.training_stats.peak_gpu_memory_mb`

## Phase 2 method comparison

For ablations and new methods, maintain the same:

- dataset split
- training budget
- evaluation cadence
- reporting metrics

Primary metrics:

- `macro f1`
- `weighted f1`
- `accuracy`
- `preference bias`

Secondary analysis should use:

- `analysis.top_confusions`
- `analysis.low_confidence_prediction_count`
- `analysis.supports_by_label`
- `cases.json`

## Paper table mapping

- Main results table:
  `result.json` plus `summary.json`
- Efficiency table:
  `summary.training_stats`
- Ablation table:
  compare multiple `summary.json` files
- Case study section:
  `cases.json`
- Error analysis section:
  `analysis.json`

## Phase 3 uncertainty-aware variant

The first uncertainty-aware implementation keeps the original EmoDynamiX
heterogeneous graph intact and adds class granules at the final dialogue-level
representation. The method is disabled by default, so baseline commands remain
comparable.

Recommended first run:

```powershell
& D:\env\Anaconda\envs\emodynamix\python.exe .\main.py `
  --model roberta-hg `
  --dataset esconv-preprocessed `
  --batch_size 4 `
  --total_steps 7500 `
  --total_epochs 8 `
  --save_steps 500 `
  --eval_steps 500 `
  --seed 114514 `
  --experiment_name uncertainty-aware-granules-v1 `
  --theory_variant granular-three-way-decision `
  --use_granular_uncertainty 1 `
  --granule_loss_weight 0.1 `
  --granule_logit_weight 0.2 `
  --granule_temperature 0.2 `
  --three_way_accept_threshold 0.45 `
  --three_way_reject_threshold 0.65 `
  --calibrate_three_way_thresholds 1 `
  --three_way_accept_quantile 0.1 `
  --three_way_reject_quantile 0.9
```

Additional fields written when the uncertainty branch is enabled:

- `result.json`: mean uncertainty, accept/boundary/reject rates, and accuracy
  inside each decision region.
- `analysis.json`: uncertainty summary, decision-region summary, boundary
  sample indices, and reject sample indices.
- `cases.json`: per-case uncertainty features and the assigned three-way
  decision.

## Phase 4 classification-oriented V2

V2 keeps the uncertainty branch available but prioritizes classification
metrics. The default command still behaves like the original baseline unless
the following switches are enabled.

Shared budget:

```powershell
& D:\env\Anaconda\envs\emodynamix\python.exe .\main.py `
  --model roberta-hg `
  --dataset esconv-preprocessed `
  --batch_size 4 `
  --total_steps 7500 `
  --total_epochs 8 `
  --save_steps 500 `
  --eval_steps 500 `
  --seed 114514
```

Run these V2 variants by appending the listed arguments:

- `v2-decoupled-granule`:
  `--experiment_name v2-decoupled-granule --theory_variant v2-classification --use_granular_uncertainty 1 --granule_loss_weight 0.03 --granule_logit_weight 0.0`
- `v2-position-readout`:
  `--experiment_name v2-position-readout --theory_variant v2-classification --use_node_position_embedding 1 --graph_readout dummy_mean_max --use_feature_layernorm 1`
- `v2-position-readout-granule`:
  `--experiment_name v2-position-readout-granule --theory_variant v2-classification --use_node_position_embedding 1 --graph_readout dummy_mean_max --use_feature_layernorm 1 --use_granular_uncertainty 1 --granule_loss_weight 0.03 --granule_logit_weight 0.0`
- `v2-position-readout-logit-adjusted`:
  `--experiment_name v2-position-readout-logit-adjusted --theory_variant v2-classification --use_node_position_embedding 1 --graph_readout dummy_mean_max --use_feature_layernorm 1 --loss_type logit_adjusted_ce --logit_adjust_tau 0.2 --label_smoothing 0.03`
- `v2-position-readout-granule-logit-adjusted`:
  `--experiment_name v2-position-readout-granule-logit-adjusted --theory_variant v2-classification --use_node_position_embedding 1 --graph_readout dummy_mean_max --use_feature_layernorm 1 --use_granular_uncertainty 1 --granule_loss_weight 0.03 --granule_logit_weight 0.0 --loss_type logit_adjusted_ce --logit_adjust_tau 0.2 --label_smoothing 0.03`
- `v2-position-readout-focal`:
  `--experiment_name v2-position-readout-focal --theory_variant v2-classification --use_node_position_embedding 1 --graph_readout dummy_mean_max --use_feature_layernorm 1 --loss_type focal --focal_gamma 1.5`
- `v2-position-readout-gat-ffn`:
  `--experiment_name v2-position-readout-gat-ffn --theory_variant v2-classification --use_node_position_embedding 1 --graph_readout dummy_mean_max --use_feature_layernorm 1 --use_gat_ffn 1`

Compare each run against the strongest baseline:

- `test macro f1 > 0.2477`
- `test weighted f1 >= 0.3004`, or within `0.003`
- `preference bias < 0.8089`

## Phase 5 multi-view classification V3

V3 adds explicit semantic, graph, and fused classification views. It can be
reported as multi-granular evidence fusion if it improves classification.

Recommended first run:

```powershell
& D:\env\Anaconda\envs\emodynamix\python.exe .\main.py `
  --model roberta-hg `
  --dataset esconv-preprocessed `
  --batch_size 4 `
  --total_steps 7500 `
  --total_epochs 8 `
  --save_steps 500 `
  --eval_steps 500 `
  --seed 114514 `
  --experiment_name v3-multiview-readout-granule `
  --theory_variant v3-multiview-classification `
  --use_node_position_embedding 1 `
  --graph_readout dummy_mean_max `
  --use_feature_layernorm 1 `
  --use_multiview_logits 1 `
  --context_logit_weight 0.3 `
  --graph_logit_weight 0.2 `
  --context_aux_loss_weight 0.2 `
  --graph_aux_loss_weight 0.1 `
  --use_granular_uncertainty 1 `
  --granule_loss_weight 0.03 `
  --granule_logit_weight 0.0
```

If this over-biases toward text-heavy classes, reduce
`--context_logit_weight` to `0.2`. If it hurts weighted F1, set
`--graph_logit_weight 0.1`.
