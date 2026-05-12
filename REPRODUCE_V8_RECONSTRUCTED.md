# Reproducing V8 Reconstructed

This repository contains the reconstructed V8 method used for the current strongest ESConv-preprocessed result.

## Environment

- Windows
- Python environment: `D:\env\Anaconda\envs\emodynamix`
- Recommended batch size on an 8GB GPU: `4`

Set cache/proxy variables if needed:

```powershell
$env:HTTP_PROXY="http://127.0.0.1:7897"
$env:HTTPS_PROXY="http://127.0.0.1:7897"
$env:HF_HOME="D:\env\hf-cache"
$env:TRANSFORMERS_CACHE="D:\env\hf-cache\transformers"
```

## Training Command

```powershell
cd D:\Users\joeis\Desktop\paper\paper\EMNLP2026\EmoDynamiX-v2

& D:\env\Anaconda\envs\emodynamix\python.exe .\main.py `
  --model roberta-hg `
  --dataset esconv-preprocessed `
  --seed 114514 `
  --experiment_preset v8_reconstructed
```

The preset expands to the V8 reconstructed structure:

- `fusion_strategy=dynamic_gate`
- `use_label_aware_decoder=1`
- `use_granular_uncertainty=1`
- `granule_loss_weight=0.03`
- `granule_logit_weight=0.0`
- `use_node_position_embedding=1`
- `graph_readout=dummy_mean_max`
- `use_feature_layernorm=1`
- `use_multiview_logits=1`

V9+ exploratory modules are instantiated only when explicitly enabled, so the default V8 reconstructed path is isolated from later experimental modules.

## Current Strongest Result

Run:

`roberta-hg-esconv-preprocessed-v8-reconstructed-label-aware-decoder-seed114514-20260512-191959`

Test checkpoint: `7500`

```text
accuracy: 0.3247
macro f1: 0.2675
weighted f1: 0.3191
preference bias: 0.2916
```

Historical comparison:

- Original baseline: macro F1 `0.2477`, weighted F1 `0.3004`, accuracy `0.3154`
- Previous V8: macro F1 `0.2605`, weighted F1 `0.3112`, accuracy `0.3216`
- Reconstructed V8: macro F1 `0.2675`, weighted F1 `0.3191`, accuracy `0.3247`
