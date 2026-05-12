# V8 Diagnostic Commands

This file freezes the next phase after V9-V16 regressions. The goal is no longer to add new architecture branches. The goal is to confirm and understand why V8 works.

Use the same environment variables before each run:

```powershell
cd D:\Users\joeis\Desktop\paper\paper\EMNLP2026\EmoDynamiX-v2

$env:HTTP_PROXY="http://127.0.0.1:7897"
$env:HTTPS_PROXY="http://127.0.0.1:7897"
$env:HF_HOME="D:\env\hf-cache"
$env:TRANSFORMERS_CACHE="D:\env\hf-cache\transformers"
```

## Run 1: V8 Confirm 10000

Purpose:

- Confirm whether V8 improves after `7500`.
- V15 showed best checkpoint can move to `7500`, so V8 should be checked with `10000` max steps.
- This is the first run to execute.

```powershell
& D:\env\Anaconda\envs\emodynamix\python.exe .\main.py `
  --model roberta-hg `
  --dataset esconv-preprocessed `
  --batch_size 4 `
  --total_steps 10000 `
  --total_epochs 8 `
  --save_steps 500 `
  --eval_steps 500 `
  --seed 114514 `
  --experiment_name v8-confirm-10000 `
  --theory_variant v8-label-aware-decoder-confirm-10000 `
  --use_node_position_embedding 1 `
  --graph_readout dummy_mean_max `
  --use_feature_layernorm 1 `
  --use_multiview_logits 1 `
  --fusion_strategy dynamic_gate `
  --fusion_gate_hidden_dim 128 `
  --fusion_view_dropout 0.05 `
  --context_aux_loss_weight 0.12 `
  --graph_aux_loss_weight 0.05 `
  --use_label_aware_decoder 1 `
  --label_decoder_dim 256 `
  --label_decoder_heads 4 `
  --label_decoder_dropout 0.1 `
  --label_decoder_logit_weight 0.25 `
  --label_decoder_aux_loss_weight 0.0 `
  --use_uncertainty_reweighting 0 `
  --use_boundary_soft_labels 0 `
  --use_dialogue_state_adapter 0 `
  --use_global_token_label_evidence 0 `
  --use_turn_label_evidence 0 `
  --use_selective_turn_evidence_gate 0 `
  --use_hierarchical_decoder 0 `
  --use_rare_granule_expert 0 `
  --use_turn_evidence_encoder 0 `
  --use_text_aware_graph_nodes 0 `
  --use_prototype_head 0 `
  --use_granular_uncertainty 1 `
  --granule_loss_weight 0.03 `
  --granule_logit_weight 0.0
```

## Run 2: V8 Without Granule Aux

Purpose:

- Test whether `granule_loss_weight=0.03` actually helps V8 classification.
- If this run is close or better, granule aux should be framed mainly as analysis rather than necessary for classification.

```powershell
& D:\env\Anaconda\envs\emodynamix\python.exe .\main.py `
  --model roberta-hg `
  --dataset esconv-preprocessed `
  --batch_size 4 `
  --total_steps 10000 `
  --total_epochs 8 `
  --save_steps 500 `
  --eval_steps 500 `
  --seed 114514 `
  --experiment_name v8-no-granule-aux-10000 `
  --theory_variant v8-ablation-no-granule-aux `
  --use_node_position_embedding 1 `
  --graph_readout dummy_mean_max `
  --use_feature_layernorm 1 `
  --use_multiview_logits 1 `
  --fusion_strategy dynamic_gate `
  --fusion_gate_hidden_dim 128 `
  --fusion_view_dropout 0.05 `
  --context_aux_loss_weight 0.12 `
  --graph_aux_loss_weight 0.05 `
  --use_label_aware_decoder 1 `
  --label_decoder_dim 256 `
  --label_decoder_heads 4 `
  --label_decoder_dropout 0.1 `
  --label_decoder_logit_weight 0.25 `
  --label_decoder_aux_loss_weight 0.0 `
  --use_uncertainty_reweighting 0 `
  --use_boundary_soft_labels 0 `
  --use_dialogue_state_adapter 0 `
  --use_global_token_label_evidence 0 `
  --use_turn_label_evidence 0 `
  --use_selective_turn_evidence_gate 0 `
  --use_hierarchical_decoder 0 `
  --use_rare_granule_expert 0 `
  --use_turn_evidence_encoder 0 `
  --use_text_aware_graph_nodes 0 `
  --use_prototype_head 0 `
  --use_granular_uncertainty 0 `
  --granule_loss_weight 0.0 `
  --granule_logit_weight 0.0
```

## Run 3: V8 Without Context/Graph Aux

Purpose:

- Test whether auxiliary context/graph losses are helping or pulling the model away from the fused decision.

```powershell
& D:\env\Anaconda\envs\emodynamix\python.exe .\main.py `
  --model roberta-hg `
  --dataset esconv-preprocessed `
  --batch_size 4 `
  --total_steps 10000 `
  --total_epochs 8 `
  --save_steps 500 `
  --eval_steps 500 `
  --seed 114514 `
  --experiment_name v8-no-view-aux-10000 `
  --theory_variant v8-ablation-no-context-graph-aux `
  --use_node_position_embedding 1 `
  --graph_readout dummy_mean_max `
  --use_feature_layernorm 1 `
  --use_multiview_logits 1 `
  --fusion_strategy dynamic_gate `
  --fusion_gate_hidden_dim 128 `
  --fusion_view_dropout 0.05 `
  --context_aux_loss_weight 0.0 `
  --graph_aux_loss_weight 0.0 `
  --use_label_aware_decoder 1 `
  --label_decoder_dim 256 `
  --label_decoder_heads 4 `
  --label_decoder_dropout 0.1 `
  --label_decoder_logit_weight 0.25 `
  --label_decoder_aux_loss_weight 0.0 `
  --use_uncertainty_reweighting 0 `
  --use_boundary_soft_labels 0 `
  --use_dialogue_state_adapter 0 `
  --use_global_token_label_evidence 0 `
  --use_turn_label_evidence 0 `
  --use_selective_turn_evidence_gate 0 `
  --use_hierarchical_decoder 0 `
  --use_rare_granule_expert 0 `
  --use_turn_evidence_encoder 0 `
  --use_text_aware_graph_nodes 0 `
  --use_prototype_head 0 `
  --use_granular_uncertainty 1 `
  --granule_loss_weight 0.03 `
  --granule_logit_weight 0.0
```

## Run 4: V8 Lower Label Decoder Weight

Purpose:

- Test whether V8's label decoder residual is too strong.
- This may protect weighted F1 if decoder helps macro but disturbs high-support classes.

```powershell
& D:\env\Anaconda\envs\emodynamix\python.exe .\main.py `
  --model roberta-hg `
  --dataset esconv-preprocessed `
  --batch_size 4 `
  --total_steps 10000 `
  --total_epochs 8 `
  --save_steps 500 `
  --eval_steps 500 `
  --seed 114514 `
  --experiment_name v8-label-decoder-w015-10000 `
  --theory_variant v8-label-decoder-weight-015 `
  --use_node_position_embedding 1 `
  --graph_readout dummy_mean_max `
  --use_feature_layernorm 1 `
  --use_multiview_logits 1 `
  --fusion_strategy dynamic_gate `
  --fusion_gate_hidden_dim 128 `
  --fusion_view_dropout 0.05 `
  --context_aux_loss_weight 0.12 `
  --graph_aux_loss_weight 0.05 `
  --use_label_aware_decoder 1 `
  --label_decoder_dim 256 `
  --label_decoder_heads 4 `
  --label_decoder_dropout 0.1 `
  --label_decoder_logit_weight 0.15 `
  --label_decoder_aux_loss_weight 0.0 `
  --use_uncertainty_reweighting 0 `
  --use_boundary_soft_labels 0 `
  --use_dialogue_state_adapter 0 `
  --use_global_token_label_evidence 0 `
  --use_turn_label_evidence 0 `
  --use_selective_turn_evidence_gate 0 `
  --use_hierarchical_decoder 0 `
  --use_rare_granule_expert 0 `
  --use_turn_evidence_encoder 0 `
  --use_text_aware_graph_nodes 0 `
  --use_prototype_head 0 `
  --use_granular_uncertainty 1 `
  --granule_loss_weight 0.03 `
  --granule_logit_weight 0.0
```

## Run 5: V8 Higher Label Decoder Weight

Purpose:

- Test whether more label-aware residual can recover macro F1 beyond V8.
- Stop this direction if weighted F1 drops clearly.

```powershell
& D:\env\Anaconda\envs\emodynamix\python.exe .\main.py `
  --model roberta-hg `
  --dataset esconv-preprocessed `
  --batch_size 4 `
  --total_steps 10000 `
  --total_epochs 8 `
  --save_steps 500 `
  --eval_steps 500 `
  --seed 114514 `
  --experiment_name v8-label-decoder-w035-10000 `
  --theory_variant v8-label-decoder-weight-035 `
  --use_node_position_embedding 1 `
  --graph_readout dummy_mean_max `
  --use_feature_layernorm 1 `
  --use_multiview_logits 1 `
  --fusion_strategy dynamic_gate `
  --fusion_gate_hidden_dim 128 `
  --fusion_view_dropout 0.05 `
  --context_aux_loss_weight 0.12 `
  --graph_aux_loss_weight 0.05 `
  --use_label_aware_decoder 1 `
  --label_decoder_dim 256 `
  --label_decoder_heads 4 `
  --label_decoder_dropout 0.1 `
  --label_decoder_logit_weight 0.35 `
  --label_decoder_aux_loss_weight 0.0 `
  --use_uncertainty_reweighting 0 `
  --use_boundary_soft_labels 0 `
  --use_dialogue_state_adapter 0 `
  --use_global_token_label_evidence 0 `
  --use_turn_label_evidence 0 `
  --use_selective_turn_evidence_gate 0 `
  --use_hierarchical_decoder 0 `
  --use_rare_granule_expert 0 `
  --use_turn_evidence_encoder 0 `
  --use_text_aware_graph_nodes 0 `
  --use_prototype_head 0 `
  --use_granular_uncertainty 1 `
  --granule_loss_weight 0.03 `
  --granule_logit_weight 0.0
```

