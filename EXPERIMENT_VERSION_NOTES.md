# EmoDynamiX-v2 Experiment Version Notes

This document records the main architectural variants, observed results, and practical conclusions for the current EmoDynamiX-v2 line. It is intended as a quick handoff note for future model-structure changes.

## Stable Environment

- Project path: `D:\Users\joeis\Desktop\paper\paper\EMNLP2026\EmoDynamiX-v2`
- Python: `D:\env\Anaconda\envs\emodynamix\python.exe`
- Recommended command pattern:

```powershell
& D:\env\Anaconda\envs\emodynamix\python.exe .\main.py ...
```

- Common environment variables:

```powershell
$env:HTTP_PROXY="http://127.0.0.1:7897"
$env:HTTPS_PROXY="http://127.0.0.1:7897"
$env:HF_HOME="D:\env\hf-cache"
$env:TRANSFORMERS_CACHE="D:\env\hf-cache\transformers"
```

- GPU memory: 8GB.
- `batch_size=16` causes OOM.
- `batch_size=4` is the current stable setting.
- Training is effectively controlled by `total_steps`, not only by `total_epochs`.

## Current Standard Setup

- Dataset: `esconv-preprocessed`
- Seed: `114514`
- Batch size: `4`
- Usual budget: `total_steps=7500`, `total_epochs=8`, `save_steps=500`, `eval_steps=500`
- Model entry: `main.py`
- Main model: `modules/roberta/model.py`, class `RobertaHeterogeneousGraph`
- Trainer: `modules/trainer/trainer.py`

## Original Baseline

Strongest original EmoDynamiX baseline:

- Run id: `roberta-hg-esconv-preprocessed-seed114514-20260510-165949`
- Best checkpoint: `6000`
- Accuracy: `0.3154`
- Macro F1: `0.2477`
- Weighted F1: `0.3004`
- Preference bias: `0.8089`

Conclusion:

- This is the baseline that later method variants should compare against.
- Longer training was not better. A run with `total_steps=25520` degraded.
- The useful checkpoint region is roughly `5000-7500`, with baseline best around `6000`.

## V1: Granular Uncertainty / Three-way Decision

Main idea:

- Add uncertainty-related outputs and class granules.
- Support three-way decision analysis: accept / boundary / reject.
- Add calibration support for three-way thresholds.

Main switches:

- `--use_granular_uncertainty`
- `--granule_loss_weight`
- `--granule_logit_weight`
- `--granule_temperature`
- `--calibrate_three_way_thresholds`

Observed behavior:

- Directly adding granule logits to main logits can improve some low-frequency classes, but often hurts dominant classes.
- Three-way calibration should affect eval artifacts only, not overwrite training results.

Conclusion:

- Keep uncertainty/granule branch mainly as auxiliary representation and analysis tool.
- Avoid strong direct logit injection from granules.

## V2: Position, Graph Readout, Loss Variants

Main idea:

- Improve graph representation with position and better readout.
- Add optional loss variants for class imbalance.

Main switches:

- `--use_node_position_embedding`
- `--graph_readout dummy_mean_max`
- `--use_gat_ffn`
- `--use_feature_layernorm`
- `--loss_type ce|focal|logit_adjusted_ce`
- `--label_smoothing`

Conclusion:

- Position embedding and `dummy_mean_max` readout are useful low-cost defaults.
- Loss-only changes are not enough for a large jump.
- `use_gat_ffn` remains optional; stop if runtime/memory cost is not worthwhile.

## V3: Multi-view Logits

Main idea:

- Split classification into context view, graph view, and fused view.
- Add auxiliary losses for context and graph logits.

Main switches:

- `--use_multiview_logits`
- `--context_logit_weight`
- `--graph_logit_weight`
- `--context_aux_loss_weight`
- `--graph_aux_loss_weight`

Best observed result:

- Checkpoint: `4500`
- Accuracy: `0.3047`
- Macro F1: `0.2497`
- Weighted F1: `0.3002`
- Preference bias: `0.5113`

Conclusion:

- Multi-view supervision is helpful for macro F1 and bias, but does not yet clearly beat baseline weighted F1.

## V4: Dynamic Gate Fusion

Main idea:

- Replace fixed multi-view fusion with a sample-level dynamic gate over context, graph, and fused logits.

Main switch:

- `--fusion_strategy dynamic_gate`

Best observed result:

- Checkpoint: `3500`
- Accuracy: `0.2967`
- Macro F1: `0.2596`
- Weighted F1: `0.3066`
- Preference bias: `0.6056`

Conclusion:

- This was the first strong method variant.
- Dynamic gating is a useful mainline component.
- V4 beats the original baseline on macro F1 and weighted F1, but accuracy is lower.

## V5: Additional Fusion Variants

Main idea:

- Explore different multi-view fusion variants.

Conclusion:

- No stable improvement over V4 was established.
- Keep V4 `dynamic_gate` as the main fusion strategy.

## V6: Classwise Dynamic Gate

Main idea:

- Increase gate flexibility by learning class-specific fusion weights.

Best observed result:

- Checkpoint: `7000`
- Accuracy: `0.2988`
- Macro F1: `0.2431`
- Weighted F1: `0.2940`
- Preference bias: `0.3793`

Conclusion:

- Classwise gate reduces preference bias but hurts classification.
- Treat as a low-bias analysis variant, not the main performance path.

## V7: Evidence, Text-aware Graph, Prototype Head

Main idea:

- Add local turn evidence encoder.
- Add text-aware graph nodes.
- Add prototype boundary head.

Main switches:

- `--use_turn_evidence_encoder`
- `--evidence_encoder_mode turn_attention|window_cls`
- `--use_text_aware_graph_nodes`
- `--use_prototype_head`

Full V7 result:

- Checkpoint: `6000`
- Accuracy: `0.3216`
- Macro F1: `0.2506`
- Weighted F1: `0.3070`
- Preference bias: `0.6205`

V7-lite result:

- Checkpoint: `4500`
- Accuracy: `0.3095`
- Macro F1: `0.2498`
- Weighted F1: `0.3019`
- Preference bias: `0.5715`

Conclusion:

- Full V7 is too slow because it adds extra RoBERTa encoding over turns.
- It improves accuracy/weighted F1 somewhat, but macro F1 does not beat V4/V8.
- Prototype head did not provide clear macro benefit.
- Do not use V7 as current mainline.

## V8: Label-aware Evidence Decoder

Main idea:

- Keep V4 dynamic gate.
- Add learnable label queries.
- Each strategy label query cross-attends to context / graph / fused view tokens.

Main switches:

- `--use_label_aware_decoder 1`
- `--label_decoder_dim 256`
- `--label_decoder_heads 4`
- `--label_decoder_dropout 0.1`
- `--label_decoder_logit_weight 0.25`
- `--label_decoder_aux_loss_weight 0.0`

Best observed result:

- Checkpoint: `6000`
- Accuracy: `0.3216`
- Macro F1: `0.2605`
- Weighted F1: `0.3112`
- Preference bias: `0.5400`

Per-class F1:

- Reflection of feelings: `0.1038`
- Self-disclosure: `0.1185`
- Question: `0.5066`
- Affirmation and Reassurance: `0.1237`
- Providing Suggestions: `0.3883`
- Restatement or Paraphrasing: `0.2974`
- Information: `0.1138`
- Others: `0.4315`

Conclusion:

- Current strongest confirmed method.
- The effective part is label-aware evidence extraction, not hand-crafted logit priors.
- Keep V8 as the main baseline for future variants.

## V8.1: Label Decoder Auxiliary Loss

Main idea:

- Add direct auxiliary CE supervision to label decoder logits.

Observed result:

- Checkpoint: `7500`
- Accuracy: `0.2991`
- Macro F1: `0.2335`
- Weighted F1: `0.2886`
- Preference bias: `0.3063`

Conclusion:

- Auxiliary supervision was too disruptive.
- Keep `--label_decoder_aux_loss_weight 0.0` unless there is a very specific reason to revisit.

## V9: Hierarchical Label-aware Strategy Decoder

Main idea:

- Add coarse strategy groups and inject group-level priors into fine logits.

Groups:

- `emotional_validation`: Reflection of feelings, Affirmation and Reassurance
- `self_reframing`: Self-disclosure, Restatement or Paraphrasing
- `information_exchange`: Question, Information
- `action_other`: Providing Suggestions, Others

Observed result:

- Checkpoint: `3500`
- Accuracy: `0.2929`
- Macro F1: `0.2435`
- Weighted F1: `0.2917`
- Preference bias: `0.5437`

Conclusion:

- Coarse group prior harms fine-grained decision boundaries.
- It may help a few rare classes, but damages dominant/mid-frequency classes.
- Do not use hierarchy prior as the current mainline.

## V10: Boundary-aware Rare Granule Expert

Main idea:

- Add rare-class expert residual only for boundary samples.
- Intended to help rare/confusing classes without disturbing easy samples.

Main switches:

- `--use_rare_granule_expert`
- `--rare_expert_weight`
- `--rare_boundary_threshold`
- `--rare_boundary_temperature`
- `--rare_expert_classes`

Observed result:

- Checkpoint: `5500`
- Accuracy: `0.2898`
- Macro F1: `0.2397`
- Weighted F1: `0.2891`
- Preference bias: `0.5516`

Conclusion:

- Significant regression.
- The expert residual disturbed the main decision boundary.
- V9 and V10 together suggest that final-logit priors/residuals are risky.
- Do not continue the "add residual to final logits" direction without strong evidence.

## V11: Turn-level Label-aware Evidence Decoder

Main idea:

- Keep V8 as the base.
- Improve decision-before features instead of modifying final logits.
- Add recent turn-level evidence tokens into the label-aware decoder.
- Label queries now attend to:
  - global context token
  - graph readout token
  - fused token
  - recent K turn evidence tokens

Main switches:

- `--use_turn_label_evidence 1`
- `--turn_label_window_size 6`
- `--turn_label_dropout 0.1`

Implementation status:

- Code implemented.
- `py_compile` passed.
- `main.py --help` shows new switches.
- Small forward smoke test passed with:
  - `logits (2, 8)`
  - `label_decoder_logits (2, 8)`
  - `turn_label_evidence_valid_count [1, 4]`
  - `fusion_gate_weights (2, 3)`
  - `granule_logits (2, 8)`

Observed result:

- Checkpoint: `7500`
- Accuracy: `0.3036`
- Macro F1: `0.2589`
- Weighted F1: `0.3039`
- Preference bias: `0.3131`

Per-class F1:

- Reflection of feelings: `0.1151`
- Self-disclosure: `0.1064`
- Question: `0.4959`
- Affirmation and Reassurance: `0.1994`
- Providing Suggestions: `0.3098`
- Restatement or Paraphrasing: `0.2558`
- Information: `0.1875`
- Others: `0.4016`

Comparison with V8:

- Macro F1 slightly lower: `0.2589` vs `0.2605`
- Weighted F1 lower: `0.3039` vs `0.3112`
- Accuracy lower: `0.3036` vs `0.3216`
- Preference bias much lower: `0.3131` vs `0.5400`
- Reflection improved: `0.1151` vs `0.1038`
- Affirmation improved: `0.1994` vs `0.1237`
- Information improved: `0.1875` vs `0.1138`
- Question slightly lower: `0.4959` vs `0.5066`
- Providing Suggestions lower: `0.3098` vs `0.3883`
- Restatement lower: `0.2558` vs `0.2974`
- Others lower: `0.4016` vs `0.4315`

Expected behavior:

- Slower than V8 because it encodes recent turns separately.
- Should be much lighter than full V7 turn evidence.
- If runtime is too high, reduce `--turn_label_window_size` from `6` to `4`.

Conclusion:

- V11 validates the feature-extraction direction because rare/ambiguous classes improve substantially.
- However, unfiltered turn evidence harms high-support classes, especially Providing Suggestions, Restatement, and Others.
- The next step should not discard turn evidence. It should make turn evidence adaptive, class-specific, or gated so that it helps rare/boundary categories without weakening dominant classes.
- Good V12 direction: V8 + selective turn evidence gate, where label queries decide how much to use turn tokens instead of always injecting them.

Recommended first V11 command:

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
  --experiment_name v11-turn-label-evidence-decoder `
  --theory_variant v11-turn-level-label-aware-evidence `
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
  --use_turn_label_evidence 1 `
  --turn_label_window_size 6 `
  --turn_label_dropout 0.1 `
  --use_hierarchical_decoder 0 `
  --use_rare_granule_expert 0 `
  --use_turn_evidence_encoder 0 `
  --use_text_aware_graph_nodes 0 `
  --use_prototype_head 0 `
  --use_granular_uncertainty 1 `
  --granule_loss_weight 0.03 `
  --granule_logit_weight 0.0
```

## V12: Selective Turn Evidence Gate

Main idea:

- Keep V8/V11 label-aware decoder line.
- Avoid V11's issue where all labels must use turn evidence equally.
- Decode base evidence from global context / graph / fused tokens.
- Decode turn evidence from recent turn tokens separately.
- Learn a per-label evidence gate with shape `[batch_size, num_classes]`.
- Final decoded label feature is:
  - base label feature
  - plus gated turn-evidence residual

Main switches:

- `--use_turn_label_evidence 1`
- `--use_selective_turn_evidence_gate 1`
- `--turn_evidence_gate_hidden_dim 128`
- `--turn_evidence_gate_temperature 1.0`
- `--turn_evidence_residual_weight 0.5`

Implementation status:

- Code implemented.
- `py_compile` passed.
- `main.py --help` shows new switches.
- Small forward smoke test passed with:
  - `logits (2, 8)`
  - `label_decoder_logits (2, 8)`
  - `turn_label_evidence_valid_count [1, 4]`
  - `turn_evidence_gate (2, 8)`
  - gate mean around `0.4757` before training
  - `fusion_gate_weights (2, 3)`
  - `granule_logits (2, 8)`

Expected behavior:

- Should preserve V11's rare-class evidence gains better than V8.
- Should protect high-support classes better than V11 because turn evidence is no longer forced into every label.
- Compare primarily against V8 and V11.

Observed result:

- Checkpoint: `4500`
- Accuracy: `0.3133`
- Macro F1: `0.2361`
- Weighted F1: `0.2943`
- Preference bias: `0.7781`

Per-class F1:

- Reflection of feelings: `0.0387`
- Self-disclosure: `0.1099`
- Question: `0.4795`
- Affirmation and Reassurance: `0.2003`
- Providing Suggestions: `0.3748`
- Restatement or Paraphrasing: `0.2604`
- Information: `0.0364`
- Others: `0.3888`

Conclusion:

- V12 did not fix V11's issue and performed worse than V8 overall.
- Selective gating protected some high-support classes better than V11, such as Providing Suggestions, but it collapsed Reflection and Information.
- The turn-evidence path appears unstable because it adds separate RoBERTa encodings whose representation space may not align well with the global context encoding.
- The current evidence-extraction direction should not keep adding extra RoBERTa turn encoders.
- Future feature work should prefer reusing token-level or hidden-state information from the original global RoBERTa pass, or improving graph/context alignment, instead of adding separate turn text encodings.

Recommended first V12 command:

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
  --experiment_name v12-selective-turn-evidence-gate `
  --theory_variant v12-selective-label-turn-evidence-gate `
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
  --use_turn_label_evidence 1 `
  --turn_label_window_size 6 `
  --turn_label_dropout 0.1 `
  --use_selective_turn_evidence_gate 1 `
  --turn_evidence_gate_hidden_dim 128 `
  --turn_evidence_gate_temperature 1.0 `
  --turn_evidence_residual_weight 0.5 `
  --use_hierarchical_decoder 0 `
  --use_rare_granule_expert 0 `
  --use_turn_evidence_encoder 0 `
  --use_text_aware_graph_nodes 0 `
  --use_prototype_head 0 `
  --use_granular_uncertainty 1 `
  --granule_loss_weight 0.03 `
  --granule_logit_weight 0.0
```

## V13: Global Token-aware Label Evidence Decoder

Main idea:

- Return to V8's stable label-aware decoder.
- Avoid V11/V12's separate turn-level RoBERTa encodings.
- Reuse the same global RoBERTa forward pass that produces the dialogue CLS.
- Extract token-level hidden states from that same pass as global evidence tokens.
- Let label queries attend to:
  - global context CLS token
  - graph readout token
  - fused token
  - selected global token evidence bank

Why this version exists:

- V11/V12 showed that local evidence can help rare classes, but separate turn encoding destabilizes the representation space.
- V13 keeps evidence in the same global contextual space as CLS, so token evidence should be less disruptive.
- It should also be faster than V11/V12 because it does not run extra RoBERTa passes for turns.

Main switches:

- `--use_global_token_label_evidence 1`
- `--global_token_evidence_max_tokens 32`
- `--global_token_evidence_dropout 0.1`
- Keep `--use_turn_label_evidence 0`
- Keep `--use_selective_turn_evidence_gate 0`

Implementation status:

- Code implemented.
- `py_compile` passed.
- `main.py --help` shows new switches.
- Small forward smoke test passed with:
  - `logits (2, 8)`
  - `label_decoder_logits (2, 8)`
  - `global_token_label_evidence_valid_count [7, 32]`
  - `fusion_gate_weights (2, 3)`
  - `granule_logits (2, 8)`

Observed result:

- Checkpoint: `6500`
- Accuracy: `0.2877`
- Macro F1: `0.2390`
- Weighted F1: `0.2883`
- Preference bias: `0.6985`

Per-class F1:

- Reflection of feelings: `0.0726`
- Self-disclosure: `0.1357`
- Question: `0.4846`
- Affirmation and Reassurance: `0.1628`
- Providing Suggestions: `0.3289`
- Restatement or Paraphrasing: `0.2450`
- Information: `0.1078`
- Others: `0.3749`

Conclusion:

- V13 also underperforms V8.
- Adding more text-token evidence to the label-aware decoder is not the current bottleneck solution.
- V11, V12, and V13 together suggest that V8's decoder gain is real but fragile; giving it more evidence tokens tends to dilute or destabilize the class boundary.
- Future architecture work should stop expanding the decoder evidence bank and instead improve dialogue-state / graph-state representation or training supervision.

Recommended first V13 command:

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
  --experiment_name v13-global-token-label-evidence `
  --theory_variant v13-global-token-aware-label-evidence `
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
  --use_global_token_label_evidence 1 `
  --global_token_evidence_max_tokens 32 `
  --global_token_evidence_dropout 0.1 `
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

## V14: Dialogue State Adapter

Main idea:

- Stop expanding the label decoder evidence bank.
- Improve pre-decision dialogue-state representation instead.
- Add a lightweight context-graph adapter after graph readout and before classification.
- Project context and graph into a shared adapter space.
- Use their difference/product interaction to produce a gate.
- Add a gated graph-derived residual into the context embedding.
- Keep final dimensions unchanged so V8 classifier, dynamic gate, and label-aware decoder remain compatible.

Why this version exists:

- V11/V12/V13 suggest that adding more text evidence to the label decoder destabilizes class boundaries.
- The next likely bottleneck is context-graph state alignment, not evidence quantity.
- V14 tries to make graph information shape the dialogue state before decision.

Main switches:

- `--use_dialogue_state_adapter 1`
- `--dialogue_state_adapter_dim 256`
- `--dialogue_state_adapter_weight 0.3`
- `--dialogue_state_adapter_dropout 0.1`
- Keep V8 decoder on.
- Keep token/turn evidence off.

Implementation status:

- Code implemented.
- `py_compile` passed.
- `main.py --help` shows new switches.
- Small forward smoke test passed with:
  - `logits (2, 8)`
  - `label_decoder_logits (2, 8)`
  - `dialogue_state_gate_mean [0.4993, 0.4973]`
  - `fusion_gate_weights (2, 3)`
  - `granule_logits (2, 8)`

Observed result:

- Checkpoint: `2000`
- Accuracy: `0.2843`
- Macro F1: `0.2409`
- Weighted F1: `0.2828`
- Preference bias: `0.4956`

Per-class F1:

- Reflection of feelings: `0.0872`
- Self-disclosure: `0.0958`
- Question: `0.4798`
- Affirmation and Reassurance: `0.1473`
- Providing Suggestions: `0.2787`
- Restatement or Paraphrasing: `0.2751`
- Information: `0.1648`
- Others: `0.3985`

Conclusion:

- V14 underperforms V8 substantially.
- Context-graph adapter did not improve the main boundary and harmed high-support classes.
- V11-V14 together suggest that adding more modules on top of V8 is not currently effective.
- The next step should be a V8-centered diagnostic and simplification phase, not further stacking.
- Strong candidate directions:
  - isolate which V8 components actually matter
  - inspect best checkpoint dynamics
  - check whether auxiliary losses are hurting or helping
  - tune around V8's successful structure instead of adding new branches
  - consider data/label-noise aware training rather than architecture expansion

Recommended first V14 command:

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
  --experiment_name v14-dialogue-state-adapter `
  --theory_variant v14-context-graph-dialogue-state-adapter `
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
  --use_dialogue_state_adapter 1 `
  --dialogue_state_adapter_dim 256 `
  --dialogue_state_adapter_weight 0.3 `
  --dialogue_state_adapter_dropout 0.1 `
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

## V15: Uncertainty-aware Sample Reweighting

Main idea:

- Stop adding architectural branches after V11-V14 regressions.
- Return to V8 as the structural backbone.
- Modify the training objective instead of the model structure.
- Down-weight hard-label CE for high-uncertainty samples after a warmup period.
- Keep auxiliary losses unchanged.
- Normalize sample weights by batch mean to preserve loss scale.

Why this version exists:

- The task has naturally soft/boundary labels.
- V8 is structurally stable, but hard CE may over-force ambiguous samples.
- This version directly supports the uncertainty-aware / three-way decision paper story.

Main switches:

- `--use_uncertainty_reweighting 1`
- `--uncertainty_reweight_strength 0.3`
- `--uncertainty_reweight_min 0.5`
- `--uncertainty_reweight_start_step 1000`
- `--uncertainty_reweight_normalize 1`

Implementation status:

- Code implemented in `modules/trainer/trainer.py`.
- `py_compile` passed.
- `main.py --help` shows new switches.
- Small loss smoke test passed.

Observed result:

- Best checkpoint: `7500`
- Accuracy: `0.3192`
- Macro F1: `0.2564`
- Weighted F1: `0.3023`
- Preference bias: `0.8105`
- Mean uncertainty: `0.7888`

Per-class F1:

- Reflection of feelings: `0.0890`
- Self-disclosure: `0.1159`
- Question: `0.5041`
- Affirmation and Reassurance: `0.1125`
- Providing Suggestions: `0.3756`
- Restatement or Paraphrasing: `0.2802`
- Information: `0.1798`
- Others: `0.3939`

Comparison with V8:

- Macro F1 lower: `0.2564` vs `0.2605`
- Weighted F1 lower: `0.3023` vs `0.3112`
- Accuracy slightly lower: `0.3192` vs `0.3216`
- Preference bias much higher: `0.8105` vs `0.5400`
- Best checkpoint moved to the maximum step `7500`.

Conclusion:

- V15 does not beat V8.
- However, best checkpoint moving to `7500` is meaningful: uncertainty reweighting changed training dynamics and appears to reduce late-stage degradation.
- The current reweighting is likely too blunt. It down-weights uncertain samples but also increases preference bias and weakens weighted F1.
- Future loss-side work should be more selective:
  - delay reweighting later
  - reduce reweight strength
  - avoid down-weighting all uncertain samples uniformly
  - only apply boundary-aware smoothing/reweighting to samples with high uncertainty and low confidence, not all uncertain cases
  - preserve large-class decision strength

Recommended first V15 command:

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
  --experiment_name v15-v8-uncertainty-reweighting `
  --theory_variant v15-uncertainty-aware-sample-reweighting `
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
  --use_uncertainty_reweighting 1 `
  --uncertainty_reweight_strength 0.3 `
  --uncertainty_reweight_min 0.5 `
  --uncertainty_reweight_start_step 1000 `
  --uncertainty_reweight_normalize 1 `
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

## V16: Boundary Soft Label Regularization

Main idea:

- Keep V8 as the structural backbone.
- Do not down-weight all uncertain samples as in V15.
- Add a small soft-label regularization term only for high-uncertainty boundary samples.
- Soft target is a mixture of:
  - hard one-hot label
  - detached current model distribution
- This lets boundary samples be learned more softly without fully ignoring them.

Main switches:

- `--use_boundary_soft_labels 1`
- `--boundary_soft_label_weight 0.03`
- `--boundary_soft_label_alpha 0.2`
- `--boundary_soft_label_threshold 0.75`
- `--boundary_soft_label_start_step 1000`

Implementation status:

- Code implemented in `modules/trainer/trainer.py`.
- `py_compile` passed.
- `main.py --help` shows new switches.
- Small loss smoke test passed.

Observed result:

- Checkpoint: `6500`
- Accuracy: `0.3143`
- Macro F1: `0.2472`
- Weighted F1: `0.3019`
- Preference bias: `0.5614`
- Mean uncertainty: `0.8146`

Per-class F1:

- Reflection of feelings: `0.0076`
- Self-disclosure: `0.1125`
- Question: `0.4970`
- Affirmation and Reassurance: `0.2480`
- Providing Suggestions: `0.3485`
- Restatement or Paraphrasing: `0.2367`
- Information: `0.1575`
- Others: `0.3700`

Conclusion:

- V16 underperforms V8.
- Boundary soft labels improved Affirmation and Information but badly collapsed Reflection and reduced high-support classes.
- Loss-side softening is also not a stable route in its current form.
- After V9-V16, the strongest reliable method remains V8.
- The project should pause architecture/loss expansion and consolidate around V8:
  - reproduce V8
  - run minimal V8-centered ablations
  - analyze V8 errors and boundary samples
  - frame uncertainty/three-way decision as analysis and decision support rather than forcing every mechanism to improve F1

Recommended first V16 command:

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
  --experiment_name v16-v8-boundary-soft-labels `
  --theory_variant v16-boundary-soft-label-regularization `
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
  --use_boundary_soft_labels 1 `
  --boundary_soft_label_weight 0.03 `
  --boundary_soft_label_alpha 0.2 `
  --boundary_soft_label_threshold 0.75 `
  --boundary_soft_label_start_step 1000 `
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

## Overall Lessons

- The most reliable improvement so far comes from better evidence extraction, especially V8 label-aware decoding.
- Direct manipulation of final logits is unstable:
  - granule logit injection can hurt dominant classes
  - hierarchy priors hurt fine-grained boundaries
  - rare expert residual caused severe regression
- Future changes should focus on representation and evidence before decision:
  - richer label-aware evidence tokens
  - turn-level context selection
  - speaker-aware / position-aware turn features
  - label query interaction
  - light contrastive separation on decoded label features
- Avoid adding strong auxiliary losses until a module already proves useful through final CE.

## Current Mainline Recommendation

Use V8 as the strongest confirmed method:

- Macro F1: `0.2605`
- Weighted F1: `0.3112`
- Accuracy: `0.3216`

After V9-V16:

- Stop adding new architecture branches for now.
- Treat V8 as the main method.
- Run V8-centered diagnostics only.
- Diagnostic commands are recorded in `V8_DIAGNOSTIC_COMMANDS.md`.
- Priority:
  - `v8-confirm-10000`
  - `v8-no-granule-aux-10000`
  - `v8-no-view-aux-10000`
  - `v8-label-decoder-w015-10000`
  - `v8-label-decoder-w035-10000`

Rationale:

- V9-V16 repeatedly show that additional priors, evidence banks, adapters, and uncertainty losses disturb V8's stable boundary.
- V8 is already stronger than the original baseline on macro F1, weighted F1, accuracy, and preference bias.
- The next useful work is confirmation, ablation, and analysis rather than more modules.

## V8 Confirm 10000 Preset Result

Preset:

- `--experiment_preset v8_confirm_10000`

Observed result:

- Best checkpoint: `6500`
- Accuracy: `0.3192`
- Macro F1: `0.2519`
- Weighted F1: `0.3054`
- Preference bias: `0.6773`
- Mean uncertainty: `0.8125`

Per-class F1:

- Reflection of feelings: `0.0230`
- Self-disclosure: `0.0856`
- Question: `0.5089`
- Affirmation and Reassurance: `0.2256`
- Providing Suggestions: `0.3477`
- Restatement or Paraphrasing: `0.2824`
- Information: `0.1500`
- Others: `0.3922`

Conclusion:

- This preset did not reproduce the historical V8 result.
- Historical V8 remains stronger:
  - Accuracy: `0.3216`
  - Macro F1: `0.2605`
  - Weighted F1: `0.3112`
  - Preference bias: `0.5400`
- Possible causes:
  - later code changes altered the effective V8 code path even when switches are off
  - historical V8 command/config had small differences not captured in the preset
  - stochastic run variation under one seed/environment
- Next step should inspect historical V8 `run_config.json` and compare it against current preset config.
- Follow-up code fix:
  - Optional modules added after V8 were moved after all V8-era modules in `RobertaHeterogeneousGraph.__init__`.
  - Reason: disabled modules still consumed random initialization and could perturb V8 classifier/gate/decoder parameters.
  - `--experiment_preset v8_confirm_10000` should be rerun after this initialization-order fix.

Result after initialization-order fix:

- Checkpoint: `10000`
- Accuracy: `0.3074`
- Macro F1: `0.2466`
- Weighted F1: `0.2951`
- Preference bias: `0.2967`

Conclusion:

- Even after moving optional modules after the V8 core path, the current codebase still does not reproduce historical V8.
- Historical V8 should be treated as the canonical V8 result because it has its own preserved run directory, config, checkpoint, and test artifacts.
- The current codebase has gone through too many post-V8 edits to be considered a clean reproduction baseline.
- Do not keep spending cycles trying to reconstruct V8 by toggling switches inside the heavily modified code.
- For paper comparison, use the preserved historical V8 run:
  - `roberta-hg-esconv-preprocessed-v8-label-aware-decoder-dynamicgate-seed114514-20260511-024944`
  - Checkpoint: `6000`
  - Accuracy: `0.3216`
  - Macro F1: `0.2605`
  - Weighted F1: `0.3112`
  - Preference bias: `0.5400`
- Debug check:
  - Current code loading historical V8 `checkpoint-6000` exactly reproduces the historical test metrics.
  - Therefore evaluation, data loading, and forward compatibility are intact.
  - The reproduction problem is specific to from-scratch training trajectory.
- Additional reproduction preset:
  - Added `--experiment_preset v8_historical_7500`.
  - It uses historical `total_steps=7500`.
  - It forces `--use_fast_tokenizer 0` to match the original slow `RobertaTokenizer` path.
