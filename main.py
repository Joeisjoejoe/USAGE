import argparse
import os
import torch
from torch.utils.data import DataLoader
from dataloaders import ESConv, DailyDialogue, ESConvPreProcessed, AnnoMIPreProcessed
from transformers import TrainingArguments
from modules.trainer import TrainerForMulticlassClassification
from modules.roberta import RobertaHeterogeneousGraph
from modules.erc import SequentialERC
from utils import seed_everything, collect_runtime_context, ensure_dir

MODELS = {
    "sequential-erc": {
        "model": SequentialERC,
        "trainer": TrainerForMulticlassClassification
    },
    "roberta-hg": {
        "model": RobertaHeterogeneousGraph,
        "trainer": TrainerForMulticlassClassification
    }
}

DATASETS = {
    "esconv": ESConv,
    "esconv-preprocessed": ESConvPreProcessed,
    "dailydialogue": DailyDialogue,
    "annomi-preprocessed": AnnoMIPreProcessed,
}

def apply_experiment_preset(args):
    if args.experiment_preset == "none":
        return args
    if args.experiment_preset not in {"v8_confirm_10000", "v8_historical_7500", "v8_reconstructed"}:
        raise ValueError(f"Unsupported experiment_preset: {args.experiment_preset}")
    is_confirm_10000 = args.experiment_preset == "v8_confirm_10000"
    is_reconstructed = args.experiment_preset == "v8_reconstructed"

    preset_values = {
        "batch_size": 4,
        "total_steps": 10000 if is_confirm_10000 else 7500,
        "total_epochs": 8,
        "save_steps": 500,
        "eval_steps": 500,
        "experiment_name": (
            "v8-confirm-10000"
            if is_confirm_10000
            else "v8-reconstructed-label-aware-decoder" if is_reconstructed else "v8-historical-7500"
        ),
        "theory_variant": (
            "v8-label-aware-decoder-confirm-10000"
            if is_confirm_10000
            else "v8-reconstructed-structure" if is_reconstructed else "v8-historical-config-7500"
        ),
        "use_node_position_embedding": 1,
        "graph_readout": "dummy_mean_max",
        "use_feature_layernorm": 1,
        "use_multiview_logits": 1,
        "fusion_strategy": "dynamic_gate",
        "fusion_gate_hidden_dim": 128,
        "fusion_view_dropout": 0.05,
        "context_aux_loss_weight": 0.12,
        "graph_aux_loss_weight": 0.05,
        "use_label_aware_decoder": 1,
        "label_decoder_dim": 256,
        "label_decoder_heads": 4,
        "label_decoder_dropout": 0.1,
        "label_decoder_logit_weight": 0.25,
        "label_decoder_aux_loss_weight": 0.0,
        "use_uncertainty_reweighting": 0,
        "use_boundary_soft_labels": 0,
        "use_dialogue_state_adapter": 0,
        "use_global_token_label_evidence": 0,
        "use_turn_label_evidence": 0,
        "use_selective_turn_evidence_gate": 0,
        "use_hierarchical_decoder": 0,
        "hierarchy_loss_weight": 0.0,
        "use_rare_granule_expert": 0,
        "use_turn_evidence_encoder": 0,
        "use_text_aware_graph_nodes": 0,
        "use_prototype_head": 0,
        "use_granular_uncertainty": 1,
        "granule_loss_weight": 0.03,
        "granule_logit_weight": 0.0,
    }
    for key, value in preset_values.items():
        setattr(args, key, value)
    return args

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--load_checkpoint', type=int, default=0)
    parser.add_argument('--eval_run_id', type=str, default='')
    parser.add_argument('--experiment_preset', type=str, default='none',
                        choices=['none', 'v8_confirm_10000', 'v8_historical_7500', 'v8_reconstructed'])
    parser.add_argument('--seed', type=int, default=114514)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--total_epochs', type=int, default=10)
    parser.add_argument('--total_steps', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--eval_steps', type=int, default=500)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--warmup', type=int, default=500)
    parser.add_argument('--exclude_others', type=int, default=0)
    parser.add_argument('--feedback_threshold', type=int, default=0)
    parser.add_argument('--erc_temperature', type=float, default=0.5)
    parser.add_argument('--erc_mixed', type=int, default=1)
    parser.add_argument('--hg_dim', type=int, default=512)
    parser.add_argument('--experiment_name', type=str, default='author-default-intent')
    parser.add_argument('--theory_variant', type=str, default='baseline')
    parser.add_argument('--run_notes', type=str, default='')
    parser.add_argument('--use_granular_uncertainty', type=int, default=0)
    parser.add_argument('--granule_loss_weight', type=float, default=0.0)
    parser.add_argument('--granule_logit_weight', type=float, default=0.0)
    parser.add_argument('--granule_temperature', type=float, default=0.2)
    parser.add_argument('--use_node_position_embedding', type=int, default=0)
    parser.add_argument('--graph_readout', type=str, default='dummy', choices=['dummy', 'dummy_mean_max'])
    parser.add_argument('--use_gat_ffn', type=int, default=0)
    parser.add_argument('--use_feature_layernorm', type=int, default=0)
    parser.add_argument('--loss_type', type=str, default='ce', choices=['ce', 'focal', 'logit_adjusted_ce'])
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--logit_adjust_tau', type=float, default=0.2)
    parser.add_argument('--focal_gamma', type=float, default=1.5)
    parser.add_argument('--use_uncertainty_reweighting', type=int, default=0)
    parser.add_argument('--uncertainty_reweight_strength', type=float, default=0.3)
    parser.add_argument('--uncertainty_reweight_min', type=float, default=0.5)
    parser.add_argument('--uncertainty_reweight_start_step', type=int, default=1000)
    parser.add_argument('--uncertainty_reweight_normalize', type=int, default=1)
    parser.add_argument('--use_boundary_soft_labels', type=int, default=0)
    parser.add_argument('--boundary_soft_label_weight', type=float, default=0.05)
    parser.add_argument('--boundary_soft_label_alpha', type=float, default=0.2)
    parser.add_argument('--boundary_soft_label_threshold', type=float, default=0.8)
    parser.add_argument('--boundary_soft_label_start_step', type=int, default=1000)
    parser.add_argument('--use_multiview_logits', type=int, default=0)
    parser.add_argument('--context_logit_weight', type=float, default=0.3)
    parser.add_argument('--graph_logit_weight', type=float, default=0.2)
    parser.add_argument('--context_aux_loss_weight', type=float, default=0.2)
    parser.add_argument('--graph_aux_loss_weight', type=float, default=0.1)
    parser.add_argument('--fusion_strategy', type=str, default='fixed',
                        choices=['fixed', 'dynamic_gate', 'classwise_dynamic_gate', 'uncertainty_weighted'])
    parser.add_argument('--fusion_gate_hidden_dim', type=int, default=128)
    parser.add_argument('--fusion_view_dropout', type=float, default=0.0)
    parser.add_argument('--fusion_gate_temperature', type=float, default=1.0)
    parser.add_argument('--fusion_fixed_residual_weight', type=float, default=0.0)
    parser.add_argument('--use_turn_evidence_encoder', type=int, default=0)
    parser.add_argument('--evidence_encoder_mode', type=str, default='turn_attention',
                        choices=['turn_attention', 'window_cls'])
    parser.add_argument('--evidence_window_size', type=int, default=6)
    parser.add_argument('--evidence_fusion_weight', type=float, default=0.3)
    parser.add_argument('--use_text_aware_graph_nodes', type=int, default=0)
    parser.add_argument('--text_node_fusion_weight', type=float, default=0.5)
    parser.add_argument('--use_prototype_head', type=int, default=0)
    parser.add_argument('--prototype_logit_weight', type=float, default=0.2)
    parser.add_argument('--prototype_temperature', type=float, default=0.2)
    parser.add_argument('--use_label_aware_decoder', type=int, default=0)
    parser.add_argument('--label_decoder_dim', type=int, default=256)
    parser.add_argument('--label_decoder_heads', type=int, default=4)
    parser.add_argument('--label_decoder_dropout', type=float, default=0.1)
    parser.add_argument('--label_decoder_logit_weight', type=float, default=0.25)
    parser.add_argument('--label_decoder_aux_loss_weight', type=float, default=0.0)
    parser.add_argument('--use_turn_label_evidence', type=int, default=0)
    parser.add_argument('--turn_label_window_size', type=int, default=6)
    parser.add_argument('--turn_label_dropout', type=float, default=0.1)
    parser.add_argument('--use_selective_turn_evidence_gate', type=int, default=0)
    parser.add_argument('--turn_evidence_gate_hidden_dim', type=int, default=128)
    parser.add_argument('--turn_evidence_gate_temperature', type=float, default=1.0)
    parser.add_argument('--turn_evidence_residual_weight', type=float, default=1.0)
    parser.add_argument('--use_global_token_label_evidence', type=int, default=0)
    parser.add_argument('--global_token_evidence_max_tokens', type=int, default=32)
    parser.add_argument('--global_token_evidence_dropout', type=float, default=0.1)
    parser.add_argument('--use_dialogue_state_adapter', type=int, default=0)
    parser.add_argument('--dialogue_state_adapter_dim', type=int, default=256)
    parser.add_argument('--dialogue_state_adapter_weight', type=float, default=0.3)
    parser.add_argument('--dialogue_state_adapter_dropout', type=float, default=0.1)
    parser.add_argument('--use_hierarchical_decoder', type=int, default=0)
    parser.add_argument('--hierarchy_logit_weight', type=float, default=0.25)
    parser.add_argument('--hierarchy_loss_weight', type=float, default=0.05)
    parser.add_argument('--hierarchy_group_scheme', type=str, default='strategy_function_4',
                        choices=['strategy_function_4'])
    parser.add_argument('--use_rare_granule_expert', type=int, default=0)
    parser.add_argument('--rare_expert_hidden_dim', type=int, default=256)
    parser.add_argument('--rare_expert_weight', type=float, default=0.25)
    parser.add_argument('--rare_boundary_threshold', type=float, default=0.75)
    parser.add_argument('--rare_boundary_temperature', type=float, default=0.08)
    parser.add_argument('--rare_expert_classes', type=str, default='0,1,3,5,6')
    parser.add_argument('--three_way_accept_threshold', type=float, default=0.45)
    parser.add_argument('--three_way_reject_threshold', type=float, default=0.65)
    parser.add_argument('--calibrate_three_way_thresholds', type=int, default=0)
    parser.add_argument('--three_way_accept_quantile', type=float, default=0.1)
    parser.add_argument('--three_way_reject_quantile', type=float, default=0.9)

    args = parser.parse_args()
    args = apply_experiment_preset(args)

    hf_home = os.environ.get("HF_HOME")
    transformers_cache = os.environ.get("TRANSFORMERS_CACHE")
    if hf_home:
        ensure_dir(hf_home)
    if transformers_cache:
        ensure_dir(transformers_cache)

    seed_everything(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MODELS[args.model]["model"](args)
    model.to(device)
    train_set = DATASETS[args.dataset]("train", args)
    valid_set = DATASETS[args.dataset]("valid", args)
    test_set = DATASETS[args.dataset]("test", args)
    print(f"Total samples: {len(train_set) + len(valid_set) + len(test_set)}")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=train_set.collate_fn)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, collate_fn=valid_set.collate_fn)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=test_set.collate_fn)

    training_args = TrainingArguments(
        output_dir=f"./{args.model}-{args.dataset}-checkpoints",
        num_train_epochs=args.total_epochs,
        warmup_steps=args.warmup,
        weight_decay=args.weight_decay,
        logging_dir=f'./{args.model}-{args.dataset}-logs',
        learning_rate=args.lr,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
    )
    runtime_context = collect_runtime_context(
        args=args,
        device=device,
        dataset_sizes={
            "train": len(train_set),
            "valid": len(valid_set),
            "test": len(test_set),
        },
        workdir=os.getcwd(),
    )
    runtime_context["train_class_counts"] = getattr(train_set, "class_counts", None)
    runtime_context["train_class_priors"] = getattr(train_set, "class_priors", None)
    trainer = MODELS[args.model]["trainer"](
        class_weights=torch.as_tensor(train_set.class_weights).clone().detach(),
        class_priors=torch.as_tensor(getattr(train_set, "class_priors", [])).clone().detach(),
        model=model,
        deciding_metric="macro f1",
        args=training_args,
        total_steps=args.total_steps,
        id2label=train_set.id2label,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        runtime_context=runtime_context,
        source_args=args,
        eval_run_id=args.eval_run_id if args.mode == "test" and args.eval_run_id else None,
    )

    if args.mode == "train":
        trainer.train()
    else:
        trainer.test(args.load_checkpoint)
