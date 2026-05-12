import torch
import os
import json
import datetime
import time
import torch.nn.functional as F
from utils import write_log, ensure_dir, write_json, append_jsonl, make_json_safe
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import transformers
from transformers import AdamW, TrainingArguments
from metrics import preference_bias
import numpy as np


class TrainerForMulticlassClassification:
    def __init__(self,
                 args: TrainingArguments = None,
                 total_steps: int = None,
                 deciding_metric: str = None,
                 class_weights: torch.Tensor = None,
                 class_priors: torch.Tensor = None,
                 id2label: dict = None,
                 model: nn.Module = None,
                 train_loader: DataLoader = None,
                 valid_loader: DataLoader = None,
                 test_loader: DataLoader = None,
                 runtime_context: dict = None,
                 source_args=None,
                 eval_run_id=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.total_steps = total_steps
        self.deciding_metric = deciding_metric
        self.model = model
        self.train_loader = train_loader
        self.id2label = id2label
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.class_weights = class_weights.float().to(self.device)
        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=self.class_weights)
        if class_priors is None or class_priors.numel() == 0:
            class_priors = torch.ones(len(id2label), dtype=torch.float)
            class_priors = class_priors / class_priors.sum()
        self.class_priors = class_priors.float().to(self.device).clamp_min(1e-12)
        self.class_log_priors = torch.log(self.class_priors)
        self.granule_loss_weight = getattr(source_args, "granule_loss_weight", 0.0) if source_args else 0.0
        self.loss_type = getattr(source_args, "loss_type", "ce") if source_args else "ce"
        self.label_smoothing = getattr(source_args, "label_smoothing", 0.0) if source_args else 0.0
        self.logit_adjust_tau = getattr(source_args, "logit_adjust_tau", 0.2) if source_args else 0.2
        self.focal_gamma = getattr(source_args, "focal_gamma", 1.5) if source_args else 1.5
        self.use_uncertainty_reweighting = bool(getattr(source_args, "use_uncertainty_reweighting", 0)) if source_args else False
        self.uncertainty_reweight_strength = getattr(source_args, "uncertainty_reweight_strength", 0.3) if source_args else 0.3
        self.uncertainty_reweight_min = getattr(source_args, "uncertainty_reweight_min", 0.5) if source_args else 0.5
        self.uncertainty_reweight_start_step = getattr(source_args, "uncertainty_reweight_start_step", 1000) if source_args else 1000
        self.uncertainty_reweight_normalize = bool(getattr(source_args, "uncertainty_reweight_normalize", 1)) if source_args else True
        self.use_boundary_soft_labels = bool(getattr(source_args, "use_boundary_soft_labels", 0)) if source_args else False
        self.boundary_soft_label_weight = getattr(source_args, "boundary_soft_label_weight", 0.05) if source_args else 0.05
        self.boundary_soft_label_alpha = getattr(source_args, "boundary_soft_label_alpha", 0.2) if source_args else 0.2
        self.boundary_soft_label_threshold = getattr(source_args, "boundary_soft_label_threshold", 0.8) if source_args else 0.8
        self.boundary_soft_label_start_step = getattr(source_args, "boundary_soft_label_start_step", 1000) if source_args else 1000
        self.context_aux_loss_weight = getattr(source_args, "context_aux_loss_weight", 0.0) if source_args else 0.0
        self.graph_aux_loss_weight = getattr(source_args, "graph_aux_loss_weight", 0.0) if source_args else 0.0
        self.label_decoder_aux_loss_weight = getattr(source_args, "label_decoder_aux_loss_weight", 0.0) if source_args else 0.0
        self.hierarchy_loss_weight = getattr(source_args, "hierarchy_loss_weight", 0.0) if source_args else 0.0
        self.three_way_accept_threshold = getattr(source_args, "three_way_accept_threshold", 0.45) if source_args else 0.45
        self.three_way_reject_threshold = getattr(source_args, "three_way_reject_threshold", 0.65) if source_args else 0.65
        self.calibrate_three_way_thresholds = bool(getattr(source_args, "calibrate_three_way_thresholds", 0)) if source_args else False
        self.three_way_accept_quantile = getattr(source_args, "three_way_accept_quantile", 0.1) if source_args else 0.1
        self.three_way_reject_quantile = getattr(source_args, "three_way_reject_quantile", 0.9) if source_args else 0.9
        self.threshold_calibration = None
        self.current_step = 0
        self.best_ckpt = None
        self.best_valid_metrics = None
        self.runtime_context = runtime_context or {}
        self.source_args = source_args
        self.run_id = eval_run_id or self.runtime_context.get(
            "run_id",
            datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        )
        self.eval_run_id = eval_run_id
        self.run_log_dir = ensure_dir(os.path.join(self.args.logging_dir, self.run_id))
        self.run_output_dir = ensure_dir(os.path.join(self.args.output_dir, self.run_id))
        self.paths = {
            "run_config": os.path.join(self.run_log_dir, "run_config.json"),
            "eval_config": os.path.join(self.run_log_dir, "eval_config.json"),
            "train_history": os.path.join(self.run_log_dir, "train_history.jsonl"),
            "summary": os.path.join(self.run_log_dir, "summary.json"),
            "summary_eval": os.path.join(self.run_log_dir, "summary_eval.json"),
            "analysis": os.path.join(self.run_log_dir, "analysis.json"),
            "analysis_eval": os.path.join(self.run_log_dir, "analysis_eval.json"),
            "result": os.path.join(self.run_log_dir, "result.json"),
            "result_eval": os.path.join(self.run_log_dir, "result_eval.json"),
            "cases": os.path.join(self.run_log_dir, "cases.json"),
            "cases_eval": os.path.join(self.run_log_dir, "cases_eval.json"),
            "log": os.path.join(self.run_log_dir, "train.log"),
        }
        self.training_stats = {
            "status": "initialized",
            "oom": False,
            "interrupted": False,
            "train_start_time": None,
            "train_end_time": None,
            "total_training_time_sec": None,
            "avg_step_time_sec": None,
            "peak_gpu_memory_mb": 0.0,
        }

        if not os.path.exists(self.args.logging_dir):
            os.makedirs(self.args.logging_dir)
        if self.eval_run_id is None:
            self._write_run_config()
        else:
            self._write_eval_config()

    def _write_run_config(self):
        run_config = {
            "run_id": self.run_id,
            "runtime_context": make_json_safe(self.runtime_context),
            "training_arguments": make_json_safe(vars(self.args)),
            "source_arguments": make_json_safe(vars(self.source_args)) if self.source_args else {},
            "paths": self.paths,
        }
        write_json(run_config, self.paths["run_config"])

    def _write_eval_config(self):
        eval_config = {
            "eval_run_id": self.eval_run_id,
            "runtime_context": make_json_safe(self.runtime_context),
            "training_arguments": make_json_safe(vars(self.args)),
            "source_arguments": make_json_safe(vars(self.source_args)) if self.source_args else {},
            "paths": self.paths,
        }
        write_json(eval_config, self.paths["eval_config"])

    def _record_history(self, record):
        append_jsonl(make_json_safe(record), self.paths["train_history"])

    def _format_metric_value(self, value):
        if value is None:
            return "None"
        if isinstance(value, (int, float, np.integer, np.floating)):
            return f"{value:.4f}"
        return str(value)

    def _max_gpu_memory_mb(self):
        if not torch.cuda.is_available():
            return 0.0
        return round(torch.cuda.max_memory_allocated() / (1024 ** 2), 4)

    def _three_way_decisions(self, uncertainty_scores):
        decisions = np.full(uncertainty_scores.shape, "boundary", dtype=object)
        decisions[uncertainty_scores <= self.three_way_accept_threshold] = "accept"
        decisions[uncertainty_scores >= self.three_way_reject_threshold] = "reject"
        return decisions

    def _calibrate_three_way_thresholds(self, uncertainty_records, split_name="valid"):
        if not uncertainty_records:
            return None
        accept_q = min(max(self.three_way_accept_quantile, 0.0), 1.0)
        reject_q = min(max(self.three_way_reject_quantile, 0.0), 1.0)
        if accept_q >= reject_q:
            raise ValueError("three_way_accept_quantile must be lower than three_way_reject_quantile")
        uncertainty_scores = np.array([r["uncertainty_score"] for r in uncertainty_records])
        self.three_way_accept_threshold = float(np.quantile(uncertainty_scores, accept_q))
        self.three_way_reject_threshold = float(np.quantile(uncertainty_scores, reject_q))
        self.threshold_calibration = {
            "enabled": True,
            "split": split_name,
            "accept_quantile": float(accept_q),
            "reject_quantile": float(reject_q),
            "accept_threshold": self.three_way_accept_threshold,
            "reject_threshold": self.three_way_reject_threshold,
            "sample_count": int(len(uncertainty_scores)),
            "mean_uncertainty": float(uncertainty_scores.mean()),
        }
        return self.threshold_calibration

    def _uncertainty_metrics(self, predictions, truths, uncertainty_records):
        if not uncertainty_records:
            return {}
        uncertainty_scores = np.array([r["uncertainty_score"] for r in uncertainty_records])
        decisions = self._three_way_decisions(uncertainty_scores)
        correct = predictions == truths
        metrics = {
            "mean uncertainty": float(uncertainty_scores.mean()),
            "three way accept rate": float(np.mean(decisions == "accept")),
            "three way boundary rate": float(np.mean(decisions == "boundary")),
            "three way reject rate": float(np.mean(decisions == "reject")),
        }
        for decision in ["accept", "boundary", "reject"]:
            mask = decisions == decision
            metrics[f"{decision} accuracy"] = float(correct[mask].mean()) if mask.any() else None
        return metrics

    def _build_analysis(self, predictions, truths, logits, confusion_matrix, cases, uncertainty_records=None):
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        confidences = probs.max(axis=-1)
        errors = predictions != truths
        supports = {}
        for idx, label in self.id2label.items():
            supports[label] = int(np.sum(truths == idx))
        top_confusions = []
        c_matrix = np.array(confusion_matrix)
        for i in range(c_matrix.shape[0]):
            for j in range(c_matrix.shape[1]):
                if i == j or c_matrix[i, j] == 0:
                    continue
                top_confusions.append({
                    "truth": self.id2label[i],
                    "prediction": self.id2label[j],
                    "count": int(c_matrix[i, j]),
                })
        top_confusions = sorted(top_confusions, key=lambda x: x["count"], reverse=True)[:10]
        low_confidence_indices = np.where(confidences < 0.4)[0].tolist()
        analysis = {
            "error_count": int(errors.sum()),
            "correct_count": int((~errors).sum()),
            "error_rate": float(errors.mean()),
            "mean_confidence": float(confidences.mean()),
            "mean_error_confidence": float(confidences[errors].mean()) if errors.any() else None,
            "mean_correct_confidence": float(confidences[~errors].mean()) if (~errors).any() else None,
            "low_confidence_prediction_count": len(low_confidence_indices),
            "low_confidence_prediction_indices": low_confidence_indices[:50],
            "error_indices": np.where(errors)[0].tolist()[:100],
            "supports_by_label": supports,
            "top_confusions": top_confusions,
            "case_count": len(cases),
        }
        if uncertainty_records:
            uncertainty_scores = np.array([r["uncertainty_score"] for r in uncertainty_records])
            decisions = self._three_way_decisions(uncertainty_scores)
            decision_summary = {}
            for decision in ["accept", "boundary", "reject"]:
                mask = decisions == decision
                decision_summary[decision] = {
                    "count": int(mask.sum()),
                    "accuracy": float((predictions[mask] == truths[mask]).mean()) if mask.any() else None,
                    "mean_uncertainty": float(uncertainty_scores[mask].mean()) if mask.any() else None,
                    "error_count": int((predictions[mask] != truths[mask]).sum()) if mask.any() else 0,
                }
            boundary_indices = np.where(decisions == "boundary")[0].tolist()
            reject_indices = np.where(decisions == "reject")[0].tolist()
            analysis["uncertainty"] = {
                "mean_uncertainty": float(uncertainty_scores.mean()),
                "mean_error_uncertainty": float(uncertainty_scores[errors].mean()) if errors.any() else None,
                "mean_correct_uncertainty": float(uncertainty_scores[~errors].mean()) if (~errors).any() else None,
                "three_way_thresholds": {
                    "accept": self.three_way_accept_threshold,
                    "reject": self.three_way_reject_threshold,
                },
                "threshold_calibration": self.threshold_calibration,
                "decision_summary": decision_summary,
                "boundary_indices": boundary_indices[:100],
                "reject_indices": reject_indices[:100],
            }
        return analysis

    def _classification_loss(self, logits, labels, sample_weights=None):
        reduction = "none" if sample_weights is not None else "mean"
        if self.loss_type == "logit_adjusted_ce":
            logits = logits + self.logit_adjust_tau * self.class_log_priors.unsqueeze(0)
            loss = F.cross_entropy(
                logits,
                labels,
                weight=self.class_weights,
                label_smoothing=self.label_smoothing,
                reduction=reduction,
            )
            return self._reduce_sample_weighted_loss(loss, sample_weights)
        if self.loss_type == "focal":
            ce_loss = F.cross_entropy(
                logits,
                labels,
                weight=self.class_weights,
                label_smoothing=self.label_smoothing,
                reduction="none",
            )
            pt = torch.exp(-ce_loss)
            loss = ((1.0 - pt) ** self.focal_gamma) * ce_loss
            return self._reduce_sample_weighted_loss(loss, sample_weights)
        loss = F.cross_entropy(
            logits,
            labels,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
            reduction=reduction,
        )
        return self._reduce_sample_weighted_loss(loss, sample_weights)

    def _reduce_sample_weighted_loss(self, loss, sample_weights=None):
        if sample_weights is None:
            return loss.mean() if loss.dim() > 0 else loss
        return (loss * sample_weights.to(loss.device)).mean()

    def _compute_uncertainty_sample_weights(self, outputs, logits):
        if (
            not self.use_uncertainty_reweighting
            or self.current_step < self.uncertainty_reweight_start_step
        ):
            return None
        uncertainty = outputs.get("uncertainty")
        if uncertainty is not None and uncertainty.get("uncertainty_score") is not None:
            uncertainty_scores = uncertainty["uncertainty_score"].detach()
        else:
            probs = torch.softmax(logits.detach(), dim=-1)
            top_probs = torch.topk(probs, k=2, dim=-1).values
            confidence = top_probs[:, 0]
            margin = top_probs[:, 0] - top_probs[:, 1]
            entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1) / np.log(probs.shape[-1])
            uncertainty_scores = (entropy + (1.0 - confidence) + (1.0 - margin)) / 3.0
        sample_weights = 1.0 - self.uncertainty_reweight_strength * uncertainty_scores
        sample_weights = torch.clamp(sample_weights, min=self.uncertainty_reweight_min, max=1.0)
        if self.uncertainty_reweight_normalize:
            sample_weights = sample_weights / sample_weights.mean().clamp_min(1e-6)
        return sample_weights

    def _compute_uncertainty_scores(self, outputs, logits):
        uncertainty = outputs.get("uncertainty")
        if uncertainty is not None and uncertainty.get("uncertainty_score") is not None:
            return uncertainty["uncertainty_score"].detach()
        probs = torch.softmax(logits.detach(), dim=-1)
        top_probs = torch.topk(probs, k=2, dim=-1).values
        confidence = top_probs[:, 0]
        margin = top_probs[:, 0] - top_probs[:, 1]
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1) / np.log(probs.shape[-1])
        return (entropy + (1.0 - confidence) + (1.0 - margin)) / 3.0

    def _boundary_soft_label_loss(self, outputs, labels):
        if (
            not self.use_boundary_soft_labels
            or self.current_step < self.boundary_soft_label_start_step
        ):
            return None
        logits = outputs.get("logits")
        uncertainty_scores = self._compute_uncertainty_scores(outputs, logits)
        boundary_mask = uncertainty_scores >= self.boundary_soft_label_threshold
        if not boundary_mask.any():
            return None
        selected_logits = logits[boundary_mask]
        selected_labels = labels[boundary_mask]
        model_targets = torch.softmax(selected_logits.detach(), dim=-1)
        hard_targets = F.one_hot(selected_labels, num_classes=selected_logits.shape[-1]).float()
        soft_targets = (
            (1.0 - self.boundary_soft_label_alpha) * hard_targets
            + self.boundary_soft_label_alpha * model_targets
        )
        log_probs = F.log_softmax(selected_logits, dim=-1)
        class_weights = self.class_weights.unsqueeze(0)
        soft_loss = -(soft_targets * log_probs * class_weights).sum(dim=-1)
        return soft_loss.mean()

    def _compute_loss(self, outputs, labels):
        labels = labels.to(self.device)
        logits = outputs.get("logits")
        sample_weights = self._compute_uncertainty_sample_weights(outputs, logits)
        loss = self._classification_loss(logits, labels, sample_weights=sample_weights)
        boundary_soft_loss = self._boundary_soft_label_loss(outputs, labels)
        if boundary_soft_loss is not None:
            loss = loss + self.boundary_soft_label_weight * boundary_soft_loss
        if self.context_aux_loss_weight > 0 and outputs.get("context_logits") is not None:
            loss = loss + self.context_aux_loss_weight * self._classification_loss(outputs["context_logits"], labels)
        if self.graph_aux_loss_weight > 0 and outputs.get("graph_logits") is not None:
            loss = loss + self.graph_aux_loss_weight * self._classification_loss(outputs["graph_logits"], labels)
        if self.label_decoder_aux_loss_weight > 0 and outputs.get("label_decoder_logits") is not None:
            loss = loss + self.label_decoder_aux_loss_weight * self._classification_loss(outputs["label_decoder_logits"], labels)
        if self.hierarchy_loss_weight > 0 and outputs.get("hierarchy_logits") is not None:
            group_ids = outputs["hierarchy_label_group_ids"].to(labels.device)
            hierarchy_labels = group_ids[labels]
            loss = loss + self.hierarchy_loss_weight * F.cross_entropy(outputs["hierarchy_logits"], hierarchy_labels)
        if self.granule_loss_weight > 0 and outputs.get("granule_logits") is not None:
            loss = loss + self.granule_loss_weight * self.cross_entropy_loss(outputs["granule_logits"], labels)
        return loss

    def _write_summary(self, test_metrics=None, selected_checkpoint=None, eval_mode=False):
        summary = {
            "run_id": self.run_id,
            "eval_mode": eval_mode,
            "status": "evaluated" if eval_mode else self.training_stats["status"],
            "best_checkpoint": self.best_ckpt,
            "selected_checkpoint": selected_checkpoint if selected_checkpoint is not None else self.best_ckpt,
            "best_valid_metrics": self.best_valid_metrics,
            "test_metrics": test_metrics,
            "threshold_calibration": self.threshold_calibration,
            "training_stats": self.training_stats,
            "paths": self.paths,
        }
        output_path = self.paths["summary_eval"] if eval_mode else self.paths["summary"]
        write_json(make_json_safe(summary), output_path)

    def evaluate(self, loader, case_study=False, return_predictions=False):
        cases = []
        self.model.eval()
        with torch.no_grad():
            predictions = []
            truths = []
            logits_all = []
            uncertainty_records = []
            bar = tqdm(loader)
            for _, batch in enumerate(bar):
                outputs = self.model(batch)
                y_pred = outputs.get("logits")
                logits_all.append(y_pred.detach().cpu())
                batch_uncertainty = None
                if outputs.get("uncertainty") is not None:
                    batch_uncertainty = {
                        k: v.detach().cpu().numpy() for k, v in outputs["uncertainty"].items()
                    }
                    for i in range(y_pred.shape[0]):
                        uncertainty_records.append({
                            k: float(v[i]) for k, v in batch_uncertainty.items()
                        })
                if case_study:
                    contexts = batch["dialogue_history"]
                    preds = torch.argmax(y_pred, dim=-1).int().cpu().detach().numpy()
                    labels = batch.get("label").detach().numpy()
                    decisions = None
                    if batch_uncertainty is not None:
                        decisions = self._three_way_decisions(batch_uncertainty["uncertainty_score"])
                    attention_weights = []
                    graph_size = len(outputs["graphs"][0]["nodes"]) - 1
                    for w in outputs["attention_weights"]:
                        attention_weights.append(w[1].detach().cpu().numpy()[-graph_size:].tolist())
                    attention_weights = np.array(attention_weights).squeeze().transpose().tolist()
                    for i in range(len(contexts)):
                        case = {
                            "dialogue_history": [contexts[i], ],
                            "strategy_history": [batch["strategy_history"][i], ],
                            "speaker_turn": [str(batch["speaker_turn"][i]), ],
                            "prediction": self.id2label[preds[i]],
                            "label": self.id2label[labels[i]],
                            "graph": outputs["graphs"],
                            "attention_weights": attention_weights,
                            "erc_logits": outputs["erc_logits"].cpu().detach().numpy().tolist()
                        }
                        if batch_uncertainty is not None:
                            case["uncertainty"] = {
                                k: float(v[i]) for k, v in batch_uncertainty.items()
                            }
                            case["three_way_decision"] = str(decisions[i])
                        cases.append(case)
                predictions.append(torch.argmax(y_pred, dim=-1))
                truths.append(batch.get("label"))
            predictions = torch.cat(predictions, dim=-1).int().cpu().detach().numpy()
            truths = torch.cat(truths, dim=-1).detach().numpy()
            logits = torch.cat(logits_all, dim=0).numpy()
            acc = accuracy_score(truths, predictions)
            f1 = f1_score(truths, predictions, average=None)
            weighted_f1 = f1_score(truths, predictions, average='weighted')
            macro_f1 = f1_score(truths, predictions, average='macro')
            micro_f1 = f1_score(truths, predictions, average='micro')
            c_matrix = confusion_matrix(truths, predictions)
            metrics = {
                'accuracy': acc,
                'macro f1': macro_f1,
                'micro f1': micro_f1,
                'weighted f1': weighted_f1,
                'confusion matrix': c_matrix.tolist(),
                'preference bias': preference_bias(c_matrix)
            }
            metrics.update(self._uncertainty_metrics(predictions, truths, uncertainty_records))
            for _id in range(len(self.id2label.keys())):
                metrics[self.id2label[_id]] = f1[_id]
            if case_study or return_predictions:
                extra = {
                    "predictions": predictions,
                    "truths": truths,
                    "logits": logits,
                    "uncertainty": uncertainty_records,
                }
                if case_study:
                    return metrics, cases, extra
                return metrics, extra
            return metrics

    def train(self):
        best_checkpoint = 0
        best_metric = 0
        step_times = []
        self.training_stats["status"] = "running"
        self.training_stats["train_start_time"] = datetime.datetime.now().isoformat()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_params = [
            {
                "params": [p for n, p in self.model.named_parameters() if
                           p.requires_grad and not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if
                           p.requires_grad and any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_params, lr=self.args.learning_rate)
        total_steps = self.args.num_train_epochs * len(self.train_loader)
        scheduler = transformers.optimization.get_linear_schedule_with_warmup(optimizer,
                                                                              num_warmup_steps=self.args.warmup_steps,
                                                                              num_training_steps=int(total_steps))
        step_counter = 0
        try:
            for epoch in range(1, int(self.args.num_train_epochs) + 1):
                stop_training = 0
                self.model.train()
                loss = torch.tensor(0)
                bar = tqdm(self.train_loader)
                for _, batch in enumerate(bar):
                    step_start = time.time()
                    optimizer.zero_grad()
                    step_counter += 1
                    self.current_step = step_counter
                    bar.set_description(f"Epoch {epoch}| Step {step_counter} | Loss: {loss:.4f}")
                    outputs = self.model(batch)
                    loss = self._compute_loss(outputs, batch.get("label"))
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    step_times.append(time.time() - step_start)
                    if step_counter % self.args.save_steps == 0:
                        save_path = os.path.join(self.run_output_dir, f"checkpoint-{step_counter}.pth")
                        self.model.save(save_path)
                    if step_counter % self.args.eval_steps == 0:
                        metrics = self.evaluate(self.valid_loader)
                        if metrics[self.deciding_metric] > best_metric:
                            best_metric = metrics[self.deciding_metric]
                            best_checkpoint = step_counter
                            self.best_valid_metrics = make_json_safe(metrics)
                        msg = f"Evaluation Step {step_counter} | "
                        for k, v in metrics.items():
                            if k != "confusion matrix":
                                msg += f"{k}: {self._format_metric_value(v)}, "
                        print(msg)
                        write_log(msg, self.paths["log"])
                        self._record_history({
                            "event": "validation",
                            "epoch": epoch,
                            "step": step_counter,
                            "metrics": metrics,
                            "is_best": best_checkpoint == step_counter,
                        })
                    if step_counter == self.total_steps:
                        stop_training = 1
                        break
                if stop_training:
                    break
            self.training_stats["status"] = "completed"
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                self.training_stats["status"] = "oom"
                self.training_stats["oom"] = True
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self._write_summary()
            else:
                self.training_stats["status"] = "failed"
                self._write_summary()
            raise
        except KeyboardInterrupt:
            self.training_stats["status"] = "interrupted"
            self.training_stats["interrupted"] = True
            self._write_summary()
            raise
        finally:
            self.training_stats["train_end_time"] = datetime.datetime.now().isoformat()
            start_dt = datetime.datetime.fromisoformat(self.training_stats["train_start_time"])
            end_dt = datetime.datetime.fromisoformat(self.training_stats["train_end_time"])
            self.training_stats["total_training_time_sec"] = round((end_dt - start_dt).total_seconds(), 4)
            self.training_stats["avg_step_time_sec"] = round(float(np.mean(step_times)), 4) if step_times else None
            self.training_stats["peak_gpu_memory_mb"] = self._max_gpu_memory_mb()
        print(f"Best checkpoint: Step {best_checkpoint}")
        self.best_ckpt = best_checkpoint
        self.test()

    def test(self, ckpt=None):
        load_ckpt = ckpt if ckpt else self.best_ckpt
        print("Testing ...")
        ckpt_path = os.path.join(self.run_output_dir, f"checkpoint-{load_ckpt}.pth")
        self.model.load(ckpt_path)
        eval_mode = self.eval_run_id is not None
        if self.calibrate_three_way_thresholds:
            _, valid_extra = self.evaluate(self.valid_loader, return_predictions=True)
            calibration = self._calibrate_three_way_thresholds(valid_extra.get("uncertainty"), split_name="valid")
            if calibration:
                print(
                    "Calibrated three-way thresholds | "
                    f"accept: {calibration['accept_threshold']:.4f}, "
                    f"reject: {calibration['reject_threshold']:.4f}"
                )
        test_metrics, cases, extra = self.evaluate(self.test_loader, case_study=True)
        result_path = self.paths["result_eval"] if eval_mode else self.paths["result"]
        cases_path = self.paths["cases_eval"] if eval_mode else self.paths["cases"]
        analysis_path = self.paths["analysis_eval"] if eval_mode else self.paths["analysis"]
        write_json(make_json_safe(test_metrics), result_path)
        write_json(make_json_safe(cases), cases_path)
        analysis = self._build_analysis(
            predictions=extra["predictions"],
            truths=extra["truths"],
            logits=extra["logits"],
            confusion_matrix=test_metrics["confusion matrix"],
            cases=cases,
            uncertainty_records=extra.get("uncertainty"),
        )
        write_json(make_json_safe(analysis), analysis_path)
        self._write_summary(test_metrics=test_metrics, selected_checkpoint=load_ckpt, eval_mode=eval_mode)
        msg = f"Test result for checkpoint {load_ckpt} | "
        for k, v in test_metrics.items():
            if k != "confusion matrix":
                msg += f"{k}: {self._format_metric_value(v)}, "
        print(msg)
