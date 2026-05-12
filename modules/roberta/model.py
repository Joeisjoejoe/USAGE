from transformers import RobertaModel, RobertaConfig, RobertaTokenizer
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import json
from modules.sddp import StructuredDialogueDiscourseParser
from modules.erc import SequentialERC
from modules.decoder import RobertaClassificationHead
from torch_geometric.nn.conv import RGATConv


class FFN(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, dropout):
        super(FFN, self).__init__()
        self.linear = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class GATLayer(nn.Module):
    def __init__(self, dim_in, dim_out, num_relations, dropout, use_ffn=False):
        super(GATLayer, self).__init__()
        self.conv = RGATConv(in_channels=dim_in, out_channels=dim_out, num_relations=num_relations)
        self.use_ffn = use_ffn
        self.dropout = nn.Dropout(dropout)
        self.ffn = FFN(dim_in=dim_out, dim_hidden=dim_in // 2, dim_out=dim_out, dropout=dropout)

    def forward(self, x, edge_index, edge_type):
        _x = x
        x, attention_weights = self.conv(x, edge_index, edge_type, return_attention_weights=True)
        x = x + _x
        if self.use_ffn:
            x = x + self.ffn(self.dropout(x))
        return x, attention_weights


class RobertaBase(nn.Module):
    def __init__(self, args=None):
        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # RoBERTa encoder
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.roberta_config = RobertaConfig.from_pretrained("roberta-base")
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    def _reset_slow_tokenizer(self):
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    def _normalize_tokenizer_inputs(self, texts):
        if isinstance(texts, str):
            return texts
        normalized_texts = []
        changed = False
        for text in texts:
            if text is None:
                normalized_texts.append("")
                changed = True
            elif isinstance(text, str):
                normalized_texts.append(text)
            elif isinstance(text, (list, tuple)):
                normalized_texts.append(" ".join("" if item is None else str(item) for item in text))
                changed = True
            else:
                normalized_texts.append(str(text))
                changed = True
        if changed:
            preview = [repr(text)[:120] for text in normalized_texts[:2]]
            print(f"Warning: normalized non-string tokenizer inputs. batch_size={len(normalized_texts)}, preview={preview}")
        return normalized_texts

    def _tokenize(self, texts):
        texts = self._normalize_tokenizer_inputs(texts)
        try:
            return self.roberta_tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        except (KeyError, RuntimeError, SystemError, ValueError) as exc:
            preview_texts = [texts] if isinstance(texts, str) else texts[:2]
            preview = [repr(text)[:200] for text in preview_texts]
            print(
                "Warning: slow RoBERTa tokenizer failed; "
                f"reloading slow tokenizer and retrying once. error={type(exc).__name__}: {exc}; "
                f"batch_size={1 if isinstance(texts, str) else len(texts)}, preview={preview}"
            )
            self._reset_slow_tokenizer()
            return self.roberta_tokenizer(texts, return_tensors='pt', padding=True, truncation=True)


    def encode(self, texts):
        tokens = self._tokenize(texts)
        outputs = self.roberta(input_ids=tokens["input_ids"].to(self.device),
                               attention_mask=tokens["attention_mask"].to(self.device))
        embeddings = outputs.last_hidden_state
        return embeddings[:, 0, :]  # take <s> token (equiv. to [CLS])

    def encode_with_token_embeddings(self, texts):
        tokens = self._tokenize(texts)
        attention_mask = tokens["attention_mask"].to(self.device)
        outputs = self.roberta(
            input_ids=tokens["input_ids"].to(self.device),
            attention_mask=attention_mask,
        )
        embeddings = outputs.last_hidden_state
        return embeddings[:, 0, :], embeddings, attention_mask

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=torch.device('cpu')), strict=False)


class RobertaHeterogeneousGraph(RobertaBase):
    def __init__(self, args, lightmode=True):
        super().__init__(args)

        self.args = args
        self.lightmode = lightmode
        graph_dim = args.hg_dim

        if "esconv" in args.dataset:
            self.strategy2id = json.load(open('data/esconv/strategies.json', 'r'))
        elif "annomi" in args.dataset:
            self.strategy2id = json.load(open('data/annomi/strategies.json', 'r'))
        self.id2strategy = {v: k for k, v in self.strategy2id.items()}
        self.id2emotion = {0: 'Neutral', 1: 'Anger', 2: 'Disgust', 3: 'Fear', 4: 'Joy', 5: 'Sadness', 6: 'Surprise'}
        self.use_node_position_embedding = bool(getattr(args, "use_node_position_embedding", 0))
        self.graph_readout = getattr(args, "graph_readout", "dummy")
        self.use_feature_layernorm = bool(getattr(args, "use_feature_layernorm", 0))
        self.use_multiview_logits = bool(getattr(args, "use_multiview_logits", 0))
        self.context_logit_weight = getattr(args, "context_logit_weight", 0.3)
        self.graph_logit_weight = getattr(args, "graph_logit_weight", 0.2)
        self.fusion_strategy = getattr(args, "fusion_strategy", "fixed")
        self.fusion_view_dropout = getattr(args, "fusion_view_dropout", 0.0)
        self.fusion_gate_temperature = max(getattr(args, "fusion_gate_temperature", 1.0), 1e-6)
        self.fusion_fixed_residual_weight = min(
            max(getattr(args, "fusion_fixed_residual_weight", 0.0), 0.0),
            1.0,
        )
        self.use_turn_evidence_encoder = bool(getattr(args, "use_turn_evidence_encoder", 0))
        self.evidence_encoder_mode = getattr(args, "evidence_encoder_mode", "turn_attention")
        self.evidence_window_size = max(getattr(args, "evidence_window_size", 6), 1)
        self.evidence_fusion_weight = getattr(args, "evidence_fusion_weight", 0.3)
        self.use_text_aware_graph_nodes = bool(getattr(args, "use_text_aware_graph_nodes", 0))
        self.text_node_fusion_weight = getattr(args, "text_node_fusion_weight", 0.5)
        self.use_prototype_head = bool(getattr(args, "use_prototype_head", 0))
        self.prototype_logit_weight = getattr(args, "prototype_logit_weight", 0.2)
        self.prototype_temperature = max(getattr(args, "prototype_temperature", 0.2), 1e-6)
        self.use_label_aware_decoder = bool(getattr(args, "use_label_aware_decoder", 0))
        self.label_decoder_dim = getattr(args, "label_decoder_dim", 256)
        self.label_decoder_logit_weight = getattr(args, "label_decoder_logit_weight", 0.25)
        self.use_turn_label_evidence = bool(getattr(args, "use_turn_label_evidence", 0))
        self.turn_label_window_size = max(getattr(args, "turn_label_window_size", 6), 1)
        self.use_selective_turn_evidence_gate = bool(getattr(args, "use_selective_turn_evidence_gate", 0))
        self.turn_evidence_gate_temperature = max(getattr(args, "turn_evidence_gate_temperature", 1.0), 1e-6)
        self.turn_evidence_residual_weight = getattr(args, "turn_evidence_residual_weight", 1.0)
        self.use_global_token_label_evidence = bool(getattr(args, "use_global_token_label_evidence", 0))
        self.global_token_evidence_max_tokens = max(getattr(args, "global_token_evidence_max_tokens", 32), 1)
        self.use_dialogue_state_adapter = bool(getattr(args, "use_dialogue_state_adapter", 0))
        self.dialogue_state_adapter_weight = getattr(args, "dialogue_state_adapter_weight", 0.3)
        self.use_hierarchical_decoder = bool(getattr(args, "use_hierarchical_decoder", 0))
        self.hierarchy_logit_weight = getattr(args, "hierarchy_logit_weight", 0.25)
        self.hierarchy_group_scheme = getattr(args, "hierarchy_group_scheme", "strategy_function_4")
        self.use_rare_granule_expert = bool(getattr(args, "use_rare_granule_expert", 0))
        self.rare_expert_weight = getattr(args, "rare_expert_weight", 0.25)
        self.rare_boundary_threshold = getattr(args, "rare_boundary_threshold", 0.75)
        self.rare_boundary_temperature = max(getattr(args, "rare_boundary_temperature", 0.08), 1e-6)

        # Frozen Pre-trained Models
        if not lightmode:
            self.dialogue_parser = StructuredDialogueDiscourseParser(ckpt_path="pre_trained_models/sddp_stac")
            for name, param in self.dialogue_parser.model.named_parameters():
                param.requires_grad = False
            self.erc = SequentialERC()
            self.erc.load("pre_trained_models/sequential_erc_model.pth")
            for name, param in self.erc.named_parameters():
                param.requires_grad = False

        self.erc_prototypes = nn.Parameter(torch.randn((7, graph_dim)))
        self.softmax = nn.Softmax(dim=-1)
        self.scalar = 100
        self.t = nn.Parameter(torch.tensor(args.erc_temperature / self.scalar))

        # GCN layers
        encoder_hidden_size = self.roberta_config.hidden_size
        self.graph_relation_dict = {"Continuation": 0, "Question-answer_pair": 1, "Contrast": 2, "Q-Elab": 3,
                                    "Explanation": 4, "Comment": 5, "Background": 6, "Result": 7, "Correction": 8,
                                    "Parallel": 9, "Alternation": 10, "Conditional": 11, "Clarification_question": 12,
                                    "Acknowledgement": 13, "Elaboration": 14, "Narration": 15, "Special": 16,
                                    "Self": 17, "Inter": 18}
        self.graph_relation_dict_inverse = {v: k for k, v in self.graph_relation_dict.items()}
        use_gat_ffn = bool(getattr(args, "use_gat_ffn", 0))
        self.conv1 = GATLayer(dim_in=graph_dim, dim_out=graph_dim, num_relations=len(self.graph_relation_dict.keys()),
                              dropout=0.2, use_ffn=use_gat_ffn)
        self.conv2 = GATLayer(dim_in=graph_dim, dim_out=graph_dim, num_relations=len(self.graph_relation_dict.keys()),
                              dropout=0.2, use_ffn=use_gat_ffn)
        self.conv3 = GATLayer(dim_in=graph_dim, dim_out=graph_dim, num_relations=len(self.graph_relation_dict.keys()),
                              dropout=0.2, use_ffn=use_gat_ffn)
        self.dummy_embedding = nn.Parameter(torch.randn(graph_dim))
        self.strategy_embedding = nn.Embedding(num_embeddings=len(self.strategy2id.keys()), embedding_dim=graph_dim)
        self.node_position_embedding = nn.Embedding(num_embeddings=6, embedding_dim=graph_dim)
        self.graph_readout_projection = nn.Linear(graph_dim * 3, graph_dim)

        # Classification head
        self.num_classes = len(self.strategy2id) - 1 if args.exclude_others else len(self.strategy2id)
        classifier_hidden_size = self.roberta_config.hidden_size + graph_dim
        self.feature_layernorm = nn.LayerNorm(classifier_hidden_size)
        self.classifier = RobertaClassificationHead(classifier_hidden_size, self.num_classes)
        self.context_classifier = nn.Linear(self.roberta_config.hidden_size, self.num_classes)
        self.graph_classifier = nn.Linear(graph_dim, self.num_classes)
        self.turn_text_projection = nn.Linear(self.roberta_config.hidden_size, graph_dim)
        self.evidence_query = nn.Linear(self.roberta_config.hidden_size, self.roberta_config.hidden_size)
        self.evidence_key = nn.Linear(self.roberta_config.hidden_size, self.roberta_config.hidden_size)
        self.evidence_dropout = nn.Dropout(0.1)
        self.class_prototypes = nn.Parameter(torch.randn((self.num_classes, classifier_hidden_size)))
        self.label_queries = nn.Parameter(torch.randn((self.num_classes, self.label_decoder_dim)))
        self.context_label_projection = nn.Linear(self.roberta_config.hidden_size, self.label_decoder_dim)
        self.graph_label_projection = nn.Linear(graph_dim, self.label_decoder_dim)
        self.fused_label_projection = nn.Linear(classifier_hidden_size, self.label_decoder_dim)
        self.label_decoder_attention = nn.MultiheadAttention(
            embed_dim=self.label_decoder_dim,
            num_heads=getattr(args, "label_decoder_heads", 4),
            dropout=getattr(args, "label_decoder_dropout", 0.1),
            batch_first=True,
        )
        self.label_decoder_norm = nn.LayerNorm(self.label_decoder_dim)
        self.label_decoder_dropout = nn.Dropout(getattr(args, "label_decoder_dropout", 0.1))
        self.label_decoder_scorer = nn.Linear(self.label_decoder_dim, 1)
        self.context_gate_projection = nn.Linear(self.roberta_config.hidden_size, graph_dim)
        self.fusion_gate = nn.Sequential(
            nn.Linear(graph_dim * 4 + 6, getattr(args, "fusion_gate_hidden_dim", 128)),
            nn.ReLU(),
            nn.Dropout(self.fusion_view_dropout),
            nn.Linear(getattr(args, "fusion_gate_hidden_dim", 128), 3),
        )
        self.classwise_fusion_gate = nn.Sequential(
            nn.Linear(graph_dim * 4 + 6, getattr(args, "fusion_gate_hidden_dim", 128)),
            nn.ReLU(),
            nn.Dropout(self.fusion_view_dropout),
            nn.Linear(getattr(args, "fusion_gate_hidden_dim", 128), self.num_classes * 3),
        )
        self.use_granular_uncertainty = bool(
            getattr(args, "use_granular_uncertainty", 0)
            or getattr(args, "granule_loss_weight", 0.0) > 0
            or getattr(args, "granule_logit_weight", 0.0) != 0
        )
        self.granule_logit_weight = getattr(args, "granule_logit_weight", 0.0)
        self.granule_temperature = max(getattr(args, "granule_temperature", 0.2), 1e-6)
        self.class_granules = nn.Parameter(torch.randn((self.num_classes, classifier_hidden_size)))

        # V9+ exploratory modules are instantiated only when explicitly enabled.
        # This keeps the V8 path structurally clean and prevents unused parameters
        # from entering the optimizer or perturbing the reproducibility surface.
        if self.use_turn_label_evidence or self.use_selective_turn_evidence_gate:
            self.turn_label_projection = nn.Linear(self.roberta_config.hidden_size, self.label_decoder_dim)
            self.turn_label_speaker_embedding = nn.Embedding(num_embeddings=2, embedding_dim=self.label_decoder_dim)
            self.turn_label_position_embedding = nn.Embedding(
                num_embeddings=self.turn_label_window_size,
                embedding_dim=self.label_decoder_dim,
            )
            self.turn_label_dropout = nn.Dropout(getattr(args, "turn_label_dropout", 0.1))
            self.turn_evidence_attention = nn.MultiheadAttention(
                embed_dim=self.label_decoder_dim,
                num_heads=getattr(args, "label_decoder_heads", 4),
                dropout=getattr(args, "label_decoder_dropout", 0.1),
                batch_first=True,
            )
            self.turn_evidence_gate = nn.Sequential(
                nn.Linear(self.label_decoder_dim * 4, getattr(args, "turn_evidence_gate_hidden_dim", 128)),
                nn.ReLU(),
                nn.Dropout(getattr(args, "label_decoder_dropout", 0.1)),
                nn.Linear(getattr(args, "turn_evidence_gate_hidden_dim", 128), 1),
            )
        if self.use_global_token_label_evidence:
            self.global_token_label_projection = nn.Linear(self.roberta_config.hidden_size, self.label_decoder_dim)
            self.global_token_label_position_embedding = nn.Embedding(
                num_embeddings=self.global_token_evidence_max_tokens,
                embedding_dim=self.label_decoder_dim,
            )
            self.global_token_label_dropout = nn.Dropout(getattr(args, "global_token_evidence_dropout", 0.1))
        if self.use_hierarchical_decoder:
            self.hierarchy_label_group_ids = self._build_hierarchy_label_group_ids()
            self.num_hierarchy_groups = int(self.hierarchy_label_group_ids.max().item()) + 1
            self.hierarchy_classifier = RobertaClassificationHead(classifier_hidden_size, self.num_hierarchy_groups)
        if self.use_rare_granule_expert:
            self.rare_expert_class_ids = self._parse_rare_expert_classes(getattr(args, "rare_expert_classes", "0,1,3,5,6"))
            self.rare_expert = nn.Sequential(
                nn.LayerNorm(classifier_hidden_size),
                nn.Linear(classifier_hidden_size, getattr(args, "rare_expert_hidden_dim", 256)),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(getattr(args, "rare_expert_hidden_dim", 256), self.num_classes),
            )
        if self.use_dialogue_state_adapter:
            dialogue_state_adapter_dim = getattr(args, "dialogue_state_adapter_dim", 256)
            dialogue_state_adapter_dropout = getattr(args, "dialogue_state_adapter_dropout", 0.1)
            self.dialogue_state_context_projection = nn.Linear(encoder_hidden_size, dialogue_state_adapter_dim)
            self.dialogue_state_graph_projection = nn.Linear(graph_dim, dialogue_state_adapter_dim)
            self.dialogue_state_residual_projection = nn.Linear(dialogue_state_adapter_dim, encoder_hidden_size)
            self.dialogue_state_gate = nn.Sequential(
                nn.Linear(dialogue_state_adapter_dim * 4, dialogue_state_adapter_dim),
                nn.ReLU(),
                nn.Dropout(dialogue_state_adapter_dropout),
                nn.Linear(dialogue_state_adapter_dim, encoder_hidden_size),
            )
            self.dialogue_state_norm = nn.LayerNorm(encoder_hidden_size)
            self.dialogue_state_dropout = nn.Dropout(dialogue_state_adapter_dropout)
        # self.classifier = RobertaClassificationHead(graph_dim, self.num_classes)
        # self.classifier = RobertaClassificationHead(self.roberta_config.hidden_size, self.num_classes)

    def _build_hierarchy_label_group_ids(self):
        if self.hierarchy_group_scheme != "strategy_function_4":
            raise ValueError(f"Unsupported hierarchy_group_scheme: {self.hierarchy_group_scheme}")
        group_by_strategy = {
            "Reflection of feelings": 0,
            "Affirmation and Reassurance": 0,
            "Self-disclosure": 1,
            "Restatement or Paraphrasing": 1,
            "Question": 2,
            "Information": 2,
            "Providing Suggestions": 3,
            "Others": 3,
        }
        group_ids = [group_by_strategy[self.id2strategy[i]] for i in range(self.num_classes)]
        return torch.tensor(group_ids, dtype=torch.long)

    def _parse_rare_expert_classes(self, rare_expert_classes):
        class_ids = []
        for class_id in str(rare_expert_classes).split(","):
            class_id = class_id.strip()
            if not class_id:
                continue
            class_id = int(class_id)
            if class_id < 0 or class_id >= self.num_classes:
                raise ValueError(f"rare_expert_classes contains invalid class id: {class_id}")
            class_ids.append(class_id)
        if not class_ids:
            raise ValueError("rare_expert_classes must contain at least one class id")
        return torch.tensor(sorted(set(class_ids)), dtype=torch.long)

    def _logit_uncertainty_features(self, logits):
        probs = torch.softmax(logits, dim=-1)
        top_probs = torch.topk(probs, k=2, dim=-1).values
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1) / np.log(self.num_classes)
        margin = top_probs[:, 0] - top_probs[:, 1]
        return torch.stack((entropy, margin), dim=-1)

    def _apply_gate_dropout(self, gate_weights):
        if not self.training or self.fusion_view_dropout <= 0:
            return gate_weights
        keep_mask = (torch.rand_like(gate_weights) > self.fusion_view_dropout).float()
        empty_rows = keep_mask.sum(dim=-1, keepdim=True) == 0
        keep_mask = torch.where(empty_rows, torch.ones_like(keep_mask), keep_mask)
        gate_weights = gate_weights * keep_mask
        return gate_weights / gate_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    def _fuse_multiview_logits(self, fused_logits, context_logits, graph_logits, context_embeddings, graph_embeddings):
        fixed_logits = fused_logits + self.context_logit_weight * context_logits + self.graph_logit_weight * graph_logits
        if self.fusion_strategy == "fixed":
            logits = fixed_logits
            return logits, None

        if self.fusion_strategy == "uncertainty_weighted":
            uncertainty_features = torch.stack((
                self._logit_uncertainty_features(context_logits)[:, 0],
                self._logit_uncertainty_features(graph_logits)[:, 0],
                self._logit_uncertainty_features(fused_logits)[:, 0],
            ), dim=-1)
            gate_weights = torch.softmax(-uncertainty_features / self.fusion_gate_temperature, dim=-1)
            gate_weights = self._apply_gate_dropout(gate_weights)
        else:
            context_gate_embeddings = self.context_gate_projection(context_embeddings)
            interaction = torch.cat((
                context_gate_embeddings,
                graph_embeddings,
                torch.abs(context_gate_embeddings - graph_embeddings),
                context_gate_embeddings * graph_embeddings,
                self._logit_uncertainty_features(context_logits),
                self._logit_uncertainty_features(graph_logits),
                self._logit_uncertainty_features(fused_logits),
            ), dim=-1)
            if self.fusion_strategy == "classwise_dynamic_gate":
                gate_logits = self.classwise_fusion_gate(interaction)
                gate_weights = gate_logits.view(-1, self.num_classes, 3)
                gate_weights = torch.softmax(gate_weights / self.fusion_gate_temperature, dim=-1)
            else:
                gate_weights = torch.softmax(self.fusion_gate(interaction) / self.fusion_gate_temperature, dim=-1)
                gate_weights = self._apply_gate_dropout(gate_weights)

        stacked_logits = torch.stack((context_logits, graph_logits, fused_logits), dim=1)
        if gate_weights.dim() == 3:
            logits = (gate_weights * stacked_logits.transpose(1, 2)).sum(dim=-1)
        else:
            logits = (gate_weights.unsqueeze(-1) * stacked_logits).sum(dim=1)
        if self.fusion_fixed_residual_weight > 0:
            residual_weight = self.fusion_fixed_residual_weight
            logits = (1.0 - residual_weight) * logits + residual_weight * fixed_logits
        return logits, gate_weights

    def _build_turn_embeddings(self, dialogue_utterances, dialogue_speakers):
        turn_texts = []
        dialogue_slices = []
        cursor = 0
        for utterances, speakers in zip(dialogue_utterances, dialogue_speakers):
            texts = [f"[{speakers[j]}] {utterances[j]}" for j in range(len(utterances))]
            turn_texts.extend(texts)
            dialogue_slices.append((cursor, cursor + len(texts)))
            cursor += len(texts)
        if not turn_texts:
            return None
        flat_embeddings = self.encode(turn_texts)
        return [flat_embeddings[start:end] for start, end in dialogue_slices]

    def _build_window_evidence_embeddings(self, dialogue_utterances, dialogue_speakers):
        evidence_texts = []
        for utterances, speakers in zip(dialogue_utterances, dialogue_speakers):
            start = max(len(utterances) - self.evidence_window_size, 0)
            window_turns = [
                f"[{speakers[j]}] {utterances[j]}"
                for j in range(start, len(utterances))
            ]
            evidence_texts.append(" ".join(window_turns))
        return self.evidence_dropout(self.encode(evidence_texts))

    def _apply_turn_evidence(self, context_embeddings, turn_embeddings_by_dialogue):
        evidence_embeddings = []
        for batch_idx, turn_embeddings in enumerate(turn_embeddings_by_dialogue):
            evidence_window = turn_embeddings[-self.evidence_window_size:]
            query = self.evidence_query(context_embeddings[batch_idx:batch_idx + 1])
            keys = self.evidence_key(evidence_window)
            scores = torch.matmul(query, keys.t()) / np.sqrt(keys.shape[-1])
            weights = torch.softmax(scores, dim=-1)
            evidence = torch.matmul(weights, evidence_window).squeeze(0)
            evidence_embeddings.append(evidence)
        evidence_embeddings = self.evidence_dropout(torch.stack(evidence_embeddings, dim=0))
        return context_embeddings + self.evidence_fusion_weight * evidence_embeddings, evidence_embeddings

    def _prototype_logits(self, embeddings):
        normalized_embeddings = F.normalize(embeddings, p=2, dim=-1)
        normalized_prototypes = F.normalize(self.class_prototypes, p=2, dim=-1)
        return normalized_embeddings @ normalized_prototypes.t() / self.prototype_temperature

    def _build_turn_label_evidence_tokens(self, dialogue_utterances, dialogue_speakers):
        turn_texts = []
        turn_metadata = []
        for batch_idx, (utterances, speakers) in enumerate(zip(dialogue_utterances, dialogue_speakers)):
            start = max(len(utterances) - self.turn_label_window_size, 0)
            selected_turns = list(range(start, len(utterances)))
            pad_offset = self.turn_label_window_size - len(selected_turns)
            for local_idx, turn_idx in enumerate(selected_turns):
                speaker = speakers[turn_idx] if turn_idx < len(speakers) else "seeker"
                speaker_id = 0 if speaker == "seeker" else 1
                position_id = min(pad_offset + local_idx, self.turn_label_window_size - 1)
                turn_texts.append(f"[{speaker}] {utterances[turn_idx]}")
                turn_metadata.append((batch_idx, position_id, speaker_id))

        batch_size = len(dialogue_utterances)
        turn_tokens = torch.zeros(
            (batch_size, self.turn_label_window_size, self.label_decoder_dim),
            device=self.device,
        )
        key_padding_mask = torch.ones(
            (batch_size, self.turn_label_window_size),
            dtype=torch.bool,
            device=self.device,
        )
        if not turn_texts:
            return turn_tokens, key_padding_mask

        encoded_turns = self.turn_label_projection(self.encode(turn_texts))
        for encoded_idx, (batch_idx, position_id, speaker_id) in enumerate(turn_metadata):
            speaker_tensor = torch.tensor(speaker_id, dtype=torch.long, device=self.device)
            position_tensor = torch.tensor(position_id, dtype=torch.long, device=self.device)
            turn_tokens[batch_idx, position_id, :] = (
                encoded_turns[encoded_idx]
                + self.turn_label_speaker_embedding(speaker_tensor)
                + self.turn_label_position_embedding(position_tensor)
            )
            key_padding_mask[batch_idx, position_id] = False
        return self.turn_label_dropout(turn_tokens), key_padding_mask

    def _build_global_token_label_evidence_tokens(self, token_embeddings, attention_mask):
        batch_tokens = []
        batch_masks = []
        for batch_idx in range(token_embeddings.shape[0]):
            valid_positions = torch.nonzero(attention_mask[batch_idx].bool(), as_tuple=False).squeeze(-1)
            if valid_positions.numel() > 2:
                valid_positions = valid_positions[1:-1]
            else:
                valid_positions = valid_positions.new_empty((0,))
            selected_positions = valid_positions[-self.global_token_evidence_max_tokens:]
            selected_count = selected_positions.numel()
            evidence_tokens = torch.zeros(
                (self.global_token_evidence_max_tokens, self.label_decoder_dim),
                device=self.device,
            )
            evidence_mask = torch.ones(
                (self.global_token_evidence_max_tokens,),
                dtype=torch.bool,
                device=self.device,
            )
            if selected_count > 0:
                projected_tokens = self.global_token_label_projection(token_embeddings[batch_idx, selected_positions, :])
                start = self.global_token_evidence_max_tokens - selected_count
                position_ids = torch.arange(
                    start,
                    self.global_token_evidence_max_tokens,
                    dtype=torch.long,
                    device=self.device,
                )
                projected_tokens = projected_tokens + self.global_token_label_position_embedding(position_ids)
                evidence_tokens[start:, :] = projected_tokens
                evidence_mask[start:] = False
            batch_tokens.append(evidence_tokens)
            batch_masks.append(evidence_mask)
        return (
            self.global_token_label_dropout(torch.stack(batch_tokens, dim=0)),
            torch.stack(batch_masks, dim=0),
        )

    def _label_aware_decoder_logits(
        self,
        context_embeddings,
        graph_embeddings,
        fused_embeddings,
        extra_view_tokens=None,
        extra_key_padding_mask=None,
    ):
        batch_size = context_embeddings.shape[0]
        base_view_tokens = torch.stack((
            self.context_label_projection(context_embeddings),
            self.graph_label_projection(graph_embeddings),
            self.fused_label_projection(fused_embeddings),
        ), dim=1)
        view_tokens = base_view_tokens
        key_padding_mask = None
        use_selective_turn_gate = (
            self.use_selective_turn_evidence_gate
            and extra_view_tokens is not None
        )
        if extra_view_tokens is not None and not use_selective_turn_gate:
            view_tokens = torch.cat((base_view_tokens, extra_view_tokens), dim=1)
            if extra_key_padding_mask is not None:
                base_mask = torch.zeros((batch_size, 3), dtype=torch.bool, device=self.device)
                key_padding_mask = torch.cat((base_mask, extra_key_padding_mask), dim=1)
        label_queries = self.label_queries.unsqueeze(0).expand(batch_size, -1, -1)
        base_decoded_labels, _ = self.label_decoder_attention(
            query=label_queries,
            key=view_tokens,
            value=view_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        decoded_labels = label_queries + self.label_decoder_dropout(base_decoded_labels)
        turn_evidence_gate = None
        if use_selective_turn_gate:
            turn_decoded_labels, _ = self.turn_evidence_attention(
                query=label_queries,
                key=extra_view_tokens,
                value=extra_view_tokens,
                key_padding_mask=extra_key_padding_mask,
                need_weights=False,
            )
            gate_features = torch.cat((
                label_queries,
                base_decoded_labels,
                turn_decoded_labels,
                torch.abs(base_decoded_labels - turn_decoded_labels),
            ), dim=-1)
            turn_evidence_gate = torch.sigmoid(
                self.turn_evidence_gate(gate_features).squeeze(-1) / self.turn_evidence_gate_temperature
            )
            decoded_labels = decoded_labels + (
                self.turn_evidence_residual_weight
                * turn_evidence_gate.unsqueeze(-1)
                * self.label_decoder_dropout(turn_decoded_labels)
            )
        decoded_labels = self.label_decoder_norm(decoded_labels)
        logits = self.label_decoder_scorer(decoded_labels).squeeze(-1)
        return logits, turn_evidence_gate

    def _boundary_gate(self, logits):
        probs = torch.softmax(logits, dim=-1)
        top_probs = torch.topk(probs, k=2, dim=-1).values
        confidence = top_probs[:, 0]
        margin = top_probs[:, 0] - top_probs[:, 1]
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1) / np.log(self.num_classes)
        boundary_score = (entropy + (1.0 - confidence) + (1.0 - margin)) / 3.0
        boundary_gate = torch.sigmoid(
            (boundary_score - self.rare_boundary_threshold) / self.rare_boundary_temperature
        )
        return boundary_gate, boundary_score

    def _rare_granule_residual(self, embeddings, base_logits):
        rare_logits = self.rare_expert(embeddings)
        rare_mask = torch.zeros(self.num_classes, device=self.device, dtype=rare_logits.dtype)
        rare_mask[self.rare_expert_class_ids.to(self.device)] = 1.0
        boundary_gate, boundary_score = self._boundary_gate(base_logits.detach())
        rare_residual = (
            self.rare_expert_weight
            * boundary_gate.unsqueeze(-1)
            * rare_mask.unsqueeze(0)
            * rare_logits
        )
        return rare_residual, rare_logits, boundary_gate, boundary_score

    def _compute_granular_uncertainty(self, embeddings, logits):
        normalized_embeddings = F.normalize(embeddings, p=2, dim=-1)
        normalized_granules = F.normalize(self.class_granules, p=2, dim=-1)
        granule_logits = normalized_embeddings @ normalized_granules.t() / self.granule_temperature
        probs = torch.softmax(logits, dim=-1)
        top_probs = torch.topk(probs, k=2, dim=-1).values
        confidence = top_probs[:, 0]
        probability_margin = top_probs[:, 0] - top_probs[:, 1]
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1) / np.log(self.num_classes)
        granule_probs = torch.softmax(granule_logits, dim=-1)
        top_granule_probs = torch.topk(granule_probs, k=2, dim=-1).values
        granule_margin = top_granule_probs[:, 0] - top_granule_probs[:, 1]
        uncertainty_score = (
            entropy + (1.0 - confidence) + (1.0 - probability_margin) + (1.0 - granule_margin)
        ) / 4.0
        return {
            "granule_logits": granule_logits,
            "confidence": confidence,
            "entropy": entropy,
            "probability_margin": probability_margin,
            "granule_margin": granule_margin,
            "uncertainty_score": uncertainty_score,
        }

    def _readout_graph_embeddings(self, graph_embeddings, dummy_indices, graph_sizes):
        dummy_embeddings = graph_embeddings[dummy_indices, :]
        if self.graph_readout == "dummy":
            return dummy_embeddings

        readouts = []
        for graph_idx, graph_size in enumerate(graph_sizes):
            start = sum(graph_sizes[:graph_idx])
            node_embeddings = graph_embeddings[start + 1:start + graph_size, :]
            if node_embeddings.shape[0] == 0:
                mean_embedding = dummy_embeddings[graph_idx]
                max_embedding = dummy_embeddings[graph_idx]
            else:
                mean_embedding = node_embeddings.mean(dim=0)
                max_embedding = node_embeddings.max(dim=0).values
            readouts.append(torch.cat((dummy_embeddings[graph_idx], mean_embedding, max_embedding), dim=-1))
        return self.graph_readout_projection(torch.stack(readouts, dim=0))

    def _apply_dialogue_state_adapter(self, context_embeddings, graph_embeddings):
        context_state = self.dialogue_state_context_projection(context_embeddings)
        graph_state = self.dialogue_state_graph_projection(graph_embeddings)
        interaction = torch.cat((
            context_state,
            graph_state,
            torch.abs(context_state - graph_state),
            context_state * graph_state,
        ), dim=-1)
        state_gate = torch.sigmoid(self.dialogue_state_gate(interaction))
        state_residual = self.dialogue_state_residual_projection(graph_state)
        enhanced_context = self.dialogue_state_norm(
            context_embeddings
            + self.dialogue_state_adapter_weight
            * state_gate
            * self.dialogue_state_dropout(state_residual)
        )
        return enhanced_context, state_gate

    def forward(self, samples):
        flattened_contexts = []
        dialogues_for_parsing = []
        texts_for_erc = []
        erc_indices = []
        strategy_indices = []
        dialogue_sizes = []
        dialogue_utterances = []
        dialogue_speakers = []
        for i in range(len(samples["dialogue_history"])):
            strategy_history = [int(s.strip()) for s in samples["strategy_history"][i][1:-1].split(",")]
            utterances = samples["dialogue_history"][i].split("</s>")
            speakers = str(samples["speaker_turn"][i]).split(" ")
            dialogue_utterances.append(utterances)
            dialogue_speakers.append(speakers)
            dialogue_sizes.append(len(utterances))
            context = " ".join([f"[{speakers[j]}] {utterances[j]}" for j in range(len(utterances))])
            dialogue_for_parsing = []
            text_for_erc = ""
            erc_index = []
            strategy_index = []
            for j in range(len(utterances)):
                turn = {
                    "speaker": speakers[j],
                    "text": utterances[j]
                }
                dialogue_for_parsing.append(turn)
                text_for_erc += " </s> " + utterances[j]
                if speakers[j] == "seeker":
                    erc_index.append(j)
                else:
                    strategy = strategy_history[j]
                    strategy = strategy if strategy != -1 else 0
                    strategy_index.append((j, strategy))
            strategy_indices.append(strategy_index)
            erc_indices.append(erc_index)
            texts_for_erc.append(text_for_erc)
            dialogues_for_parsing.append(dialogue_for_parsing)
            flattened_contexts.append(context)
        global_token_label_evidence_tokens = None
        global_token_label_evidence_mask = None
        if self.use_global_token_label_evidence:
            context_embeddings, context_token_embeddings, context_attention_mask = self.encode_with_token_embeddings(
                flattened_contexts
            )
            (
                global_token_label_evidence_tokens,
                global_token_label_evidence_mask,
            ) = self._build_global_token_label_evidence_tokens(context_token_embeddings, context_attention_mask)
        else:
            context_embeddings = self.encode(flattened_contexts)
        turn_embeddings_by_dialogue = None
        evidence_embeddings = None
        if self.use_turn_evidence_encoder and self.evidence_encoder_mode == "window_cls":
            evidence_embeddings = self._build_window_evidence_embeddings(dialogue_utterances, dialogue_speakers)
            context_embeddings = context_embeddings + self.evidence_fusion_weight * evidence_embeddings
        if (
            (self.use_turn_evidence_encoder and self.evidence_encoder_mode == "turn_attention")
            or self.use_text_aware_graph_nodes
        ):
            turn_embeddings_by_dialogue = self._build_turn_embeddings(dialogue_utterances, dialogue_speakers)
        if (
            self.use_turn_evidence_encoder
            and self.evidence_encoder_mode == "turn_attention"
            and turn_embeddings_by_dialogue is not None
        ):
            context_embeddings, evidence_embeddings = self._apply_turn_evidence(
                context_embeddings,
                turn_embeddings_by_dialogue,
            )
        # Discourse dependency parsing
        if self.lightmode:
            parsed_dialogues = samples["parsed_dialogue"]
        else:
            parsed_dialogues = self.dialogue_parser.parse(dialogues_for_parsing)
        # Emotion recognition
        erc_input = {
            "texts": texts_for_erc
        }
        if self.lightmode:
            erc_logits = self.softmax(samples["erc_logits"].to(self.device) / (self.t * self.scalar))
        else:
            erc_logits = self.softmax(self.erc(erc_input)["logits"] / (self.t * self.scalar))
        if self.args.erc_mixed:
            erc_embeddings = erc_logits @ self.erc_prototypes
        else:
            erc_tags = torch.argmax(erc_logits, dim=-1)
            erc_embeddings = self.erc_prototypes[erc_tags, :]
        # Build heterogeneous graph
        graphs = []
        graph_inputs = {
            "embeddings": [],
            "edges": [],
            "edge_types": []
        }
        dummy_indices = []
        graph_sizes = []
        for i in range(len(samples["dialogue_history"])):
            nodes = ["DUMMY"] * (dialogue_sizes[i] + 1)
            pos = torch.tensor([dialogue_sizes[i], ] + np.arange(dialogue_sizes[i]).tolist()).to(self.device)
            pos = torch.clamp(pos, max=self.node_position_embedding.num_embeddings - 1)
            pos_embeddings = self.node_position_embedding(pos)
            for j in erc_indices[i]:
                nodes[j + 1] = self.id2emotion[torch.argmax(erc_logits[j + sum(dialogue_sizes[:i]), :], dim=-1).item()]
            for j, sid in strategy_indices[i]:
                nodes[j + 1] = self.id2strategy[sid]
            node_embeddings = torch.zeros((len(nodes), self.args.hg_dim)).to(self.device)
            node_embeddings[0, :] = node_embeddings[0, :] + self.dummy_embedding
            if self.use_text_aware_graph_nodes and turn_embeddings_by_dialogue is not None:
                turn_node_embeddings = self.turn_text_projection(turn_embeddings_by_dialogue[i])
                node_embeddings[1:, :] = node_embeddings[1:, :] + self.text_node_fusion_weight * turn_node_embeddings
            erc_indices_1 = np.array(erc_indices[i]) + 1
            erc_indices_2 = np.array(erc_indices[i]) + sum(dialogue_sizes[:i])
            node_embeddings[erc_indices_1, :] = node_embeddings[erc_indices_1, :] + erc_embeddings[erc_indices_2, :]
            strategy_indices_1 = np.array([s[0] for s in strategy_indices[i]]) + 1
            node_embeddings[strategy_indices_1, :] = node_embeddings[strategy_indices_1, :] + self.strategy_embedding(
                torch.tensor([s[1] for s in strategy_indices[i]]).int().to(self.device))
            if self.use_node_position_embedding:
                node_embeddings = node_embeddings + pos_embeddings
            edges = []
            edge_types = []
            for head, tail, tp in parsed_dialogues[i]:
                if head != 0:
                    edges.append([head, tail])
                    edge_types.append(tp)
            for j in range(1, len(nodes)):
                edges.append([j, 0])
                if j - 1 in erc_indices[i]:
                    edge_types.append(self.graph_relation_dict["Inter"])
                else:
                    edge_types.append(self.graph_relation_dict["Self"])
            graph = {
                "nodes": nodes,
                "edges": edges,
                "edge_types": edge_types,
            }
            graphs.append(graph)
            dummy_indices.append(sum(graph_sizes))
            graph_inputs["embeddings"].append(node_embeddings)
            for head, tail in edges:
                graph_inputs["edges"].append([head + sum(graph_sizes), tail + sum(graph_sizes)])
            graph_inputs["edge_types"].extend(edge_types)
            graph_sizes.append(len(nodes))
        graph_inputs["embeddings"] = torch.cat(graph_inputs["embeddings"], dim=0)
        batch_edges = [[], []]
        for head, tail in graph_inputs["edges"]:
            batch_edges[0].append(head)
            batch_edges[1].append(tail)
        graph_inputs["edges"] = torch.tensor(batch_edges).to(self.device)
        graph_inputs["edge_types"] = torch.tensor(graph_inputs["edge_types"]).to(self.device)
        # Graph Layers
        graph_embeddings, atten_weights_1 = self.conv1(graph_inputs["embeddings"], graph_inputs["edges"],
                                                       graph_inputs["edge_types"])
        graph_embeddings, atten_weights_2 = self.conv2(graph_embeddings, graph_inputs["edges"],
                                                       graph_inputs["edge_types"])
        graph_embeddings, atten_weights_3 = self.conv3(graph_embeddings, graph_inputs["edges"],
                                                       graph_inputs["edge_types"])
        graph_embeddings = self._readout_graph_embeddings(graph_embeddings, dummy_indices, graph_sizes)
        dialogue_state_gate = None
        if self.use_dialogue_state_adapter:
            context_embeddings, dialogue_state_gate = self._apply_dialogue_state_adapter(
                context_embeddings,
                graph_embeddings,
            )
        # Prediction
        embeddings = torch.cat((graph_embeddings, context_embeddings), dim=-1)
        if self.use_feature_layernorm:
            embeddings = self.feature_layernorm(embeddings)
        # embeddings = graph_embeddings
        # embeddings = context_embeddings
        logits = self.classifier(embeddings)
        hierarchy_logits = None
        hierarchy_group_priors = None
        if self.use_hierarchical_decoder:
            hierarchy_logits = self.hierarchy_classifier(embeddings)
            hierarchy_log_probs = F.log_softmax(hierarchy_logits, dim=-1)
            label_group_ids = self.hierarchy_label_group_ids.to(self.device)
            hierarchy_group_priors = hierarchy_log_probs[:, label_group_ids]
            logits = logits + self.hierarchy_logit_weight * hierarchy_group_priors
        label_decoder_logits = None
        turn_label_evidence_mask = None
        turn_evidence_gate = None
        if self.use_label_aware_decoder:
            extra_label_evidence_tokens = []
            extra_label_evidence_masks = []
            if self.use_turn_label_evidence:
                turn_label_evidence_tokens, turn_label_evidence_mask = self._build_turn_label_evidence_tokens(
                    dialogue_utterances,
                    dialogue_speakers,
                )
                extra_label_evidence_tokens.append(turn_label_evidence_tokens)
                extra_label_evidence_masks.append(turn_label_evidence_mask)
            if global_token_label_evidence_tokens is not None:
                extra_label_evidence_tokens.append(global_token_label_evidence_tokens)
                extra_label_evidence_masks.append(global_token_label_evidence_mask)
            label_evidence_tokens = None
            label_evidence_mask = None
            if extra_label_evidence_tokens:
                label_evidence_tokens = torch.cat(extra_label_evidence_tokens, dim=1)
                label_evidence_mask = torch.cat(extra_label_evidence_masks, dim=1)
            label_decoder_logits, turn_evidence_gate = self._label_aware_decoder_logits(
                context_embeddings=context_embeddings,
                graph_embeddings=graph_embeddings,
                fused_embeddings=embeddings,
                extra_view_tokens=label_evidence_tokens,
                extra_key_padding_mask=label_evidence_mask,
            )
            logits = logits + self.label_decoder_logit_weight * label_decoder_logits
        prototype_logits = None
        if self.use_prototype_head:
            prototype_logits = self._prototype_logits(embeddings)
            logits = logits + self.prototype_logit_weight * prototype_logits
        rare_logits = None
        rare_boundary_gate = None
        rare_boundary_score = None
        if self.use_rare_granule_expert:
            rare_residual, rare_logits, rare_boundary_gate, rare_boundary_score = self._rare_granule_residual(
                embeddings=embeddings,
                base_logits=logits,
            )
            logits = logits + rare_residual
        context_logits = None
        graph_logits = None
        fusion_gate_weights = None
        if self.use_multiview_logits:
            context_logits = self.context_classifier(context_embeddings)
            graph_logits = self.graph_classifier(graph_embeddings)
            logits, fusion_gate_weights = self._fuse_multiview_logits(
                fused_logits=logits,
                context_logits=context_logits,
                graph_logits=graph_logits,
                context_embeddings=context_embeddings,
                graph_embeddings=graph_embeddings,
            )
        uncertainty = None
        if self.use_granular_uncertainty:
            uncertainty = self._compute_granular_uncertainty(embeddings, logits)
            logits = logits + self.granule_logit_weight * uncertainty["granule_logits"]
            uncertainty = self._compute_granular_uncertainty(embeddings, logits)
        outputs = {
            "logits": logits,
            "graphs": graphs,
            "attention_weights": [atten_weights_1, atten_weights_2, atten_weights_3],
            "erc_logits": erc_logits,
        }
        if context_logits is not None and graph_logits is not None:
            outputs["context_logits"] = context_logits
            outputs["graph_logits"] = graph_logits
        if prototype_logits is not None:
            outputs["prototype_logits"] = prototype_logits
        if label_decoder_logits is not None:
            outputs["label_decoder_logits"] = label_decoder_logits
        if turn_label_evidence_mask is not None:
            outputs["turn_label_evidence_valid_count"] = (~turn_label_evidence_mask).sum(dim=-1)
        if global_token_label_evidence_mask is not None:
            outputs["global_token_label_evidence_valid_count"] = (~global_token_label_evidence_mask).sum(dim=-1)
        if turn_evidence_gate is not None:
            outputs["turn_evidence_gate"] = turn_evidence_gate
        if hierarchy_logits is not None:
            outputs["hierarchy_logits"] = hierarchy_logits
            outputs["hierarchy_label_group_ids"] = self.hierarchy_label_group_ids.to(self.device)
            outputs["hierarchy_group_priors"] = hierarchy_group_priors
        if rare_logits is not None:
            outputs["rare_logits"] = rare_logits
            outputs["rare_boundary_gate"] = rare_boundary_gate
            outputs["rare_boundary_score"] = rare_boundary_score
            outputs["rare_expert_class_ids"] = self.rare_expert_class_ids.to(self.device)
        if evidence_embeddings is not None:
            outputs["evidence_embeddings"] = evidence_embeddings
        if fusion_gate_weights is not None:
            outputs["fusion_gate_weights"] = fusion_gate_weights
        if dialogue_state_gate is not None:
            outputs["dialogue_state_gate_mean"] = dialogue_state_gate.mean(dim=-1)
        if uncertainty is not None:
            outputs["granule_logits"] = uncertainty["granule_logits"]
            outputs["uncertainty"] = {
                k: v for k, v in uncertainty.items() if k != "granule_logits"
            }
        return outputs
