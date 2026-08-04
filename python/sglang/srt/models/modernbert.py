# python/sglang/srt/models/modernbert.py
# SPDX-License-Identifier: Apache-2.0

from typing import Iterable, Optional, Tuple

import torch
from torch import nn

from sglang.srt.layers.activation import get_act_fn
from sglang.srt.layers.pooler import CrossEncodingPooler, Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.sparse_pooler import SparsePooler
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader

ModernBertConfig = None


class ModernBertEmbeddings(nn.Module):
    """Token embedding + LayerNorm. No position embeddings — RoPE lives in attention."""

    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.tok_embeddings = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.norm = nn.LayerNorm(
            config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.norm(self.tok_embeddings(input_ids))


class ModernBertMLP(nn.Module):
    """GeGLU: Wi projects to 2x intermediate, chunk into (input, gate), act(input)*gate, Wo back down."""

    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.Wi = nn.Linear(
            config.hidden_size, int(config.intermediate_size) * 2, bias=config.mlp_bias
        )
        self.act = get_act_fn(config.hidden_activation)
        self.Wo = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.mlp_bias
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_, gate = self.Wi(hidden_states).chunk(2, dim=-1)
        return self.Wo(self.act(input_) * gate)


class ModernBertAttention(nn.Module):
    """Fused Wqkv, RoPE, RadixAttention with per-layer sliding window.

    layer_id % global_attn_every_n_layers == 0 -> full attention, global theta.
    Everything else -> local attention, local theta, sliding_window_size set.
    """

    def __init__(
        self,
        config: ModernBertConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({config.hidden_size}) not divisible by "
                f"num_attention_heads ({config.num_attention_heads})"
            )

        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.all_head_size = self.num_heads * self.head_dim

        self.Wqkv = nn.Linear(
            config.hidden_size,
            3 * self.all_head_size,
            bias=config.attention_bias,
        )
        self.Wo = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.attention_bias
        )

        is_global = layer_id % config.global_attn_every_n_layers == 0
        rope_theta = (
            config.global_rope_theta if is_global else config.local_rope_theta
        )
        # HF's sliding_window is a half-window (config.local_attention // 2);
        # RadixAttention wants the same convention BertEncoder/gemma2 use —
        # inclusive-to-exclusive off-by-one is the caller's job, not ours here,
        # since we're passing through config's own half-window value unmodified,
        # same as gemma2 passes config.sliding_window straight into RadixAttention.
        sliding_window_size = None if is_global else config.local_attention // 2

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=rope_theta,
            is_neox_style=True,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            scaling=self.head_dim**-0.5,
            num_kv_heads=self.num_heads,
            layer_id=layer_id,
            sliding_window_size=sliding_window_size or -1,
            attn_type=AttentionType.ENCODER_ONLY,
            quant_config=quant_config,
            prefix=prefix,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv = self.Wqkv(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        return self.Wo(attn_output)


class ModernBertEncoderLayer(nn.Module):
    def __init__(
        self,
        config: ModernBertConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        # Layer 0 skips its pre-attention norm — the embedding LayerNorm already
        # normalized the input, mirrors HF's attn_norm = nn.Identity() for layer 0.
        self.attn_norm = (
            nn.Identity()
            if layer_id == 0
            else nn.LayerNorm(
                config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
            )
        )
        self.attn = ModernBertAttention(
            config, layer_id, quant_config=quant_config, prefix=prefix
        )
        self.mlp_norm = nn.LayerNorm(
            config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
        )
        self.mlp = ModernBertMLP(config)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        attn_output = self.attn(positions, self.attn_norm(hidden_states), forward_batch)
        hidden_states = hidden_states + attn_output
        hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))
        return hidden_states


class ModernBertPredictionHead(nn.Module):
    """dense -> act -> norm, sits between pooled hidden state and classifier."""

    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.dense = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.classifier_bias
        )
        self.act = get_act_fn(config.classifier_activation)
        self.norm = nn.LayerNorm(
            config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(self.act(self.dense(hidden_states)))


class ModernBertEncoder(nn.Module):
    def __init__(
        self,
        config: ModernBertConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ModernBertEncoderLayer(config, layer_id, quant_config=quant_config, prefix=prefix)
                for layer_id in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states, forward_batch)
        return hidden_states


class ModernBertBaseModel(nn.Module):
    """Embeddings -> alternating global/local RoPE layers -> final_norm.
    Owns an optional Pooler(CLS/MEAN), same role XLMRobertaBaseModel.pooler plays
    for CrossEncodingPooler's constructor.
    """

    def __init__(
        self,
        *,
        config: ModernBertConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        add_pooling_layer: bool = False,
    ):
        super().__init__()
        self.config = config
        self.embeddings = ModernBertEmbeddings(config)
        self.encoder = ModernBertEncoder(config, quant_config=quant_config, prefix=prefix)
        self.final_norm = nn.LayerNorm(
            config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
        )
        pooling_type = (
            PoolingType.MEAN
            if config.classifier_pooling == "mean"
            else PoolingType.CLS
        )
        self.pooler = (
            Pooler(pooling_type=pooling_type, normalize=True)
            if add_pooling_layer
            else None
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
    ) -> torch.Tensor:
        assert get_embedding == True

        hidden_states = (
            self.embeddings(input_ids) if input_embeds is None else input_embeds
        )
        hidden_states = self.encoder(positions, hidden_states, forward_batch)
        return self.final_norm(hidden_states)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if self.pooler is None and "pooler" in name:
                continue
            # Same choice you flagged for xlmroberta: no continue-on-missing
            # guard here. A bad name KeyErrors immediately instead of
            # silently no-oping, so a checkpoint-prefix mismatch fails loud
            # at load time rather than shipping a half-initialized model.
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)


class ModernBertModel(nn.Module):
    """Dense (+ optional sparse) embedding model. Mirrors XLMRobertaModel's shape."""

    def __init__(
        self,
        *,
        config: ModernBertConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        sparse_head: Optional[str] = None,
        model_path: Optional[str] = None,
    ):
        super().__init__()
        self.model = ModernBertBaseModel(
            config=config, quant_config=quant_config, prefix=prefix
        )
        if sparse_head is not None:
            self._is_sparse = True
            self._model_path = model_path
            self._sparse_head = sparse_head
            self.pooler = SparsePooler(config=config)
            self._special_tokens = [
                t
                for t in (
                    config.bos_token_id,
                    config.eos_token_id,
                    config.pad_token_id,
                    config.cls_token_id,
                    config.sep_token_id,
                )
                if t is not None
            ]
        else:
            self._is_sparse = False
            pooling_type = (
                PoolingType.MEAN
                if config.classifier_pooling == "mean"
                else PoolingType.CLS
            )
            self.pooler = Pooler(pooling_type=pooling_type, normalize=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids, positions, forward_batch, input_embeds, get_embedding
        )
        embeddings = self.pooler(hidden_states, forward_batch)

        if self._is_sparse:
            for token_id in self._special_tokens:
                embeddings.embeddings[:, token_id] = 0.0
            embeddings.embeddings = embeddings.embeddings.to_sparse()

        return embeddings

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        self.model.load_weights(weights)

        if self._is_sparse:
            sparse_dict = ModernBertModel._load_sparse_linear(
                self._model_path, self._sparse_head
            )
            self.pooler.load_weights(sparse_dict)

    @staticmethod
    def _load_sparse_linear(model_path_or_dir: str, sparse_head: str) -> dict:
        """Load sparse_head from local dir or HF Hub. Identical to XLMRobertaModel's —
        same download_from_hf / torch.load path, no ModernBert-specific change needed."""
        import os

        from sglang.srt.utils.hf_transformers_utils import download_from_hf

        if os.path.isdir(model_path_or_dir):
            path = os.path.join(model_path_or_dir, sparse_head)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"'{sparse_head}' not found in {model_path_or_dir}"
                )
        else:
            local_dir = download_from_hf(model_path_or_dir, allow_patterns=sparse_head)
            path = os.path.join(local_dir, sparse_head)

        return torch.load(path)


class ModernBertForSequenceClassification(nn.Module):
    """Rerank / cross-encoder head. Matches gte-reranker-modernbert-base and
    cross-encoder/ettin-reranker-*-v1's checkpoint shape.

    Fix from last pass: CrossEncodingPooler needs self.model.pooler (a real
    Pooler(CLS) instance) as its third arg, not folded into classifier. That's
    what makes CrossEncodingPooler.forward take the CLS-pool-then-classify
    branch instead of classify-per-token-then-stack.
    """

    def __init__(
        self,
        *,
        config: ModernBertConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.model = ModernBertBaseModel(
            config=config,
            quant_config=quant_config,
            prefix=prefix,
            add_pooling_layer=True,
        )
        self.head = ModernBertPredictionHead(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.pooler = CrossEncodingPooler(
            config,
            nn.Sequential(self.head, self.classifier),
            self.model.pooler,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = True,
    ) -> torch.Tensor:
        assert (
            get_embedding
        ), "ModernBertForSequenceClassification is only used for rerank"

        hidden_states = self.model(
            input_ids, positions, forward_batch, input_embeds, get_embedding
        )
        return self.pooler(hidden_states, forward_batch)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        self_weights = []

        def weight_filter():
            for name, weight in weights:
                if name.startswith("model."):
                    yield (name[len("model.") :], weight)
                else:
                    self_weights.append((name, weight))

        self.model.load_weights(weight_filter())

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in self_weights:
            if name.startswith("head") or name.startswith("classifier"):
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = [ModernBertModel, ModernBertForSequenceClassification]
