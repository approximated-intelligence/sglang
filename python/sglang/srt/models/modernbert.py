# python/sglang/srt/models/modernbert.py
# SPDX-License-Identifier: Apache-2.0
"""
ModernBERT implementation for SGLang
Supports dense and sparse embeddings.
"""

from typing import Iterable, Optional

import torch
from torch import nn

from sglang.srt.layers.activation import get_act_fn
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.layers.sparse_pooler import SparsePooler
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils.hf_transformers_utils import download_from_hf

ModernBertConfig = None


# ---------------------------------------------------------------------------
# Pure PyTorch RoPE – no vllm or CUDA dependency, works on CPU
# ---------------------------------------------------------------------------
class ModernBertRotaryEmbedding(nn.Module):
    """RoPE that matches ModernBERT’s Neox‑style rotation."""

    def __init__(self, dim: int, max_position: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position = max_position
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor):
        """Apply RoPE to q and k. positions: [num_tokens]."""
        freqs = torch.outer(positions.float(), self.inv_freq)  # [T, dim//2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [T, dim]
        cos = emb.cos().to(q.dtype)
        sin = emb.sin().to(q.dtype)

        def rotate_half(x):
            x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        cos = cos.unsqueeze(1)  # [T, 1, dim]
        sin = sin.unsqueeze(1)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------
class ModernBertEmbeddings(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.tok_embeddings = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.norm = nn.LayerNorm(
            config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
        )
        self.dropout = nn.Dropout(getattr(config, "embedding_dropout", 0.0))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.norm(self.tok_embeddings(input_ids)))


class ModernBertAttention(nn.Module):
    def __init__(
        self,
        config: ModernBertConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.all_head_size = self.num_heads * self.head_dim

        self.Wqkv = nn.Linear(
            config.hidden_size, 3 * self.all_head_size, bias=config.attention_bias
        )
        self.Wo = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.attention_bias
        )
        self.dropout = nn.Dropout(config.attention_dropout)

        # Determine layer type
        if getattr(config, "layer_types", None) is not None:
            is_global = config.layer_types[layer_id] == "full_attention"
        else:
            is_global = layer_id % config.global_attn_every_n_layers == 0

        # RoPE theta
        if getattr(config, "rope_parameters", None) is not None:
            key = "full_attention" if is_global else "sliding_attention"
            rope_theta = config.rope_parameters[key]["rope_theta"]
        else:
            rope_theta = (
                config.global_rope_theta if is_global else config.local_rope_theta
            )

        # Pure PyTorch RoPE – no vllm needed
        self.rotary_emb = ModernBertRotaryEmbedding(
            self.head_dim,
            max_position=config.max_position_embeddings,
            base=rope_theta,
        )

        sliding_window_size = None if is_global else (config.local_attention // 2)

        self.attn = RadixAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scaling=self.head_dim**-0.5,
            num_kv_heads=self.num_heads,
            layer_id=layer_id,
            sliding_window_size=sliding_window_size or -1,
            attn_type=AttentionType.ENCODER_ONLY,
            pos_encoding_mode="NONE",  # we apply RoPE ourselves
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
        # Reshape for RoPE: [T, num_heads, head_dim]
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_heads, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        q = q.reshape(-1, self.num_heads * self.head_dim)
        k = k.reshape(-1, self.num_heads * self.head_dim)

        attn_output = self.attn(q, k, v, forward_batch)
        return self.dropout(self.Wo(attn_output))


class ModernBertMLP(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.Wi = nn.Linear(
            config.hidden_size, int(config.intermediate_size) * 2, bias=config.mlp_bias
        )
        self.Wo = nn.Linear(
            int(config.intermediate_size), config.hidden_size, bias=config.mlp_bias
        )
        self.act = get_act_fn(config.hidden_activation)
        self.dropout = nn.Dropout(config.mlp_dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_, gate = self.Wi(hidden_states).chunk(2, dim=-1)
        return self.dropout(self.Wo(self.act(input_) * gate))


class ModernBertEncoderLayer(nn.Module):
    def __init__(
        self,
        config: ModernBertConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        # Layer 0 already has embedding norm -> skip attention norm
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
        hidden_states = hidden_states + self.attn(
            positions, self.attn_norm(hidden_states), forward_batch
        )
        hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))
        return hidden_states


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
                ModernBertEncoderLayer(
                    config, i, quant_config=quant_config, prefix=prefix
                )
                for i in range(config.num_hidden_layers)
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
    def __init__(
        self,
        *,
        config: ModernBertConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.embeddings = ModernBertEmbeddings(config)
        self.encoder = ModernBertEncoder(
            config, quant_config=quant_config, prefix=prefix
        )
        self.final_norm = nn.LayerNorm(
            config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
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
        assert get_embedding
        hidden_states = (
            self.embeddings(input_ids) if input_embeds is None else input_embeds
        )
        return self.final_norm(self.encoder(positions, hidden_states, forward_batch))

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if name.startswith("model."):
                name = name[6:]
            if name.startswith("layers."):
                name = "encoder." + name
            if name.startswith(("head.", "classifier.", "decoder.")):
                continue
            if (param := params_dict.get(name)) is not None:
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


class ModernBertForMaskedLM(nn.Module):
    """ModernBERT for MaskedLM, repurposed as embedding model.
    Supports dense and sparse (SparsePooler) pooling.
    """

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
            self._sparse_head = sparse_head
            self._model_path = model_path
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
                if getattr(config, "classifier_pooling", "cls") == "mean"
                else PoolingType.CLS
            )
            self.pooler = Pooler(pooling_type=pooling_type, normalize=True)

    @torch.no_grad()
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

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        self.model.load_weights(weights)
        if self._is_sparse:
            sparse_dict = self._load_sparse_linear(self._model_path, self._sparse_head)
            self.pooler.load_weights(sparse_dict)

    @staticmethod
    def _load_sparse_linear(model_path_or_dir: str, sparse_head: str) -> dict:
        import os

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


class ModernBertModel(ModernBertForMaskedLM):
    """ModernBertModel architecture – identical to ModernBertForMaskedLM for embedding use."""

    pass


EntryClass = [ModernBertModel, ModernBertForMaskedLM]
