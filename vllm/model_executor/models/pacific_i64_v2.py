# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright INL Dynamics / Complexity-ML
"""
Pacific-I64 v2 / Complexity model for vLLM inference.

Extends v1 with Sort-and-Split dispatch and Routed GQA:
- RoutedGQA: Q/O routed (E expert weight sets via bmm), K/V shared
- SortSplitMLP: argsort → fixed split N/E → bmm dispatch, zero waste
- Mu-Guidance: cross-layer equilibrium signal

GitHub: https://github.com/Complexity-ML/complexity-framework
"""

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsPP
from .utils import (
    PPMissingLayer,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)


# =============================================================================
# Sort-and-Split routing helper
# =============================================================================


def _routed_proj(
    x: torch.Tensor,
    weight: torch.Tensor,
    sort_idx: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Sort-and-split projection via bmm.

    Args:
        x: [N, dim] flattened input
        weight: [E, dim_in, dim_out] expert weights
        sort_idx: [N] precomputed argsort
        num_experts: number of experts

    Returns:
        [N, dim_out] unsorted output
    """
    N, dim = x.shape
    chunk = N // num_experts
    sorted_x = x[sort_idx]

    # bmm: [E, chunk, dim] @ [E, dim, out] → [E, chunk, out]
    sorted_out = torch.bmm(sorted_x.view(num_experts, chunk, dim), weight).reshape(
        N, -1
    )

    out = torch.zeros(N, sorted_out.shape[-1], device=x.device, dtype=sorted_out.dtype)
    out[sort_idx] = sorted_out
    return out


# =============================================================================
# Sort-Split MLP
# =============================================================================


class SortSplitMLP(nn.Module):
    """Sort-and-Split MoE MLP with bmm dispatch.

    Each expert gets exactly N/E tokens. Gate+Up fused, SiLU activation.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 4,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.expert_intermediate_size = intermediate_size // num_experts

        # Expert weights: [E, hidden, inter*2] and [E, inter, hidden]
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, hidden_size, self.expert_intermediate_size * 2)
        )
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, self.expert_intermediate_size, hidden_size)
        )
        nn.init.kaiming_uniform_(self.gate_up_proj, a=5**0.5)
        nn.init.zeros_(self.down_proj)

    def forward(
        self,
        hidden_states: torch.Tensor,
        sort_idx: torch.Tensor,
    ) -> torch.Tensor:
        N = hidden_states.shape[0]
        chunk = N // self.num_experts
        E = self.num_experts

        sorted_x = hidden_states[sort_idx]

        # bmm gate+up: [E, chunk, hidden] @ [E, hidden, inter*2]
        gu = torch.bmm(sorted_x.view(E, chunk, self.hidden_size), self.gate_up_proj)
        gate, up = gu.chunk(2, dim=-1)
        activated = F.silu(gate) * up

        # bmm down: [E, chunk, inter] @ [E, inter, hidden]
        sorted_out = torch.bmm(activated, self.down_proj).reshape(N, self.hidden_size)

        out = torch.zeros(
            N, self.hidden_size, device=hidden_states.device, dtype=sorted_out.dtype
        )
        out[sort_idx] = sorted_out
        return out


# =============================================================================
# Standard MLP (fallback for dense baseline / run 1)
# =============================================================================


class ComplexityMLP(nn.Module):
    """Standard SwiGLU MLP using vLLM parallel layers."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


# =============================================================================
# Routed GQA Attention
# =============================================================================


class RoutedGQAAttention(nn.Module):
    """
    Routed Grouped Query Attention for vLLM.

    Q and O projections are routed (E expert weight sets via sort-and-split bmm).
    K and V are shared (single weight set) for compatible attention spaces.
    Uses vLLM's PagedAttention for the actual attention computation.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        num_experts: int = 4,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 2048,
        quant_config: QuantizationConfig | None = None,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        tp_size = get_tensor_model_parallel_world_size()

        self.total_num_heads = num_heads
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.head_dim = hidden_size // num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        # Routed Q/O: [E, hidden, q_dim] and [E, q_dim, hidden]
        q_dim = self.num_heads * self.head_dim
        self.q_proj_w = nn.Parameter(torch.empty(num_experts, hidden_size, q_dim))
        self.o_proj_w = nn.Parameter(torch.empty(num_experts, q_dim, hidden_size))
        nn.init.kaiming_uniform_(self.q_proj_w, a=5**0.5)
        nn.init.zeros_(self.o_proj_w)

        # Shared K/V
        self.k_proj = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=num_kv_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.k_proj",
        )
        self.v_proj = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=num_kv_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.v_proj",
        )

        # QK Norm
        self.use_qk_norm = getattr(config, "use_qk_norm", True)
        if self.use_qk_norm:
            rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

        # RoPE
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=max_position_embeddings,
            is_neox_style=True,
            rope_parameters={"base": rope_theta},
        )

        # Attention (uses vLLM PagedAttention / FlashAttention)
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        sort_idx: torch.Tensor,
        mu_prev: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Routed Q projection
        q = _routed_proj(hidden_states, self.q_proj_w, sort_idx, self.num_experts)

        # Shared K/V
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        # QK Norm
        if self.use_qk_norm:
            q = q.view(-1, self.num_heads, self.head_dim)
            k = k.view(-1, self.num_kv_heads, self.head_dim)
            q = self.q_norm(q)
            k = self.k_norm(k)
            q = q.view(-1, self.q_size)
            k = k.view(-1, self.kv_size)

        # RoPE
        q, k = self.rotary_emb(positions, q, k)

        # Attention (PagedAttention via vLLM)
        attn_output = self.attn(q, k, v)

        # Routed O projection
        output = _routed_proj(attn_output, self.o_proj_w, sort_idx, self.num_experts)

        return output


# =============================================================================
# Mu-Guidance
# =============================================================================


class MuGuidance(nn.Module):
    """Mu-Guidance — learnable equilibrium with contextual projection."""

    def __init__(self, hidden_size: int, mu_min: float = 0.0, mu_max: float = 2.0):
        super().__init__()
        self.mu = nn.Parameter(torch.full((hidden_size,), (mu_min + mu_max) / 2))
        self.mu_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.mu_proj.weight)
        self.mu_min = mu_min
        self.mu_max = mu_max

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        mu_clamped = torch.clamp(self.mu, self.mu_min, self.mu_max)
        return mu_clamped + self.mu_proj(hidden_states)


# =============================================================================
# Decoder Layer
# =============================================================================


class ComplexityDecoderLayerV2(nn.Module):
    """Complexity v2 decoder layer: RoutedGQA → Mu-Guidance → SortSplitMLP."""

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        num_experts = getattr(config, "num_experts", 4)

        # Detect attention type
        attn_type = getattr(config, "attention_type", "routed_gqa")
        mlp_type = getattr(config, "mlp_type", "sort_split")

        rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=rms_norm_eps)

        # Attention
        self.use_routed_gqa = attn_type in ("routed_gqa", "sort_split_gqa")
        if self.use_routed_gqa:
            self.self_attn = RoutedGQAAttention(
                config=config,
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=getattr(
                    config, "num_key_value_heads", config.num_attention_heads
                ),
                num_experts=num_experts,
                rope_theta=getattr(config, "rope_theta", 10000.0),
                max_position_embeddings=getattr(
                    config, "max_position_embeddings", 2048
                ),
                quant_config=quant_config,
                cache_config=cache_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            # Fallback to standard GQA (for dense baseline)
            from .pacific_i64 import ComplexityAttention

            self.self_attn = ComplexityAttention(
                config=config,
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=getattr(
                    config, "num_key_value_heads", config.num_attention_heads
                ),
                rope_theta=getattr(config, "rope_theta", 10000.0),
                max_position_embeddings=getattr(
                    config, "max_position_embeddings", 2048
                ),
                quant_config=quant_config,
                cache_config=cache_config,
                prefix=f"{prefix}.self_attn",
            )

        # Mu-Guidance
        self.mu_guidance = MuGuidance(hidden_size=config.hidden_size)

        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=rms_norm_eps)

        # MLP
        self.use_sort_split = mlp_type in ("sort_split", "sort_split_moe")
        if self.use_sort_split:
            self.mlp = SortSplitMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_experts=num_experts,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = ComplexityMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        sort_idx: torch.Tensor | None = None,
        mu_prev: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.use_routed_gqa:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                sort_idx=sort_idx,
                mu_prev=mu_prev,
            )
        else:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                mu_prev=mu_prev,
            )
        hidden_states = residual + hidden_states

        # Mu-Guidance
        mu_current = self.mu_guidance(hidden_states)

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.use_sort_split:
            hidden_states = self.mlp(hidden_states, sort_idx=sort_idx)
        else:
            hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, mu_current


# =============================================================================
# Model
# =============================================================================


@support_torch_compile
class ComplexityModelV2(nn.Module):
    """Complexity v2 transformer: RoutedGQA + SortSplitMLP + Mu-Guidance."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.vocab_size = config.vocab_size
        self.num_experts = getattr(config, "num_experts", 4)

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: ComplexityDecoderLayerV2(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=maybe_prefix(prefix, "layers"),
        )

        rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "mu_prev"],
            config.hidden_size,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_tokens(input_ids)
            mu_prev = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            mu_prev = intermediate_tensors.get("mu_prev")

        # Precompute sort_idx once for all layers
        sort_idx = None
        if self.num_experts > 1 and input_ids is not None:
            N = hidden_states.shape[0]
            expert_ids = input_ids.reshape(-1)[:N] % self.num_experts
            sort_idx = expert_ids.argsort(stable=True)

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            if isinstance(layer, PPMissingLayer):
                continue

            hidden_states, mu_current = layer(
                positions=positions,
                hidden_states=hidden_states,
                sort_idx=sort_idx,
                mu_prev=mu_prev,
            )

            if mu_current is not None:
                mu_prev = mu_current

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {
                    "hidden_states": hidden_states,
                    "mu_prev": mu_prev,
                }
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)


# =============================================================================
# Causal LM
# =============================================================================


class ComplexityV2ForCausalLM(nn.Module, SupportsPP):
    """
    Complexity v2 model for causal language modeling.

    Sort-Split MoE + Routed GQA + Mu-Guidance.
    Compatible with vLLM inference engine.
    """

    packed_modules_mapping = {}

    supported_lora_modules = [
        "k_proj",
        "v_proj",
    ]

    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    embedding_padding_modules = ["lm_head"]

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config

        self.model = ComplexityModelV2(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        if get_pp_group().is_last_rank:
            if getattr(config, "tie_word_embeddings", True):
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )

            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(
                config.vocab_size, scale=logit_scale
            )
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights from complexity-framework checkpoint.

        Handles:
        - 3D routed tensors (q_proj_w, o_proj_w, gate_up_proj, down_proj)
        - Shared K/V (standard 2D)
        - Mu-Guidance weight remapping
        - Tied embeddings
        """
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for ckpt_name, loaded_weight in weights:
            name = ckpt_name
            if not name.startswith("model.") and name != "lm_head.weight":
                name = "model." + name

            # Remap old dynamics keys to mu_guidance
            name = name.replace(".dynamics.mu", ".mu_guidance.mu")
            name = name.replace(".dynamics.mu_proj.", ".mu_guidance.mu_proj.")

            # Skip rotary_emb.inv_freq — vLLM recomputes it
            if "rotary_emb.inv_freq" in name:
                continue

            # Skip token_to_expert — buffer, not parameter
            if "token_to_expert" in name:
                continue

            # Tied embeddings
            if ckpt_name == "lm_head.weight":
                if getattr(self.config, "tie_word_embeddings", True):
                    embed_name = "model.embed_tokens.weight"
                    if embed_name in params_dict:
                        param = params_dict[embed_name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                        loaded_params.add(embed_name)
                else:
                    if name in params_dict:
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                        loaded_params.add(name)
                continue

            if name == "model.embed_tokens.weight" and name in loaded_params:
                continue

            # Standard parameter loading (handles 2D and 3D tensors)
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params


# HuggingFace compatibility aliases
DeepV2ForCausalLM = ComplexityV2ForCausalLM
