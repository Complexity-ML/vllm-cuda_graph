# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright INL Dynamics / Complexity-ML
"""Configuration for Complexity / Pacific-I64 models."""

from transformers import AutoConfig, AutoTokenizer, PretrainedConfig, PreTrainedTokenizer, PreTrainedTokenizerFast


class DeepConfig(PretrainedConfig):
    model_type = "deep"

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=18,
        num_attention_heads=12,
        num_key_value_heads=4,
        intermediate_size=2048,
        vocab_size=32000,
        max_position_embeddings=2048,
        attention_type="gqa",
        attention_dropout=0.0,
        use_qk_norm=True,
        sliding_window=None,
        rope_theta=10000.0,
        rope_type="standard",
        mlp_type="token_routed",
        hidden_act="silu",
        num_experts=4,
        shared_expert=True,
        use_mu_guidance=True,
        use_mu_projection=False,
        disable_mu_guidance=False,
        norm_type="rmsnorm",
        norm_eps=1e-6,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        use_sdpa=True,
        use_cache=True,
        initializer_range=0.02,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.attention_type = attention_type
        self.attention_dropout = attention_dropout
        self.use_qk_norm = use_qk_norm
        self.sliding_window = sliding_window
        self.rope_theta = rope_theta
        self.rope_type = rope_type
        self.mlp_type = mlp_type
        self.hidden_act = hidden_act
        self.num_experts = num_experts
        self.shared_expert = shared_expert
        self.use_mu_guidance = use_mu_guidance
        self.use_mu_projection = use_mu_projection
        self.disable_mu_guidance = disable_mu_guidance
        self.norm_type = norm_type
        self.norm_eps = norm_eps
        self.rms_norm_eps = rms_norm_eps
        self.tie_word_embeddings = tie_word_embeddings
        self.use_sdpa = use_sdpa
        self.use_cache = use_cache
        self.initializer_range = initializer_range
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


# Dummy slow tokenizer — prevents transformers from trying convert_slow_tokenizer
class _DeepSlowTokenizer(PreTrainedTokenizer):
    vocab_files_names: dict[str, str] = {}

    def get_vocab(self):
        return {}

    def _tokenize(self, text):
        raise NotImplementedError("Use the fast tokenizer")

    def save_vocabulary(self, save_directory, filename_prefix=None):
        return ()


# Register with transformers so AutoConfig and AutoTokenizer resolve "deep"
AutoConfig.register("deep", DeepConfig)
AutoTokenizer.register(
    DeepConfig,
    slow_tokenizer_class=_DeepSlowTokenizer,
    fast_tokenizer_class=PreTrainedTokenizerFast,
)
