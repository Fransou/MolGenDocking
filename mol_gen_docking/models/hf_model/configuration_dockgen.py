from typing import Any, Optional

from transformers.models.qwen3 import Qwen3Config


class DockGenConfig(Qwen3Config):
    model_type = "dockgen"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Default tensor parallel plan for base model `Qwen3`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        prot_embedding_dim: int = 1024,
        mm_token_id: int = 151655,
        vocab_size: int = 151936,
        hidden_size: int = 4096,
        intermediate_size: int = 22016,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 32,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = True,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[float] = None,
        attention_bias: bool = False,
        use_sliding_window: bool = False,
        sliding_window: int = 4096,
        max_window_layers: int = 28,
        layer_types: Optional[str] = None,
        attention_dropout: float = 0.0,
        **kwargs: Any,
    ):
        self.prot_embedding_dim = prot_embedding_dim
        self.mm_token_id = mm_token_id
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            attention_bias=attention_bias,
            use_sliding_window=use_sliding_window,
            sliding_window=sliding_window,
            max_window_layers=max_window_layers,
            layer_types=layer_types,
            attention_dropout=attention_dropout,
            **kwargs,
        )

    @classmethod
    def from_qwen3_config(
        cls,
        qwen3_config: Qwen3Config,
        prot_embedding_dim: int = 1024,
        mm_token_id: int = 151655,
        **kwargs: Any,
    ) -> "DockGenConfig":
        """Create a DockGenConfig from a Qwen3Config."""
        return cls(
            prot_embedding_dim=prot_embedding_dim,
            mm_token_id=mm_token_id,
            vocab_size=qwen3_config.vocab_size,
            hidden_size=qwen3_config.hidden_size,
            intermediate_size=qwen3_config.intermediate_size,
            num_hidden_layers=qwen3_config.num_hidden_layers,
            num_attention_heads=qwen3_config.num_attention_heads,
            num_key_value_heads=qwen3_config.num_key_value_heads,
            head_dim=qwen3_config.head_dim,
            hidden_act=qwen3_config.hidden_act,
            max_position_embeddings=qwen3_config.max_position_embeddings,
            initializer_range=qwen3_config.initializer_range,
            rms_norm_eps=qwen3_config.rms_norm_eps,
            use_cache=qwen3_config.use_cache,
            tie_word_embeddings=qwen3_config.tie_word_embeddings,
            rope_theta=qwen3_config.rope_theta,
            rope_scaling=qwen3_config.rope_scaling,
            attention_bias=qwen3_config.attention_bias,
            use_sliding_window=qwen3_config.use_sliding_window,
            sliding_window=qwen3_config.sliding_window,
            max_window_layers=qwen3_config.max_window_layers,
            layer_types=qwen3_config.layer_types,
            attention_dropout=qwen3_config.attention_dropout,
            **kwargs,
        )
