# Adapted from: https://github.com/vllm-project/vllm/blob/0384aa7150c4c9778efca041ffd1beb3ad2bd694/vllm/transformers_utils/configs/kimi_linear.py
from transformers.configuration_utils import PretrainedConfig

from sglang.srt.configs.mamba_utils import KimiLinearCacheParams, KimiLinearStateShape


class KimiLinearConfig(PretrainedConfig):
    model_type = "kimi_linear"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        model_type="kimi_linear",
        vocab_size=163840,
        hidden_size=4096,
        head_dim=None,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        rope_theta=10000.0,
        rope_scaling=None,
        tie_word_embeddings=False,
        moe_intermediate_size: int | None = None,
        moe_renormalize: bool = True,
        moe_router_activation_func: str = "sigmoid",
        num_experts: int | None = None,
        num_experts_per_token: int | None = None,
        num_shared_experts: int = 0,
        routed_scaling_factor: float = 1.0,
        first_k_dense_replace: int = 0,
        moe_layer_freq: int = 1,
        use_grouped_topk: bool = True,
        num_expert_group: int = 1,
        topk_group: int = 1,
        q_lora_rank: int | None = None,
        kv_lora_rank: int | None = None,
        qk_nope_head_dim: int | None = None,
        qk_rope_head_dim: int | None = None,
        v_head_dim: int | None = None,
        mla_use_nope: bool | None = False,
        num_nextn_predict_layers: int = 0,
        linear_attn_config: dict | None = None,
        attn_res_enabled: bool = False,
        attn_res_num_blocks: int | None = None,
        moe_score_before_experts: bool = False,
        **kwargs,
    ):
        self.model_type = model_type
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.head_dim = (
            head_dim if head_dim is not None else hidden_size // num_attention_heads
        )
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.mla_use_nope = mla_use_nope
        # moe config
        self.n_routed_experts = self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.moe_renormalize = moe_renormalize
        self.num_shared_experts = num_shared_experts
        self.routed_scaling_factor = routed_scaling_factor
        # When True, the router gate weight is multiplied onto the expert
        # INPUT (before the SwiGLU), then the routed outputs are summed —
        # the torchtitan ``MoE.score_before_experts=True`` convention this
        # checkpoint was TRAINED with. Default False = the standard
        # score-after (weight on expert output) of official Kimi-Linear /
        # DeepSeek-V3 / the HF reference, so stock Kimi-Linear is untouched.
        # Only the AttnRes overlay reads this (sets FusedMoE
        # ``apply_router_weight_on_input``); see attn_res_overlay.py.
        self.moe_score_before_experts = moe_score_before_experts
        self.moe_router_activation_func = moe_router_activation_func
        assert self.moe_router_activation_func in ("softmax", "sigmoid")
        self.moe_intermediate_size = moe_intermediate_size
        self.first_k_dense_replace = first_k_dense_replace
        self.moe_layer_freq = moe_layer_freq
        self.use_grouped_topk = use_grouped_topk
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.num_nextn_predict_layers = num_nextn_predict_layers

        # Block Attention Residual (Kimi paper §5) — formal config fields so
        # the inference overlay validates the block count against the trained
        # checkpoint instead of silently falling back to a hardcoded default.
        # ``attn_res_num_blocks`` is the number of committed blocks the
        # residual stream is partitioned into (N in the paper; S = layers per
        # block = num_hidden_layers // N). Only meaningful when
        # ``attn_res_enabled`` is True.
        self.attn_res_enabled = attn_res_enabled
        self.attn_res_num_blocks = attn_res_num_blocks
        if attn_res_enabled:
            if attn_res_num_blocks is None:
                raise ValueError(
                    "attn_res_enabled=True requires attn_res_num_blocks to be "
                    "set (the trained block count); got None."
                )
            if not (1 <= attn_res_num_blocks <= num_hidden_layers):
                raise ValueError(
                    f"attn_res_num_blocks={attn_res_num_blocks} out of range "
                    f"[1, num_hidden_layers={num_hidden_layers}]."
                )
            if num_hidden_layers % attn_res_num_blocks != 0:
                raise ValueError(
                    f"num_hidden_layers={num_hidden_layers} must be divisible "
                    f"by attn_res_num_blocks={attn_res_num_blocks} (each block "
                    f"must hold an equal number of layers)."
                )

        if linear_attn_config is not None:
            assert linear_attn_config["kda_layers"] is not None
            assert linear_attn_config["full_attn_layers"] is not None
        self.linear_attn_config = linear_attn_config

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def is_mla(self):
        return (
            self.q_lora_rank is not None
            or self.kv_lora_rank is not None
            or self.qk_nope_head_dim is not None
            or self.qk_rope_head_dim is not None
            or self.v_head_dim is not None
            or self.mla_use_nope is True
        )

    @property
    def is_moe(self):
        return self.num_experts is not None

    @property
    def is_linear_attn(self) -> bool:
        return not (
            self.linear_attn_config is None
            or (
                isinstance(self.linear_attn_config, dict)
                and self.linear_attn_config["kda_layers"] is not None
                and len(self.linear_attn_config["kda_layers"]) == 0
            )
        )

    def is_kda_layer(self, layer_idx: int):
        return (
            self.linear_attn_config is not None
            and (layer_idx + 1) in self.linear_attn_config["kda_layers"]
        )

    @property
    def linear_layer_ids(self):
        return [i for i in range(self.num_hidden_layers) if self.is_kda_layer(i)]

    @property
    def full_attention_layer_ids(self):
        return [i for i in range(self.num_hidden_layers) if not self.is_kda_layer(i)]

    @property
    def mamba2_cache_params(self) -> KimiLinearCacheParams:
        from sglang.srt.layers.dp_attention import get_attention_tp_size

        shape = KimiLinearStateShape.create(
            tp_world_size=get_attention_tp_size(),
            num_heads=self.linear_attn_config["num_heads"],
            head_dim=self.linear_attn_config["head_dim"],
            conv_kernel_size=self.linear_attn_config["short_conv_kernel_size"],
        )

        return KimiLinearCacheParams(shape=shape, layers=self.linear_layer_ids)


# ---------------------------------------------------------------------------
# Hybrid linear-attention (KDA) <-> MambaRadixCache wiring.
#
# Kimi-Linear interleaves KDA linear-attention layers with full MLA layers and
# keeps a recurrent SSM state per KDA layer. A plain RadixCache reuses token-id
# prefixes but never checkpoints that recurrent state, so a prefix-cache hit
# feeds KDA a state that was never saved at the prefix boundary -> dirty state
# -> image-blind / inconsistent generations (see kda_backend.forward_extend
# ``has_initial_state``). The fix is to route Kimi-Linear through
# ``MambaRadixCache`` (which snapshots / forks / evicts the SSM state alongside
# the token prefix). That selection is gated by ``Scheduler.is_hybrid_ssm``,
# which becomes True iff a registered ``LinearAttnModelSpec`` for this model
# sets ``uses_mamba_radix_cache=True``.
#
# We register here (module import time, pulled in transitively via
# ``sglang.srt.configs.__init__`` long before ``ServerArgs`` runs its
# per-model adjustments) so both the by-config lookup
# (``get_linear_attn_config`` -> is_hybrid_ssm) and the by-arch lookup
# (``get_linear_attn_spec_by_arch`` -> ServerArgs page_size=1 / overlap-off)
# resolve.
#
# ``unwrap_text_config=True``: the by-config lookup calls
# ``hf_config.get_text_config()`` first, so the multimodal carrier
# ``KimiAttnResVLConfig`` (whose ``text_config`` IS a ``KimiLinearConfig``)
# also matches; for the bare LM configs ``get_text_config()`` returns ``self``.
#
# ``arch_names``: the deployed ``architectures[0]`` strings. The VLM carrier is
# ``KimiAttnResVLForConditionalGeneration``; the text-only overlay /
# upstream LM are ``KimiBlockAttnResForCausalLM`` / ``KimiLinearForCausalLM``.
from sglang.srt.configs.linear_attn_model_registry import (  # noqa: E402
    LinearAttnModelSpec,
    register_linear_attn_model,
)

register_linear_attn_model(
    LinearAttnModelSpec(
        config_class=KimiLinearConfig,
        backend_class_name=(
            "sglang.srt.layers.attention.linear.kda_backend.KDAAttnBackend"
        ),
        arch_names=[
            "KimiAttnResVLForConditionalGeneration",
            "KimiBlockAttnResForCausalLM",
            "KimiLinearForCausalLM",
        ],
        uses_mamba_radix_cache=True,
        support_mamba_cache=True,
        # MambaRadixCache v1 has no extra-buffer support for KDA yet; keep the
        # no_buffer path (page_size=1 + overlap-off), which ServerArgs selects
        # when this is False.
        support_mamba_cache_extra_buffer=False,
        unwrap_text_config=True,
    )
)
