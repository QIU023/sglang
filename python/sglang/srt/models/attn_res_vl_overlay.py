# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501
"""LLaVA-style multimodal overlay for Block AttnRes (Kimi paper §5).

Wraps :class:`KimiBlockAttnResForCausalLM` (the text-only AttnRes
overlay in :mod:`sglang.srt.models.attn_res_overlay`) with a frozen
SigLIP vision tower + a 2-layer MLP projector, mirroring the LLaVA-1.5
recipe and matching the architecture under
``phase5/multimodal_model.py`` used to train our 447M VLM ckpt.

Architecture (top-level parameters):

  - ``vision_tower``: HF ``transformers.SiglipVisionModel`` instance
    (frozen). Loaded once at __init__ from a path given by the HF
    config (typically ``google/siglip-base-patch16-224``). NOT in
    the model's safetensors — pulled fresh from the HF hub cache.
  - ``mm_projector.projector.{fc1,fc2}.{weight,bias}``: 2-layer MLP
    ``vision_hidden_size → llm_hidden_size → llm_hidden_size`` with
    bias and GELU between layers (matches Projector in
    ``phase5/multimodal_model.py`` exactly).
  - ``language_model``: :class:`KimiBlockAttnResForCausalLM`. Carries
    all the AttnRes residual-overlay logic. Loaded from the
    LM-portion of our DCP→HF safetensors.

Forward path uses SGLang's ``general_mm_embed_routine`` to splice
projected vision features into the LM input embedding sequence at
``image_token_id`` (32000, Llama-3.1 reserved special token reused
as <image>).

Sibling of :class:`KimiVLForConditionalGeneration` in
``sglang/srt/models/kimi_vl.py`` — same structural pattern, with
the LM swapped for our AttnRes overlay and the vision tower
swapped from MoonViT to SigLIP. PR-targeted as a separate model
class so the text-only AttnRes overlay can ship first.
"""
from __future__ import annotations

import logging
import os as _os
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.srt.configs.kimi_attn_res_vl import KimiAttnResVLConfig
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.attn_res_overlay import KimiBlockAttnResForCausalLM
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)

# Env-flag-guarded NaN-hunt instrumentation. Set ATTNRES_NAN_TRACE=1 to
# log shape/min/max/mean/nan/inf at each VL forward stage (vision /
# projector / embed-merge). Mirrors the per-layer LM tracing already in
# attn_res_overlay.py. Off by default — zero cost in production.
_VL_NAN_TRACE = _os.environ.get("ATTNRES_NAN_TRACE", "0") == "1"


def _vl_stats(t, name: str) -> None:
    if not _VL_NAN_TRACE or t is None:
        return
    try:
        tf = t.detach().float()
        logger.warning(
            f"[VL-NaN-TRACE] {name}: shape={tuple(t.shape)} dtype={t.dtype} "
            f"min={tf.min().item():.4f} max={tf.max().item():.4f} "
            f"mean={tf.mean().item():.4f} nan={bool(tf.isnan().any().item())} "
            f"inf={bool(tf.isinf().any().item())}"
        )
    except Exception as e:  # pragma: no cover
        logger.warning(f"[VL-NaN-TRACE] {name}: <stat error: {e}>")


_DEFAULT_IMAGE_TOKEN_ID = 32000  # Llama-3.1 reserved special token, see phase5/multimodal_dataset.py
_DEFAULT_VISION_HIDDEN_SIZE = 768  # google/siglip-base-patch16-224
_DEFAULT_VISION_TOWER = "google/siglip-base-patch16-224"


class KimiAttnResVLProjector(nn.Module):
    """2-layer MLP projector matching the trained ``Projector`` in
    ``phase5/multimodal_model.py``.

    fc1: vision_hidden_size → llm_hidden_size  (bias=True)
    fc2: llm_hidden_size    → llm_hidden_size  (bias=True)
    GELU between.

    The unusual choice of fc2 input == fc1 output == llm_hidden_size
    (rather than the more common ``vision → 4*llm → llm``) follows
    the actual training-side architecture; tweaking it would break
    weight-load compat with our 447M VLM ckpt.
    """

    def __init__(self, vision_hidden_size: int, llm_hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(vision_hidden_size, llm_hidden_size, bias=True)
        self.fc2 = nn.Linear(llm_hidden_size, llm_hidden_size, bias=True)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(vision_features)))


class _MmProjectorBundle(nn.Module):
    """Trivial wrapper so the parameter prefix is ``mm_projector.projector.*``,
    matching the DCP keys our trainer wrote (see
    ``phase5/runs/.../checkpoint/step-2500`` metadata).
    """

    def __init__(self, vision_hidden_size: int, llm_hidden_size: int):
        super().__init__()
        self.projector = KimiAttnResVLProjector(
            vision_hidden_size=vision_hidden_size, llm_hidden_size=llm_hidden_size
        )

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        return self.projector(vision_features)


def _build_frozen_siglip(vision_tower_path: str) -> nn.Module:
    """Load a frozen HF SiglipVisionModel.

    Imported lazily so machines without `transformers` (or for whom
    SigLIP is not available) can still load this module to read the
    class definition, but instantiation requires `transformers`.
    """
    try:
        from transformers import SiglipVisionModel
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "transformers.SiglipVisionModel is required to instantiate "
            "KimiAttnResVLForConditionalGeneration. Install transformers."
        ) from e

    model = SiglipVisionModel.from_pretrained(vision_tower_path)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


class KimiAttnResVLForConditionalGeneration(nn.Module):
    """Top-level VLM model class for SGLang.

    Mirrors :class:`KimiVLForConditionalGeneration` (kimi_vl.py)
    structurally; uses our :class:`KimiBlockAttnResForCausalLM` as
    the language backbone instead of :class:`DeepseekV2ForCausalLM`.

    Config requirements:
      - ``config.text_config``: a :class:`KimiLinearConfig` (or compat),
        passed to the language model constructor.
      - ``config.vision_tower_path``: HF id / local path for SigLIP.
      - ``config.vision_hidden_size``: int, output dim of vision tower.
      - ``config.image_token_id``: int, sentinel id in the LM vocab
        marking image positions (32000 by default, Llama-3.1 reserved).
    """

    # SGLang multimodal dispatch hooks
    @classmethod
    def get_model_config_for_expert_location(cls, config) -> Optional[Any]:  # pragma: no cover
        return None

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config

        # Pull the LM config out. Support either a nested dict-style
        # config (HF transformers convention) or a flat KimiLinearConfig
        # so this works with both: (a) freshly-converted HF safetensors
        # whose config.json names this class as the architecture, and
        # (b) hand-rolled instantiation from torchtitan's flavor registry.
        text_config = getattr(config, "text_config", None) or config
        self.vision_tower_path = getattr(
            config, "vision_tower_path", _DEFAULT_VISION_TOWER
        )
        self.vision_hidden_size = getattr(
            config, "vision_hidden_size", _DEFAULT_VISION_HIDDEN_SIZE
        )
        self.image_token_id = getattr(
            config, "image_token_id", _DEFAULT_IMAGE_TOKEN_ID
        )
        llm_hidden_size = getattr(text_config, "hidden_size")

        # Vision tower: HF SigLIP, frozen. Not in safetensors; loaded
        # fresh from the HF hub cache. 93M params — small enough that
        # we don't need to TP-shard it (vision forward runs once per
        # image at prefill time).
        self.vision_tower = _build_frozen_siglip(self.vision_tower_path)

        # Projector: 2-layer MLP. Trained.
        self.mm_projector = _MmProjectorBundle(
            vision_hidden_size=self.vision_hidden_size,
            llm_hidden_size=llm_hidden_size,
        )

        # Language model: the AttnRes overlay (text-only, already
        # supports inputs_embeds).
        self.language_model = KimiBlockAttnResForCausalLM(
            config=text_config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )
        # general_mm_embed_routine expects ``language_model.get_input_embeddings()``
        # to return the token-embedding module. The upstream
        # KimiLinearForCausalLM (our parent) does not yet implement
        # this hook; bind it inline to avoid a separate upstream patch.
        if not hasattr(self.language_model, "get_input_embeddings"):
            _embed_tokens = self.language_model.model.embed_tokens
            self.language_model.get_input_embeddings = lambda: _embed_tokens
        # Also bind the input-embed kwarg alias the routine uses
        # (``input_embeds`` vs the upstream ``inputs_embeds``).
        if not hasattr(self.language_model, "_lm_forward_orig"):
            self.language_model._lm_forward_orig = self.language_model.forward
            def _forward_with_input_embeds_alias(
                input_ids=None, positions=None, forward_batch=None,
                input_embeds=None, inputs_embeds=None,
                pp_proxy_tensors=None, **kw,
            ):
                if input_embeds is not None and inputs_embeds is None:
                    inputs_embeds = input_embeds
                return self.language_model._lm_forward_orig(
                    input_ids=input_ids, positions=positions,
                    forward_batch=forward_batch, inputs_embeds=inputs_embeds,
                    pp_proxy_tensors=pp_proxy_tensors,
                )
            self.language_model.forward = _forward_with_input_embeds_alias

        self.quant_config = quant_config

    # ------------------------------------------------------------------
    # multimodal hooks expected by SGLang's general_mm_embed_routine
    # ------------------------------------------------------------------

    def get_input_embeddings(self) -> nn.Module:
        """Return the LM's token embedding module so the routine can
        produce the base text-embed sequence. ``general_mm_embed_routine``
        looks this up to embed non-image positions.
        """
        return self.language_model.model.embed_tokens

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """Vision tower → projector. Called once per image batch in
        the prefill phase.

        SGLang batches image features as ``items``, each carrying a
        pre-processed pixel tensor (already normalized & resized by
        the image processor). We concatenate, run vision_tower with
        ``no_grad``, then project. Returns a flat
        ``[total_vision_tokens, llm_hidden_size]`` tensor that
        ``general_mm_embed_routine`` slots into the LM input embeds at
        image-token positions.
        """
        pixel_values = (
            torch.cat([item.feature for item in items], dim=0)
            .to(self.vision_tower.device)
            .to(next(self.vision_tower.parameters()).dtype)
        )

        # SGLang sometimes hands us pre-projected embeddings (e.g. for
        # cached prefills). Detect by shape: if last dim == llm_hidden,
        # assume already projected.
        if (
            pixel_values.dim() == 2
            and pixel_values.shape[-1] == self.config.text_config.hidden_size
        ):
            return pixel_values

        _vl_stats(pixel_values, "vision.pixel_values")
        with torch.no_grad():
            vision_out = self.vision_tower(pixel_values=pixel_values)
            vision_features = vision_out.last_hidden_state  # [B, N_vis, D_vis]
        _vl_stats(vision_features, "vision.last_hidden_state")

        # Cast to projector's dtype (LM is bf16; vision_tower is fp32
        # by default since it was loaded outside our default-dtype
        # context). The projector weights are bf16 per __init__.
        proj_dtype = next(self.mm_projector.parameters()).dtype
        vision_features = vision_features.to(proj_dtype)
        B, N_vis, D_vis = vision_features.shape
        flat = vision_features.reshape(B * N_vis, D_vis)
        projected = self.mm_projector(flat)  # [B*N_vis, D_llm]
        _vl_stats(projected, "projector.output")
        return projected

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        """Our processor already inserts ``[image_token_id] * num_vision_tokens``
        at each ``<image>`` position in input_ids, so SGLang's standard
        post-processing is a no-op for us.

        The standard ``MultiModalityDataPaddingPatternMultimodalTokens``
        pattern expects ``mm_items[i].offsets`` (the positions where
        image tokens should be expanded to N feature slots). We don't
        set offsets — instead we pre-expand at processor time. So just
        return input_ids verbatim.
        """
        return input_ids

    # ------------------------------------------------------------------
    # forward + load
    # ------------------------------------------------------------------

    def _collect_attention_layers(self) -> list:
        """Build the per-layer attention-module list indexed by global
        ``layer_id``, mirroring ``ModelRunner.init_piecewise_cuda_graphs``
        (model_runner.py:~2850).

        Both KDA *and* MLA layers route through ``context.attention_layers``:

        * KDA layers: ``RadixLinearAttention.forward`` takes the
          ``unified_linear_attention_with_output`` fast path when
          ``get_forward_context()`` is non-None, and that helper indexes
          ``context.attention_layers[layer_id]`` to fetch the live KDA
          module (radix_linear_attention.py:119).
        * MLA layers: ``RadixAttention.forward`` ALSO takes the
          ``unified_attention_with_output`` fast path under a non-None
          forward context (radix_attention.py:122), which indexes
          ``context.attention_layers[layer_id]`` and passes the resolved
          module as the ``layer`` arg to
          ``hybrid_linear_attn_backend.forward``.

        The list is consulted *only on the extend/prefill path* —
        ``radix_attention.py:122`` gates the fast path on
        ``forward_batch.forward_mode.is_extend()``; decode takes the
        ``else`` branch and passes ``self`` directly, never touching the
        list. During VLM prefill, ``DeepseekV2AttentionMLA`` runs the
        un-absorbed MHA path (``forward_mha.py`` → ``attn_mha``), NOT the
        absorbed ``attn_mqa`` path. So for MLA layers the list must carry
        ``attn_mha`` — its ``head_dim`` is ``qk_nope+qk_rope`` (the q
        actually produced during prefill). Storing ``attn_mqa`` (head_dim
        ``kv_lora_rank+qk_rope``) makes ``flashinfer_mla_backend`` reshape
        the prefill q with the wrong head_dim and ``RuntimeError`` on the
        ``q.view`` (canonical LM path dodges this because its piecewise
        runner installs the context only for decode, so prefill MLA takes
        the ``else`` branch and the list is never consulted for MLA).

        Therefore the list MUST be (a) non-empty, (b) indexed by the
        *global* layer index, and (c) carry a real ``RadixAttention`` /
        ``RadixLinearAttention`` for EVERY layer — no ``None`` holes.

        Per-layer module pick: ``self_attn.attn`` (KDA
        ``RadixLinearAttention``) if present, else ``self_attn.attn_mha``
        (DeepSeek MLA — the prefill module). Cached after first build.
        """
        cached = getattr(self, "_attnres_attention_layers", None)
        if cached is not None:
            return cached
        layers = []
        lm_model = self.language_model.model
        for idx in range(len(lm_model.layers)):
            layer = lm_model.layers[idx]
            self_attn = getattr(layer, "self_attn", None)
            attn_mod = None
            if self_attn is not None:
                # KDA linear-attn layer.
                attn_mod = getattr(self_attn, "attn", None)
                if attn_mod is None:
                    # DeepSeek-V2 MLA layer: register attn_mha (the
                    # un-absorbed prefill module), NOT attn_mqa. The list
                    # is only indexed on the extend/prefill path, and
                    # prefill runs forward_mha.py -> attn_mha. attn_mqa's
                    # head_dim (kv_lora_rank+qk_rope) would mismatch the
                    # prefill q (qk_nope+qk_rope) and crash the backend's
                    # q.view. Both share layer_id; attn_mha is the one
                    # whose head_dim/v_head_dim match the prefill tensors.
                    attn_mod = getattr(self_attn, "attn_mha", None)
            if attn_mod is None:
                # Should never happen for our Kimi configs (every layer is
                # KDA or MLA). Fail loud rather than leave a None hole that
                # KeyErrors deep in the attention backend.
                raise RuntimeError(
                    f"attn_res_vl_overlay: could not resolve an attention "
                    f"module for layer {idx} (self_attn={type(self_attn).__name__ if self_attn else None}); "
                    f"_collect_attention_layers would leave a None hole that "
                    f"breaks context.attention_layers indexing."
                )
            layers.append(attn_mod)
        self._attnres_attention_layers = layers
        return layers

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
    ):
        # KDA's ``RadixLinearAttention.forward`` branches on
        # ``forward_batch.forward_mode.is_extend() and
        # get_forward_context() is not None``. The global
        # ``_forward_context`` is normally installed by SGLang's
        # piecewise CUDA-graph runner — but that runner is disabled for
        # this multimodal model (``init_piecewise_cuda_graphs`` bails on
        # non-language-model carriers / ``input_embeds`` prefills), so
        # during VLM prefill the context is None. KDA then falls to the
        # ``else`` branch which passes the *un-narrowed* ``mixed_qkv``
        # (still padded to the cuda-graph bucket length) straight to the
        # linear-attn backend, corrupting the KDA recurrent state for
        # every KDA layer. The damage cascades through the AttnRes
        # residual stream and the model emits NaN logits ->
        # ``argmax(NaN)`` -> token id 0 (``!`` in this tokenizer) ->
        # the all-``!!!!`` garbage symptom. This reproduces even on a
        # weight-verified checkpoint, confirming it is a forward-path
        # wiring bug, not a checkpoint problem.
        #
        # Fix: install the forward context ourselves around the embed
        # routine, with a correctly layer-id-indexed attention-layers
        # list so the ``unified_linear_attention_with_output`` custom op
        # narrows ``mixed_qkv``/``a``/``b`` to ``real_num_tokens`` and
        # picks up the right KDA module. ``moe_layers`` / ``moe_fusions``
        # stay empty — they are only consumed by piecewise-graph capture,
        # which is off here, and KimiMoE does its own routing.
        from sglang.srt.compilation.piecewise_context_manager import (
            get_forward_context,
            set_forward_context,
        )

        if (
            forward_batch.forward_mode.is_extend()
            and get_forward_context() is None
        ):
            attention_layers = self._collect_attention_layers()
            with set_forward_context(
                forward_batch=forward_batch,
                attention_layers=attention_layers,
                quant_config=self.quant_config,
                moe_layers=[],
                moe_fusions=[],
            ):
                return general_mm_embed_routine(
                    input_ids=input_ids,
                    forward_batch=forward_batch,
                    language_model=self.language_model,
                    data_embedding_funcs={
                        Modality.IMAGE: self.get_image_feature,
                    },
                    positions=positions,
                )

        return general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            data_embedding_funcs={
                Modality.IMAGE: self.get_image_feature,
            },
            positions=positions,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Three-way routing of incoming HF safetensors keys.

        1. ``vision_tower.*`` and ``vision_model.*`` → frozen, so we
           skip them entirely. We loaded SigLIP from HF in __init__;
           the safetensors don't carry it.
        2. ``mm_projector.projector.fc{1,2}.{weight,bias}`` → load
           directly into our projector via the default loader.
        3. Everything else → delegated to the language model's
           ``load_weights``, with the ``language_model.`` prefix
           stripped (so its internal name lookup matches the LM-only
           HF safetensors keys without a top-level prefix).
        """
        params_dict = dict(self.named_parameters())
        lm_weights: list[Tuple[str, torch.Tensor]] = []

        for entry in weights:
            name = entry[0]
            tensor = entry[1]

            # 1. vision tower — frozen, skip.
            if name.startswith("vision_tower.") or name.startswith("vision_model."):
                continue

            # 2. projector.
            if name.startswith("mm_projector."):
                if name not in params_dict:
                    logger.warning(f"unexpected projector key skipped: {name}")
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, tensor)
                continue

            # 3. language model. Strip an optional ``language_model.``
            # prefix so the LM sees its own keys.
            lm_name = name
            if lm_name.startswith("language_model."):
                lm_name = lm_name[len("language_model.") :]
            lm_weights.append((lm_name, tensor))

        if lm_weights:
            self.language_model.load_weights(iter(lm_weights))


# Self-register the HF config with transformers' AutoConfig so checkpoints
# whose ``model_type`` is ``kimi_attn_res_vl`` can be loaded without
# requiring an upstream-PR-pending edit to SGLang's _CONFIG_REGISTRY.
# Idempotent — silently no-ops on re-import.
try:
    from transformers import AutoConfig as _AutoConfig

    _AutoConfig.register("kimi_attn_res_vl", KimiAttnResVLConfig)
except (ValueError, ImportError):
    # ValueError: already registered. ImportError: no transformers (won't
    # actually serve, but keep the import non-fatal for tooling).
    pass


EntryClass = [KimiAttnResVLForConditionalGeneration]
