# SPDX-License-Identifier: Apache-2.0
"""Config for the Kimi AttnRes VLM (LLaVA-style: SigLIP + projector + AttnRes LM).

Sibling of :class:`sglang.srt.configs.kimi_vl.KimiVLConfig` —
identical shape, but the text-config is a
:class:`sglang.srt.configs.kimi_linear.KimiLinearConfig` (so the LM
layer count, AttnRes block count, KDA layer schedule, etc. flow
through SGLang's existing kimi_linear loader path) and the vision
side is parametrised by an HF id rather than a dedicated config
(SigLIP is loaded fresh from HF in
``KimiAttnResVLForConditionalGeneration.__init__``).
"""
from __future__ import annotations

from typing import Optional, Union

from transformers.configuration_utils import PretrainedConfig

from sglang.srt.configs.kimi_linear import KimiLinearConfig


class KimiAttnResVLConfig(PretrainedConfig):
    """HF config for the multimodal AttnRes carrier.

    Fields:
        text_config: Config of the LM (KimiLinearConfig). Either an
            already-constructed instance or a kwargs dict.
        vision_tower_path: HF id (or local path) for the SigLIP
            vision tower. Default: ``google/siglip-base-patch16-224``.
        vision_hidden_size: Output dim of the vision tower's
            ``last_hidden_state``. Used to size the projector's fc1.
            Default: 768 (siglip-base-patch16-224).
        image_token_id: Sentinel id in the LM vocab marking
            image-insertion positions. The training dataset and
            inference-side processor must agree on this. Default:
            32000 (Llama-3.1 reserved special token, repurposed by
            phase5/multimodal_dataset.py).
        ignore_index: Loss ignore-index for image / pad tokens.
        media_placeholder_token_id: Alias of image_token_id used by
            SGLang's multimodal padding routines.
    """

    model_type = "kimi_attn_res_vl"

    def __init__(
        self,
        text_config: Optional[Union[dict, KimiLinearConfig]] = None,
        vision_tower_path: str = "google/siglip-base-patch16-224",
        vision_hidden_size: int = 768,
        image_token_id: int = 32000,
        ignore_index: int = -100,
        media_placeholder_token_id: Optional[int] = None,
        pad_token_id: int = 0,
        **kwargs,
    ):
        if text_config is None:
            text_config = KimiLinearConfig()
        elif isinstance(text_config, dict):
            text_config = KimiLinearConfig(**text_config)
        self.text_config = text_config

        self.vision_tower_path = vision_tower_path
        self.vision_hidden_size = vision_hidden_size
        self.image_token_id = image_token_id
        self.ignore_index = ignore_index
        # Both SGLang multimodal helpers and our overlay use the same
        # sentinel; keep the alias for compatibility with the
        # KimiVL-style dispatch path.
        self.media_placeholder_token_id = (
            media_placeholder_token_id
            if media_placeholder_token_id is not None
            else image_token_id
        )

        super().__init__(pad_token_id=pad_token_id, **kwargs)


# Self-register with HF AutoConfig at module-import time. SGLang's
# ``configs/__init__.py`` imports this class, so the registration
# fires during SGLang startup before any ``get_config()`` call —
# which means ``AutoConfig.from_pretrained(...)`` recognises
# ``model_type='kimi_attn_res_vl'`` checkpoints out of the box.
# Idempotent: AutoConfig.register raises ValueError on duplicates,
# which we silently swallow.
try:
    from transformers import AutoConfig as _AutoConfig

    _AutoConfig.register("kimi_attn_res_vl", KimiAttnResVLConfig)
except (ValueError, ImportError):
    pass
