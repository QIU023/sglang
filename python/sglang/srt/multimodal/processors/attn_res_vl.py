"""Multimodal processor for the Kimi AttnRes VLM (LLaVA-style overlay).

Mirror of :mod:`sglang.srt.multimodal.processors.kimi_vl` but for our
:class:`KimiAttnResVLForConditionalGeneration`. Uses LLaVA conventions
(``<image>`` placeholder + a fixed number of vision-feature slots
matching SigLIP's patch grid) but without needing the HF
``LlavaProcessor`` — our trained image-token id (32000) is *NOT* a
text-level special token in the Llama-3.1 tokenizer (it is a regular
vocab entry that the multimodal dataset bypasses by inserting integer
IDs directly into ``input_ids`` during training). So we tokenize the
text alone, drop the ``<image>`` placeholder, and splice in
``[image_token_id] * num_vision_tokens`` at each placeholder
position manually.

Vision features themselves (the SigLIP+projector outputs) are loaded
on the GPU side via :meth:`KimiAttnResVLForConditionalGeneration.get_image_feature`
during the forward pass; this processor only needs to produce the
correct ``input_ids`` and feed the raw pixel tensors as
``MultimodalDataItem.feature``.
"""
from __future__ import annotations

from io import BytesIO
from typing import Dict, List, Union

import numpy as np
import torch
from PIL import Image

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.attn_res_vl_overlay import (
    KimiAttnResVLForConditionalGeneration,
)
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)


# Number of vision tokens emitted per image by SigLIP-base-patch16-224.
# 224 / 16 = 14 patches per side, 14 × 14 = 196.
_NUM_VISION_TOKENS = 196


def _load_image_to_pil(item: Union[str, bytes, Dict, "Image.Image"]) -> Image.Image:
    """Best-effort conversion of an image input to a PIL Image."""
    if isinstance(item, Image.Image):
        return item
    if isinstance(item, dict):
        # SGLang's image_data dict variants
        for k in ("image", "data", "url", "path", "bytes"):
            if k in item:
                return _load_image_to_pil(item[k])
        raise ValueError(f"unsupported dict image item: keys={list(item.keys())}")
    if isinstance(item, (bytes, bytearray)):
        return Image.open(BytesIO(item)).convert("RGB")
    if isinstance(item, str):
        # Path or URL string
        return Image.open(item).convert("RGB")
    raise ValueError(f"unsupported image item type: {type(item).__name__}")


class KimiAttnResVLImageProcessor(SGLangBaseProcessor):
    models = [KimiAttnResVLForConditionalGeneration]
    gpu_image_decode = False  # PIL/HF SigLIP processor expects host-side images

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.image_token_id: int = int(
            getattr(hf_config, "image_token_id",
                    getattr(hf_config, "media_placeholder_token_id", 32000))
        )
        self.image_token_str: str = "<image>"
        self.num_vision_tokens: int = _NUM_VISION_TOKENS
        # Find the underlying SigLIP image processor on the bundled
        # HF processor (may be a LlavaProcessor whose ``image_processor``
        # attr is a SiglipImageProcessor, or a bare SiglipImageProcessor).
        ip = getattr(_processor, "image_processor", None) or _processor
        self._image_processor = ip
        # Find the tokenizer for text-only tokenization.
        tk = getattr(_processor, "tokenizer", None)
        if tk is None:
            raise RuntimeError(
                "no tokenizer found on the HF processor — cannot tokenize "
                "text inputs for the AttnRes VLM"
            )
        self._tokenizer = tk

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes, Dict]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        # 1. Tokenize text with the <image> placeholder REPLACED by an
        #    empty string. This is robust — the placeholder may
        #    tokenize to multiple subword tokens which would cause
        #    LlavaProcessor's text-level checking to disagree with our
        #    integer-id splicing logic.
        if isinstance(input_text, str):
            prompt_text = input_text
        else:
            # input_text may already be a list[int] of pre-tokenized
            # ids; in that case the caller has already prepared the
            # full sequence and we just route the images through.
            return MultimodalProcessorOutput(
                input_ids=list(input_text),
                mm_items=self._build_mm_items(image_data),
                im_token_id=self.image_token_id,
            )

        # Split the prompt around <image> placeholders.
        chunks = prompt_text.split(self.image_token_str)
        n_images_in_text = len(chunks) - 1
        n_images = len(image_data) if image_data else 0

        # Align: at least one of (text placeholders, image inputs) must
        # exist; if text has no placeholders but images are supplied,
        # prepend one placeholder per image to honor the request.
        if n_images_in_text == 0 and n_images > 0:
            prompt_text = (
                (self.image_token_str + "\n") * n_images + prompt_text
            )
            chunks = prompt_text.split(self.image_token_str)
            n_images_in_text = len(chunks) - 1

        # Tokenize each text chunk separately and splice in image
        # token IDs between them, tracking offsets per image.
        input_ids: list[int] = []
        image_offsets: list[tuple[int, int]] = []
        for i, chunk in enumerate(chunks):
            chunk_ids = self._tokenizer.encode(
                chunk, add_special_tokens=(i == 0)
            )
            input_ids.extend(chunk_ids)
            if i < len(chunks) - 1:
                start = len(input_ids)
                input_ids.extend(
                    [self.image_token_id] * self.num_vision_tokens
                )
                # SGLang's offset convention is inclusive on both ends:
                # token_count = end - start + 1. So for 196 vision
                # tokens, end = start + 195.
                image_offsets.append((start, start + self.num_vision_tokens - 1))

        mm_items = self._build_mm_items(image_data)
        # Bind one offset tuple per image item, in order.
        for item, off in zip(mm_items, image_offsets):
            item.offsets = [off]

        return MultimodalProcessorOutput(
            input_ids=input_ids,
            mm_items=mm_items,
            im_token_id=self.image_token_id,
        )

    def _build_mm_items(
        self, image_data: List[Union[str, bytes, Dict]] | None
    ) -> List[MultimodalDataItem]:
        if not image_data:
            return []
        items: list[MultimodalDataItem] = []
        for img in image_data:
            pil = _load_image_to_pil(img)
            # Run the SigLIP image processor (host-side preprocessing
            # to the standard 224x224 normalised tensor).
            ip_out = self._image_processor(images=pil, return_tensors="pt")
            pixel_values = ip_out.pixel_values  # [1, 3, 224, 224]
            items.append(
                MultimodalDataItem(
                    feature=pixel_values,
                    modality=Modality.IMAGE,
                    # SGLang's mm_inputs path computes the placement
                    # mask by ``input_ids == item.pad_value``, so this
                    # must match the image-token id we splice in.
                    pad_value=self.image_token_id,
                )
            )
        return items
