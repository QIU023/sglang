"""Compatibility shim (fork attention_residual_inference, 2026-05-31).

Upstream PR #24450 moved the routed-experts capturer out of
``sglang.srt.layers.moe.routed_experts_capturer`` into
``sglang.srt.state_capturer.*``, and renamed ``RoutedExpertsOutput`` to
``TopkCaptureOutput`` (state_capturer/base.py). The fork's model_runner.py
still imports the old path/name (used only as a type annotation). Re-export
the new symbols so both old and new import sites work.
"""
from sglang.srt.state_capturer.routed_experts import (  # noqa: F401
    RoutedExpertsCapturer,
    get_global_experts_capturer,
    set_global_experts_capturer,
)
# RoutedExpertsOutput was renamed to TopkCaptureOutput.
from sglang.srt.state_capturer.base import (  # noqa: F401
    TopkCaptureOutput as RoutedExpertsOutput,
)
