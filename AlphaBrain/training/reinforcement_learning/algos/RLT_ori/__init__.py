"""Faithful implementation of the RL Token encoder-decoder.

Reference: "RL Token: Bootstrapping Online RL with Vision-Language-Action
Models" (Physical Intelligence, 2026).

This module (``RLT_ori``) is the reference track that follows the reference
construction line-by-line. It sits alongside the sibling ``RLActionToken``
module, which is the pragmatic track already shipped in production; the
``README.md`` in this directory explains the differences between the two.
"""

from AlphaBrain.training.reinforcement_learning.algos.RLT_ori.encoder_decoder import (
    RLTokenEncoder,
    RLTokenDecoder,
    RLTokenEncoderDecoder,
)
from AlphaBrain.training.reinforcement_learning.algos.RLT_ori.vla_features import (
    get_vla_hidden_states,
    pad_mask_from_attention,
)

__all__ = [
    "RLTokenEncoder",
    "RLTokenDecoder",
    "RLTokenEncoderDecoder",
    "get_vla_hidden_states",
    "pad_mask_from_attention",
]
