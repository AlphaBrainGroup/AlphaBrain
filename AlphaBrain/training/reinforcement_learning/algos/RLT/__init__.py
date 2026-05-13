"""Faithful implementation of the RL Token encoder-decoder.

Reference: "RL Token: Bootstrapping Online RL with Vision-Language-Action
Models" (Physical Intelligence, 2026).

This module (``RLT``) is the reference track that follows the reference
construction line-by-line. It sits alongside the sibling ``RLT_a``
module, which is the pragmatic track already shipped in production; the
``README.md`` in this directory explains the differences between the two.
"""

from AlphaBrain.training.reinforcement_learning.algos.RLT.encoder_decoder import (
    RLTokenEncoder,
    RLTokenDecoder,
    RLTokenEncoderDecoder,
)
from AlphaBrain.training.reinforcement_learning.algos.RLT.vla_features import (
    get_vla_hidden_states,
    get_vla_hidden_states_and_action,
    pad_mask_from_attention,
    compact_by_mask,
)

__all__ = [
    "RLTokenEncoder",
    "RLTokenDecoder",
    "RLTokenEncoderDecoder",
    "get_vla_hidden_states",
    "get_vla_hidden_states_and_action",
    "pad_mask_from_attention",
    "compact_by_mask",
]
