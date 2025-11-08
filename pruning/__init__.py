"""Pruning methods package"""

from .magnitude_pruning import (
    magnitude_prune_global,
    magnitude_prune_layerwise,
    get_pruning_mask,
    apply_pruning_mask
)
from .structured_pruning import (
    structured_prune_filters,
    structured_prune_channels,
    structured_prune_combined
)
from .lottery_ticket import (
    LotteryTicketPruner,
    lottery_ticket_prune,
    iterative_magnitude_pruning
)

__all__ = [
    'magnitude_prune_global',
    'magnitude_prune_layerwise',
    'get_pruning_mask',
    'apply_pruning_mask',
    'structured_prune_filters',
    'structured_prune_channels',
    'structured_prune_combined',
    'LotteryTicketPruner',
    'lottery_ticket_prune',
    'iterative_magnitude_pruning'
]
