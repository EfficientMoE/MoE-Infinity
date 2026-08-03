"""GLM-5.2 DSA indexer classification utilities.

Classifies per-layer DSA indexer ownership:
  - 'full'   : layer owns (stores/trains) its own indexer weights
  - 'shared' : layer reuses the nearest preceding 'full' layer's indexer
  - 'none'   : layer has no DSA indexer (dense-only layer)

This is used by offload code to avoid double-loading shared indexers.
"""
from __future__ import annotations

from typing import Dict, List, Optional


def get_indexer_types(config) -> List[str]:
    """Return the per-layer indexer_types list from the GLM-5.2 config.

    If ``config.indexer_types`` is present (a list of strings), it is returned
    directly.  Otherwise the list is derived from ``index_topk_freq`` and
    ``first_k_dense_replace``:

    * Layers 0 .. first_k_dense_replace-1 are dense; they are labelled 'none'.
    * Among the remaining (sparse) layers, the first of every ``index_topk_freq``
      consecutive layers is 'full'; the rest are 'shared'.

    Note: in the real GLM-5.2-FP8 config ``indexer_types`` is always present
    and the first ``first_k_dense_replace`` entries are 'full' (not 'none'),
    because those dense layers still participate in DSA indexing.  The fallback
    derivation uses 'none' for dense layers as a conservative default when the
    explicit list is unavailable.
    """
    types = getattr(config, "indexer_types", None)
    if types is not None:
        return list(types)

    # Derive from frequency fields
    n: int = config.num_hidden_layers
    freq: int = getattr(config, "index_topk_freq", 1) or 1
    first_dense: int = getattr(config, "first_k_dense_replace", 0) or 0

    out: List[str] = []
    for i in range(n):
        if i < first_dense:
            out.append("none")  # dense layer — no sparse indexer
        elif (i - first_dense) % freq == 0:
            out.append("full")
        else:
            out.append("shared")
    return out


def owns_indexer(config, layer_id: int) -> bool:
    """Return True iff *layer_id* owns (stores) its own DSA indexer weights.

    A layer owns an indexer when its entry in ``indexer_types`` is ``'full'``.
    """
    types = get_indexer_types(config)
    if layer_id < 0 or layer_id >= len(types):
        return False
    return types[layer_id] == "full"


def indexer_owner_map(config) -> Dict[int, Optional[int]]:
    """Map each layer index to the layer whose indexer it uses.

    Returns a dict ``{layer_id: owner_layer_id}`` where:

    * ``'full'`` layers map to themselves.
    * ``'shared'`` layers map to the nearest preceding ``'full'`` layer
      (or ``None`` if no preceding ``'full'`` layer exists).
    * ``'none'`` / dense layers map to ``None``.
    """
    types = get_indexer_types(config)
    owner: Dict[int, Optional[int]] = {}
    last_full: Optional[int] = None
    for i, t in enumerate(types):
        if t == "full":
            last_full = i
            owner[i] = i
        elif t == "shared":
            owner[i] = last_full  # None if no 'full' seen yet
        else:  # "none" / dense
            owner[i] = None
    return owner


def num_owned_indexers(config) -> int:
    """Return the count of layers that own a DSA indexer (``'full'`` entries)."""
    return sum(1 for t in get_indexer_types(config) if t == "full")
