"""Integer capacity allocation utilities for fixed-budget OASRS variants."""

from __future__ import annotations

from typing import Dict, Mapping, TypeVar

K = TypeVar("K")


def allocate_capacities(
    budget: int,
    weights: Mapping[K, float],
    minimum: int = 1,
    maximums: Mapping[K, int] | None = None,
) -> Dict[K, int]:
    """Allocate an integer budget by Hamilton's largest-remainder method.

    Every key gets ``minimum`` capacity. Remaining capacity is split by
    non-negative weights, with optional finite-population upper bounds.
    The result sums to ``min(budget, sum(maximums))`` when capped.
    """
    if not isinstance(budget, int) or budget < 0:
        raise ValueError("budget must be a non-negative integer")
    if not isinstance(minimum, int) or minimum < 0:
        raise ValueError("minimum must be a non-negative integer")
    keys = list(weights)
    if not keys:
        return {}
    if budget < minimum * len(keys):
        raise ValueError("budget is too small to satisfy minimum per active stratum")
    caps = {key: (maximums or {}).get(key, budget) for key in keys}
    if any(cap < minimum for cap in caps.values()):
        raise ValueError("maximum capacity cannot be below minimum")
    target = min(budget, sum(caps.values()))
    allocation = {key: minimum for key in keys}
    remaining = target - sum(allocation.values())
    clean = {key: max(0.0, float(weights[key])) for key in keys}

    while remaining:
        eligible = [key for key in keys if allocation[key] < caps[key]]
        if not eligible:
            break
        total_weight = sum(clean[key] for key in eligible)
        if total_weight == 0:
            shares = {key: remaining / len(eligible) for key in eligible}
        else:
            shares = {key: remaining * clean[key] / total_weight for key in eligible}
        floors = {key: min(caps[key] - allocation[key], int(shares[key])) for key in eligible}
        used = sum(floors.values())
        for key, amount in floors.items():
            allocation[key] += amount
        remaining -= used
        if remaining == 0:
            break
        order = sorted(eligible, key=lambda key: (shares[key] - int(shares[key]), str(key)), reverse=True)
        progressed = False
        for key in order:
            if remaining == 0:
                break
            if allocation[key] < caps[key]:
                allocation[key] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return allocation
