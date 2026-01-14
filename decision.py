"""
decision.py

Decision layer: chooses the best action (aim + shoot) from a WorldState.

Constraints:
- Deterministic (no randomness by default).
- No image ops here (no cv2). Geometry and color-only simulation only.
- Must prioritize solvability and safety; score maximization is secondary.

The decision value is:
V = alphaSafety + betaStability + gammaFutureGain + deltaScore - epsRisk - zetaAimDifficulty

Default weights (required):
alpha=5.0, beta=1.0, gamma=0.5, delta=0.1, eps=2.0, zeta=1.5
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from world_model import (
    BallState,
    WorldState,
    interpolate_polyline,
    polyline_arc_lengths,
    resolve_cascading_simulation,
    gap_closure_likelihood,
)

Point = Tuple[float, float]


DEFAULT_WEIGHTS: Dict[str, float] = {
    "alpha": 5.0,
    "beta": 1.0,
    "gamma": 0.5,
    "delta": 0.1,
    "eps": 2.0,
    "zeta": 1.5,
}


@dataclass
class DecisionResult:
    """Chosen action for the current frame."""
    target_x: float
    target_y: float
    angle_radians: float
    insert_index: int
    value: float
    feasible: bool
    debug: Dict[str, Any]


def choose_best_action(world: WorldState, weights: Optional[Dict[str, float]] = None) -> Optional[DecisionResult]:
    """
    Choose the best deterministic action based on WorldState.

    Returns None if no feasible action can be found (e.g., missing path or shooter color).
    """
    if world.path_polyline is None or len(world.path_polyline) < 2:
        return None
    if world.shooter_color is None:
        return None
    if world.num_balls < 1:
        return None

    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)

    poly = world.path_polyline
    s_table = polyline_arc_lengths(poly)

    # Extract chain arrays
    balls: List[BallState] = list(world.balls)
    balls = sorted(balls, key=lambda b: (b.s, b.x, b.y, b.ball_id))
    colors = [b.color for b in balls]
    s_positions = [float(b.s) for b in balls]

    # Generate candidate insertion indices: between balls and around group boundaries
    candidates = _candidate_insert_indices(colors)

    frog = (float(world.frog_x), float(world.frog_y))

    best: Optional[DecisionResult] = None
    for idx in candidates:
        action = _evaluate_candidate(world, s_table, s_positions, colors, idx, frog, w)
        if action is None:
            continue
        if best is None:
            best = action
            continue
        # Deterministic tie-breaking: higher value, then smaller aim difficulty, then smaller index
        if action.value > best.value + 1e-9:
            best = action
        elif abs(action.value - best.value) <= 1e-9:
            if action.debug.get("aim_difficulty", 1e9) < best.debug.get("aim_difficulty", 1e9) - 1e-9:
                best = action
            elif abs(action.debug.get("aim_difficulty", 0.0) - best.debug.get("aim_difficulty", 0.0)) <= 1e-9:
                if action.insert_index < best.insert_index:
                    best = action

    return best


def _candidate_insert_indices(colors: Sequence[Optional[str]]) -> List[int]:
    """
    Candidate indices: all gaps plus midpoints of color groups.

    Deterministic ordering: increasing index.
    """
    n = len(colors)
    idxs = set([0, n])
    # All gaps
    for i in range(n - 1):
        idxs.add(i + 1)
    # Group boundaries and centers
    i = 0
    while i < n:
        j = i + 1
        while j < n and colors[j] == colors[i] and colors[i] is not None:
            j += 1
        if j - i >= 2:
            idxs.add((i + j) // 2)
        idxs.add(i)
        idxs.add(j)
        i = j
    return sorted([int(x) for x in idxs if 0 <= x <= n])


def _evaluate_candidate(
    world: WorldState,
    s_table: np.ndarray,
    s_positions: Sequence[float],
    colors: Sequence[Optional[str]],
    insert_index: int,
    frog: Point,
    w: Dict[str, float],
) -> Optional[DecisionResult]:
    """
    Evaluate a single candidate insertion index using geometry + cascade simulation.
    """
    n = len(s_positions)
    if n == 0:
        return None

    # Choose target arc-length:
    # - If inserting between i-1 and i, aim at midpoint of their s.
    # - If inserting at ends, aim slightly inside.
    idx = int(np.clip(insert_index, 0, n))

    if idx == 0:
        s_t = float(s_positions[0] - 0.5 * world.median_spacing)
    elif idx == n:
        s_t = float(s_positions[-1] + 0.5 * world.median_spacing)
    else:
        s_t = float(0.5 * (s_positions[idx - 1] + s_positions[idx]))

    s_t = float(np.clip(s_t, float(s_table[0]), float(s_table[-1])))
    tx, ty = interpolate_polyline(world.path_polyline, s_table, s_t)

    # Aim geometry
    dx, dy = float(tx - frog[0]), float(ty - frog[1])
    dist = float(np.hypot(dx, dy))
    if dist <= 1e-6:
        return None
    angle = float(np.arctan2(dy, dx))

    # Clearance: distance from aim ray to other balls (excluding immediate neighbors)
    clearance = _ray_clearance(frog, (tx, ty), world.balls, exclude_index=idx)
    # Clutter risk: how many different colors nearby around insertion
    clutter = _local_color_clutter(colors, idx, window=4)

    # Feasibility check: require minimal clearance
    feasible = bool(clearance >= 0.08)

    # Color-only simulation
    new_colors, removed = resolve_cascading_simulation(colors, idx, world.shooter_color)

    # Stability: gaps around insertion relative to median spacing
    local_gaps = _local_gaps(s_positions, idx)
    stability = gap_closure_likelihood(local_gaps, world.median_spacing)

    # Future gain: heuristic potential for follow-up matches
    future_gain = _future_match_potential(new_colors)

    # Risk: combine clutter and low clearance
    risk = float(np.clip(0.5 * clutter + (1.0 - clearance), 0.0, 1.0))

    # Aim difficulty: longer distance and lower clearance increases difficulty
    aim_difficulty = float(np.clip((dist / (dist + 300.0)) + (1.0 - clearance), 0.0, 1.0))

    # Safety: must dominate. Combine clearance (main) with "color compatibility" at target.
    color_safety = _color_compatibility(colors, idx, world.shooter_color)
    safety = float(np.clip(0.7 * clearance + 0.3 * color_safety, 0.0, 1.0))

    # Score: removed balls (scaled by delta=0.1 by default)
    score = float(removed)

    # Strong feasibility penalty
    infeasible_penalty = -10.0 if not feasible else 0.0

    value = (
        w["alpha"] * safety
        + w["beta"] * stability
        + w["gamma"] * future_gain
        + w["delta"] * score
        - w["eps"] * risk
        - w["zeta"] * aim_difficulty
        + infeasible_penalty
    )

    debug: Dict[str, Any] = {
        "insert_index": int(idx),
        "s_target": float(s_t),
        "target": [float(tx), float(ty)],
        "dist": float(dist),
        "clearance": float(clearance),
        "clutter": float(clutter),
        "feasible": bool(feasible),
        "removed": int(removed),
        "stability": float(stability),
        "future_gain": float(future_gain),
        "risk": float(risk),
        "aim_difficulty": float(aim_difficulty),
        "safety": float(safety),
        "value": float(value),
    }

    return DecisionResult(
        target_x=float(tx),
        target_y=float(ty),
        angle_radians=float(angle),
        insert_index=int(idx),
        value=float(value),
        feasible=bool(feasible),
        debug=debug,
    )


def _ray_clearance(origin: Point, target: Point, balls: Sequence[BallState], exclude_index: int) -> float:
    """
    Compute normalized clearance of the shot ray to other balls.

    Returns value in [0,1] where 1 is very clear.
    Deterministic.
    """
    ox, oy = float(origin[0]), float(origin[1])
    tx, ty = float(target[0]), float(target[1])
    vx, vy = tx - ox, ty - oy
    v2 = vx * vx + vy * vy
    if v2 <= 1e-9:
        return 0.0

    min_margin = float("inf")
    for i, b in enumerate(balls):
        if abs(i - exclude_index) <= 1:
            continue
        bx, by = float(b.x), float(b.y)
        # Project ball center onto ray segment [origin, target]
        t = ((bx - ox) * vx + (by - oy) * vy) / v2
        t = float(np.clip(t, 0.0, 1.0))
        px, py = ox + t * vx, oy + t * vy
        d = float(np.hypot(bx - px, by - py))
        margin = d - float(max(1.0, b.r))
        if margin < min_margin:
            min_margin = margin

    if min_margin == float("inf"):
        return 1.0

    # Normalize: margin >= 2*median_radius -> 1.0; margin <= 0 -> 0.0
    scale = 40.0
    return float(np.clip(min_margin / scale, 0.0, 1.0))


def _local_color_clutter(colors: Sequence[Optional[str]], idx: int, window: int = 4) -> float:
    """Compute local color clutter risk in [0,1] around insertion index."""
    n = len(colors)
    lo = max(0, idx - window)
    hi = min(n, idx + window)
    subset = [c for c in colors[lo:hi] if c is not None]
    if not subset:
        return 0.0
    uniq = len(set(subset))
    # Normalize: 1 color -> 0, 4+ colors -> 1
    return float(np.clip((uniq - 1) / 3.0, 0.0, 1.0))


def _local_gaps(s_positions: Sequence[float], idx: int) -> List[float]:
    """Return gaps around insertion index for stability estimation."""
    n = len(s_positions)
    gaps: List[float] = []
    if n < 2:
        return gaps
    if 0 < idx < n:
        gaps.append(float(s_positions[idx] - s_positions[idx - 1]))
    if 1 < idx < n:
        gaps.append(float(s_positions[idx - 1] - s_positions[idx - 2]))
    if 0 < idx < n - 1:
        gaps.append(float(s_positions[idx + 1] - s_positions[idx]))
    return gaps


def _future_match_potential(colors: Sequence[Optional[str]]) -> float:
    """Heuristic future gain in [0,1]: number of pairs that could become triples soon."""
    if not colors:
        return 0.0
    pot = 0
    for i in range(len(colors) - 1):
        if colors[i] is not None and colors[i] == colors[i + 1]:
            pot += 1
    # Normalize by length
    return float(np.clip(pot / max(1.0, 10.0), 0.0, 1.0))


def _color_compatibility(colors: Sequence[Optional[str]], idx: int, shot_color: Optional[str]) -> float:
    """
    Compatibility of inserting shot_color at idx based on neighbors.
    Returns in [0,1].
    """
    if shot_color is None:
        return 0.0
    n = len(colors)
    left = colors[idx - 1] if idx - 1 >= 0 and idx - 1 < n else None
    right = colors[idx] if idx >= 0 and idx < n else None
    if left == shot_color and right == shot_color:
        return 1.0
    if left == shot_color or right == shot_color:
        return 0.7
    # Riskier insertion into mixed colors
    return 0.2
