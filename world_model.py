"""
World Model layer: converts Perception outputs into a persistent, symbolic WorldState.

This file is intentionally pixel-free (NO cv2 / NO image processing).
It only operates on the dict returned by perception.py:

perception_output = {
  "frog": {"x": int, "y": int},
  "path": {"points": [(x,y), ...], "end_point": (xe, ye)},
  "balls": [
     {"id","x","y","r","color","order","distance_to_end","danger"}, ...
  ]
}

Key goal of this rewrite:
- Make world_model directly compatible with perception outputs.
- Avoid "strict" behaviors that effectively block playing (e.g., missing shooter_color).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]


@dataclass
class BallState:
    """A single chain ball state projected onto the path polyline."""
    ball_id: int
    x: int
    y: int
    r: int
    s: float  # arc-length coordinate along the path
    color: Optional[str]
    color_conf: float
    frames_seen: int


@dataclass
class WorldState:
    """
    Symbolic game state (no pixels).

    Kept compatible with decision.py (do not remove/rename fields used there).
    """
    frame_index: int
    frog_x: float
    frog_y: float
    shooter_color: Optional[str]
    next_color: Optional[str]
    num_balls: int
    median_radius: float
    median_spacing: float
    chain_velocity: float
    path_num_points: int
    balls: List[BallState]
    path_polyline: Optional[List[Point]]


# -----------------------
# Persistent memory
# -----------------------
WORLD_MEMORY: Dict[str, Any] = {
    "next_id": 1,
    "balls": {},  # id -> dict(s, frame_index, color)
    "last_frame_index": -10**9,
    "velocity": 0.0,
    "median_spacing": 0.0,
    # to avoid stalling when Perception doesn't provide shooter/next
    "shooter_color": None,
    "next_color": None,
}

# "Strict" tracking can be fragile and is NOT required for playing.
# Keep it disabled until the pipeline is stable.
ENABLE_TRACKING: bool = False


def reset_world() -> None:
    """Reset persistent world memory."""
    WORLD_MEMORY["next_id"] = 1
    WORLD_MEMORY["balls"] = {}
    WORLD_MEMORY["last_frame_index"] = -10**9
    WORLD_MEMORY["velocity"] = 0.0
    WORLD_MEMORY["median_spacing"] = 0.0
    WORLD_MEMORY["shooter_color"] = None
    WORLD_MEMORY["next_color"] = None


def polyline_arc_lengths(poly: Sequence[Point]) -> np.ndarray:
    """Compute cumulative arc lengths for a polyline (deterministic)."""
    if len(poly) < 2:
        return np.array([0.0], dtype=np.float64)
    pts = np.array(poly, dtype=np.float64)
    d = np.sqrt(np.sum((pts[1:] - pts[:-1]) ** 2, axis=1))
    s = np.concatenate([[0.0], np.cumsum(d)])
    return s


def interpolate_polyline(poly: Sequence[Point], s_table: np.ndarray, s_target: float) -> Point:
    """Interpolate a point along the polyline at arc-length coordinate s_target."""
    if len(poly) == 0:
        return (0.0, 0.0)
    if len(poly) == 1 or s_table.size == 1:
        return (float(poly[0][0]), float(poly[0][1]))

    s_target = float(np.clip(s_target, float(s_table[0]), float(s_table[-1])))
    idx = int(np.searchsorted(s_table, s_target, side="right") - 1)
    idx = int(np.clip(idx, 0, len(poly) - 2))
    s0 = float(s_table[idx])
    s1 = float(s_table[idx + 1])
    p0 = poly[idx]
    p1 = poly[idx + 1]
    if abs(s1 - s0) < 1e-9:
        return (float(p0[0]), float(p0[1]))
    t = (s_target - s0) / (s1 - s0)
    x = (1.0 - t) * float(p0[0]) + t * float(p1[0])
    y = (1.0 - t) * float(p0[1]) + t * float(p1[1])
    return (x, y)


def project_point_on_polyline(point: Point, poly: Sequence[Point]) -> Tuple[float, Point]:
    """
    Project a point onto a polyline and return (s, closest_point).

    s is the arc-length coordinate from the start of the polyline.
    Deterministic and robust for short polylines.
    """
    px, py = float(point[0]), float(point[1])
    if len(poly) == 0:
        return 0.0, (px, py)
    if len(poly) == 1:
        return 0.0, (float(poly[0][0]), float(poly[0][1]))

    s_table = polyline_arc_lengths(poly)
    best_dist2 = float("inf")
    best_s = 0.0
    best_p = (float(poly[0][0]), float(poly[0][1]))

    for i in range(len(poly) - 1):
        x0, y0 = float(poly[i][0]), float(poly[i][1])
        x1, y1 = float(poly[i + 1][0]), float(poly[i + 1][1])
        vx, vy = x1 - x0, y1 - y0
        denom = vx * vx + vy * vy
        if denom <= 1e-12:
            t = 0.0
        else:
            t = ((px - x0) * vx + (py - y0) * vy) / denom
        t = float(np.clip(t, 0.0, 1.0))
        cx = x0 + t * vx
        cy = y0 + t * vy
        dist2 = (px - cx) ** 2 + (py - cy) ** 2
        if dist2 < best_dist2:
            best_dist2 = dist2
            seg_len = float(s_table[i + 1] - s_table[i])
            best_s = float(s_table[i] + t * seg_len)
            best_p = (float(cx), float(cy))

    return best_s, best_p


def robust_median_spacing(s_positions: Sequence[float], median_radius: float) -> float:
    """Compute robust median spacing (ignore outliers)."""
    if len(s_positions) < 2:
        return float(max(1.0, 2.0 * median_radius))
    s_sorted = np.sort(np.array(s_positions, dtype=np.float64))
    gaps = np.diff(s_sorted)
    if gaps.size == 0:
        return float(max(1.0, 2.0 * median_radius))
    med = float(np.median(gaps))
    mad = float(np.median(np.abs(gaps - med))) + 1e-6
    good = gaps[np.abs(gaps - med) <= max(3.0 * mad, 0.35 * med)]
    if good.size == 0:
        return float(max(1.0, med))
    return float(np.median(good))


def gap_closure_likelihood(local_gaps: Sequence[float], median_spacing: float) -> float:
    """
    Deterministic gap-closure likelihood metric derived from median_spacing and local gaps.
    Returns in [0,1].
    """
    if median_spacing <= 1e-6:
        return 0.0
    if not local_gaps:
        return 0.5
    g = float(np.median(np.array(local_gaps, dtype=np.float64)))
    dev = abs(g - median_spacing) / median_spacing
    return float(np.clip(1.0 - dev, 0.0, 1.0))


def _as_point_list(points: Any) -> Optional[List[Point]]:
    """Convert to list[(float,float)] or return None."""
    if not points:
        return None
    out: List[Point] = []
    try:
        for p in points:
            out.append((float(p[0]), float(p[1])))
    except Exception:
        return None
    return out if len(out) >= 2 else None


def _guess_shooter_color_from_chain(chain_colors: Sequence[Optional[str]]) -> Optional[str]:
    """
    Deterministic fallback when perception doesn't provide shooter_color.
    Choose the most frequent color in chain (mode), tie-break lexicographically.
    """
    vals = [c for c in chain_colors if c is not None]
    if not vals:
        return None
    counts: Dict[str, int] = {}
    for c in vals:
        counts[str(c)] = counts.get(str(c), 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _extract_perception_fields(
    perception_output: Dict[str, Any],
) -> Tuple[Tuple[float, float], Optional[List[Point]], List[Dict[str, Any]], Optional[str], Optional[str]]:
    """
    Extract (frog_center, polyline, balls_dicts, shooter_color, next_color)
    from the perception schema.
    """
    frog = perception_output.get("frog") or {}
    frog_center = (float(frog.get("x", 0.0)), float(frog.get("y", 0.0)))

    path = perception_output.get("path") or {}
    polyline = _as_point_list(path.get("points"))

    balls = perception_output.get("balls") or []
    if not isinstance(balls, list):
        balls = []

    # Optional (future): if you add these keys in perception later, we will use them.
    shooter_color = None
    next_color = None

    shooter = perception_output.get("shooter")
    if isinstance(shooter, dict):
        shooter_color = shooter.get("color")

    nxt = perception_output.get("next")
    if isinstance(nxt, dict):
        next_color = nxt.get("color")
    frog_ball = perception_output.get("frog_ball")
    if isinstance(frog_ball, dict):
        shooter_color_p = frog_ball.get("color")


    return frog_center, polyline, balls, shooter_color, next_color


def build_world_state(
    frame_index: int,
    perception_or_polyline: Any,
    chain_balls_legacy: Optional[Any] = None,
    frog_center_legacy: Optional[Tuple[float, float]] = None,
    shooter_color_override: Optional[str] = None,
    next_color_override: Optional[str] = None,
) -> WorldState:
    """
    Build a WorldState and update WORLD_MEMORY.

    ✅ New (recommended):
        ws = build_world_state(frame_index, perception_output_dict)

    ✅ Legacy compatibility (old signature style):
        ws = build_world_state(frame_index, polyline, chain_balls, frog_center, shooter_ball_or_color, next_color)
    """
    # -------------------------
    # 1) Normalize inputs
    # -------------------------
    if isinstance(perception_or_polyline, dict):
        frog_center, polyline, balls_dicts, shooter_color_p, next_color_p = _extract_perception_fields(
            perception_or_polyline
        )

        chain_balls: List[Dict[str, Any]] = []
        for b in balls_dicts:
            try:
                chain_balls.append(
                    {
                        "x": int(b.get("x", 0)),
                        "y": int(b.get("y", 0)),
                        "r": int(b.get("r", 0)),
                        "color": b.get("color"),
                        "color_conf": float(b.get("color_conf", 1.0)),  # not provided by perception -> default
                        "frames_seen": int(b.get("frames_seen", 1)),    # not provided by perception -> default
                    }
                )
            except Exception:
                continue

        shooter_color = shooter_color_override or shooter_color_p
        next_color = next_color_override or next_color_p

    else:
        # Legacy usage
        polyline = _as_point_list(perception_or_polyline)
        frog_center = frog_center_legacy or (0.0, 0.0)

        chain_balls = []
        if chain_balls_legacy:
            for b in list(chain_balls_legacy):
                chain_balls.append(
                    {
                        "x": int(getattr(b, "x", 0)),
                        "y": int(getattr(b, "y", 0)),
                        "r": int(getattr(b, "r", 0)),
                        "color": getattr(b, "color", None),
                        "color_conf": float(getattr(b, "color_conf", 1.0)),
                        "frames_seen": int(getattr(b, "frames_seen", 1)),
                    }
                )

        # In legacy style, shooter_color_override might have been a shooter_ball object or a color string.
        shooter_color = None
        if shooter_color_override is not None and not isinstance(shooter_color_override, str):
            shooter_color = getattr(shooter_color_override, "color", None)
        elif isinstance(shooter_color_override, str):
            shooter_color = shooter_color_override

        next_color = next_color_override

    if polyline is None or len(polyline) < 2:
        # Fallback (prevents crashes). Decision can still act, but aim will be meaningless.
        polyline = [(0.0, 0.0), (1.0, 0.0)]

    s_table = polyline_arc_lengths(polyline)

    # -------------------------
    # 2) Basic statistics
    # -------------------------
    radii = [int(b.get("r", 0)) for b in chain_balls if int(b.get("r", 0)) > 0]
    median_radius = float(np.median(np.array(radii, dtype=np.float64))) if radii else 10.0

    # Project chain balls to path and sort by s
    proj: List[Tuple[float, Dict[str, Any]]] = []
    for b in chain_balls:
        s, _ = project_point_on_polyline((float(b["x"]), float(b["y"])), polyline)
        proj.append((float(s), b))

    proj.sort(key=lambda t: (float(t[0]), int(t[1].get("x", 0)), int(t[1].get("y", 0))))
    s_positions = [s for (s, _) in proj]

    median_spacing = robust_median_spacing(s_positions, median_radius=median_radius)
    WORLD_MEMORY["median_spacing"] = float(median_spacing)

    # -------------------------
    # 3) BallState list
    # -------------------------
    balls_out: List[BallState] = []
    matched_velocities: List[float] = []

    if ENABLE_TRACKING:
        # Optional tracking (relaxed). Keep it off until you need velocity/IDs.
        mem_balls: Dict[int, Dict[str, Any]] = WORLD_MEMORY.get("balls", {})
        mem_items = sorted(mem_balls.items(), key=lambda kv: (float(kv[1]["s"]), int(kv[0])))
        used_ids = set()

        for s, b in proj:
            best_id: Optional[int] = None
            best_ds = float("inf")

            for bid, info in mem_items:
                if bid in used_ids:
                    continue
                ds = abs(float(info["s"]) - float(s))

                # ✅ Relaxed gate (less strict than before)
                gate = max(3.0 * median_spacing, 4.0 * median_radius)

                if ds <= gate and ds < best_ds:
                    # Soft preference by color (never hard-block)
                    if b.get("color") is not None and info.get("color") is not None and b.get("color") != info.get("color"):
                        ds *= 1.10
                    best_ds = ds
                    best_id = int(bid)

            if best_id is None:
                best_id = int(WORLD_MEMORY["next_id"])
                WORLD_MEMORY["next_id"] = int(WORLD_MEMORY["next_id"]) + 1

            used_ids.add(best_id)

            prev_info = mem_balls.get(best_id)
            if prev_info is not None:
                dt = int(frame_index) - int(prev_info.get("frame_index", frame_index))
                if dt > 0:
                    matched_velocities.append(float((float(s) - float(prev_info["s"])) / float(dt)))

            balls_out.append(
                BallState(
                    ball_id=int(best_id),
                    x=int(b.get("x", 0)),
                    y=int(b.get("y", 0)),
                    r=int(b.get("r", 0)),
                    s=float(s),
                    color=b.get("color"),
                    color_conf=float(b.get("color_conf", 1.0)),
                    frames_seen=int(b.get("frames_seen", 1)),
                )
            )

        # Update memory
        new_mem: Dict[int, Dict[str, Any]] = {}
        for bs in balls_out:
            new_mem[int(bs.ball_id)] = {"s": float(bs.s), "frame_index": int(frame_index), "color": bs.color}
        WORLD_MEMORY["balls"] = new_mem
        WORLD_MEMORY["last_frame_index"] = int(frame_index)

        if matched_velocities:
            WORLD_MEMORY["velocity"] = float(np.median(np.array(matched_velocities, dtype=np.float64)))
        chain_velocity = float(WORLD_MEMORY.get("velocity", 0.0))

    else:
        # ✅ Non-strict mode: IDs by s-order, never blocks gameplay.
        for i, (s, b) in enumerate(proj):
            balls_out.append(
                BallState(
                    ball_id=int(i + 1),
                    x=int(b.get("x", 0)),
                    y=int(b.get("y", 0)),
                    r=int(b.get("r", 0)),
                    s=float(s),
                    color=b.get("color"),
                    color_conf=float(b.get("color_conf", 1.0)),
                    frames_seen=int(b.get("frames_seen", 1)),
                )
            )
        chain_velocity = 0.0

    # -------------------------
    # 4) Shooter / next colors (important to keep playing)
    # -------------------------
    chain_colors = [bs.color for bs in balls_out]

    if shooter_color is None:
        shooter_color = WORLD_MEMORY.get("shooter_color")
    if shooter_color is None:
        # ✅ Final fallback so Decision does not return None
        shooter_color = _guess_shooter_color_from_chain(chain_colors)
    WORLD_MEMORY["shooter_color"] = shooter_color

    if next_color is None:
        next_color = WORLD_MEMORY.get("next_color")
    WORLD_MEMORY["next_color"] = next_color

    wx, wy = float(frog_center[0]), float(frog_center[1])

    return WorldState(
        frame_index=int(frame_index),
        frog_x=float(wx),
        frog_y=float(wy),
        shooter_color=shooter_color,
        next_color=next_color,
        num_balls=int(len(balls_out)),
        median_radius=float(median_radius),
        median_spacing=float(median_spacing),
        chain_velocity=float(chain_velocity),
        path_num_points=int(len(polyline)),
        balls=balls_out,
        path_polyline=list(polyline),
    )


def resolve_cascading_simulation(
    colors: Sequence[Optional[str]],
    insert_index: int,
    insert_color: Optional[str],
) -> Tuple[List[Optional[str]], int]:
    """
    Deterministic color-only cascade simulation.

    Insert insert_color at insert_index (clamped), then remove any contiguous run >=3.
    Repeat until stable.
    Returns (new_colors, total_removed).
    """
    lst: List[Optional[str]] = list(colors)
    if insert_color is None:
        return lst, 0
    idx = int(np.clip(int(insert_index), 0, len(lst)))
    lst.insert(idx, insert_color)

    total_removed = 0
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(lst):
            j = i + 1
            while j < len(lst) and lst[j] == lst[i] and lst[i] is not None:
                j += 1
            run_len = j - i
            if lst[i] is not None and run_len >= 3:
                total_removed += run_len
                del lst[i:j]
                changed = True
                i = max(0, i - 2)
            else:
                i = j
    return lst, int(total_removed)


def worldstate_to_dict(ws: WorldState) -> Dict[str, Any]:
    """Serialize key numeric fields of WorldState for logging."""
    return {
        "frame_index": int(ws.frame_index),
        "frog_x": float(ws.frog_x),
        "frog_y": float(ws.frog_y),
        "num_balls": int(ws.num_balls),
        "median_radius": float(ws.median_radius),
        "median_spacing": float(ws.median_spacing),
        "chain_velocity": float(ws.chain_velocity),
        "path_num_points": int(ws.path_num_points),
        "shooter_color": ws.shooter_color,
        "next_color": ws.next_color,
    }
