from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import yaml
from pytransform3d import rotations as rt


class SafetyConfigError(RuntimeError):
    pass


@dataclass
class Rect2D:
    x_min: float
    x_max: float
    y_min: float
    y_max: float


@dataclass
class HeightBand:
    z_min: float
    z_max: float


@dataclass
class Pillar:
    center_xy: Tuple[float, float]
    radius: float
    margin: float = 0.03


@dataclass
class SafetyParams:
    yaw_range: Tuple[float, float]
    pitch_range: Tuple[float, float]
    roll_range: Tuple[float, float]
    forward_axis: int = 0
    world_forward: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    world_up: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    max_forward_tilt: float = np.deg2rad(75)
    v_max_pos: float = 0.40
    v_max_rot: float = np.deg2rad(150)
    boundary_slowdown_dist: float = 0.03
    hold_on_violation: bool = True
    inter_arm_min_dist: float = 0.14


@dataclass
class ArmState:
    ee_pos: np.ndarray
    ee_quat: np.ndarray


class BimanualSafetyManager:
    def __init__(
        self,
        left_rect: Rect2D,
        right_rect: Rect2D,
        hband: HeightBand,
        pillars_left: List[Pillar],
        pillars_right: List[Pillar],
        params: SafetyParams,
    ):
        self.rect = {0: left_rect, 1: right_rect}
        self.hband = hband
        self.params = params
        self.pillars = {0: pillars_left, 1: pillars_right}
        self._last_cmd: Dict[int, Optional[ArmState]] = {0: None, 1: None}

        self._fwd = _unit(np.array(params.world_forward, dtype=float))
        self._up = _unit(np.array(params.world_up, dtype=float))

    def filter_target(
        self,
        arm_index: int,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        other_arm_state: Optional[ArmState],
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        p = target_pos.copy()
        q = target_quat.copy()

        p = self._clamp_to_workspace(arm_index, p)
        p = self._push_from_pillars(arm_index, p)
        q = self._project_orientation(q)

        if other_arm_state is not None:
            if np.linalg.norm(p - other_arm_state.ee_pos) < self.params.inter_arm_min_dist:
                if self.params.hold_on_violation:
                    last = self._last_cmd[arm_index]
                    if last is not None:
                        return last.ee_pos.copy(), last.ee_quat.copy(), False
                direction = _unit(p - other_arm_state.ee_pos)
                if np.linalg.norm(direction) < 1e-12:
                    direction = np.array([1.0, 0.0, 0.0])
                p = other_arm_state.ee_pos + direction * self.params.inter_arm_min_dist

        p = self._ease_near_edges(arm_index, p)
        p, q = self._rate_limit(arm_index, p, q, dt)

        self._last_cmd[arm_index] = ArmState(p.copy(), q.copy())
        return p, q, True

    def _clamp_to_workspace(self, arm: int, p: np.ndarray) -> np.ndarray:
        r = self.rect[arm]
        p[0] = np.clip(p[0], r.x_min, r.x_max)
        p[1] = np.clip(p[1], r.y_min, r.y_max)
        p[2] = np.clip(p[2], self.hband.z_min, self.hband.z_max)
        return p

    def _push_from_pillars(self, arm: int, p: np.ndarray) -> np.ndarray:
        for pillar in self.pillars[arm]:
            cx, cy = pillar.center_xy
            dx = p[0] - cx
            dy = p[1] - cy
            dist = np.hypot(dx, dy)
            limit = pillar.radius + pillar.margin
            if dist < limit and dist > 1e-6:
                scale = limit / dist
                p[0] = cx + dx * scale
                p[1] = cy + dy * scale
            elif dist <= 1e-6:
                p[0] = cx + limit
                p[1] = cy
        return p

    def _project_orientation(self, q: np.ndarray) -> np.ndarray:
        R = rt.matrix_from_quaternion(q)
        ypr = rt.euler_from_matrix(R, 2, 1, 0, extrinsic=False)
        ypr = np.array(
            [
                _clamp_angle(ypr[0], self.params.yaw_range),
                _clamp_angle(ypr[1], self.params.pitch_range),
                _clamp_angle(ypr[2], self.params.roll_range),
            ]
        )
        R = rt.matrix_from_euler(ypr, 2, 1, 0, extrinsic=False)

        forward = R[:, self.params.forward_axis]
        theta = _angle_between(forward, self._fwd)
        if theta > self.params.max_forward_tilt:
            proj = self._fwd * forward.dot(self._fwd)
            lateral = forward - proj
            if np.linalg.norm(lateral) < 1e-8:
                v_new = self._fwd
            else:
                u = _unit(lateral)
                v_new = (
                    np.cos(self.params.max_forward_tilt) * self._fwd
                    + np.sin(self.params.max_forward_tilt) * u
                )
            side = _unit(np.cross(self._up, v_new))
            up = _unit(np.cross(v_new, side))
            R = np.column_stack([v_new, side, up])
        return rt.quaternion_from_matrix(R)

    def _ease_near_edges(self, arm: int, p: np.ndarray) -> np.ndarray:
        return p

    def _rate_limit(
        self, arm: int, p: np.ndarray, q: np.ndarray, dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        last = self._last_cmd[arm]
        if last is None or dt <= 0:
            return p, q

        delta = p - last.ee_pos
        max_step = self.params.v_max_pos * dt
        norm = np.linalg.norm(delta)
        if max_step > 0 and norm > max_step:
            p = last.ee_pos + delta * (max_step / norm)

        dq = rt.concatenate_quaternions(rt.quaternion_conjugate(last.ee_quat), q)
        angle = 2.0 * np.arccos(np.clip(dq[0], -1.0, 1.0))
        if angle > 1e-6:
            limit = self.params.v_max_rot * dt
            if limit <= 0:
                return p, last.ee_quat.copy()
            t = min(1.0, limit / angle)
            q = _slerp(last.ee_quat, q, t)
        return p, q


def load_safety_config(
    config_path: Union[str, Path, None]
) -> Tuple[Rect2D, Rect2D, HeightBand, List[Pillar], List[Pillar], SafetyParams]:
    path = resolve_safety_config_path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Safety config not found: {path}")

    with path.open("r") as f:
        raw_cfg = yaml.safe_load(f) or {}

    rectangles = raw_cfg.get("rectangles", {})
    left_rect = _parse_rect(rectangles.get("left"), "rectangles.left")
    right_rect_cfg = rectangles.get("right", rectangles.get("left"))
    right_rect = _parse_rect(right_rect_cfg, "rectangles.right")

    height_cfg = raw_cfg.get("height_band", {})
    z_range = _require_sequence(height_cfg.get("z"), 2, "height_band.z")
    hband = HeightBand(z_min=float(z_range[0]), z_max=float(z_range[1]))

    pillars_cfg = raw_cfg.get("pillars", {})
    left_pillars = _parse_pillars(pillars_cfg.get("left", []), "pillars.left")
    right_pillars = _parse_pillars(
        pillars_cfg.get("right", pillars_cfg.get("left", [])), "pillars.right"
    )

    params_cfg = raw_cfg.get("params", {})
    params = SafetyParams(
        yaw_range=_parse_angle_range(params_cfg, "yaw_range", (-np.pi / 2, np.pi / 2)),
        pitch_range=_parse_angle_range(
            params_cfg, "pitch_range", (-np.deg2rad(70), np.deg2rad(70))
        ),
        roll_range=_parse_angle_range(
            params_cfg, "roll_range", (-np.deg2rad(60), np.deg2rad(60))
        ),
        forward_axis=int(params_cfg.get("forward_axis", 0)),
        world_forward=_parse_vector(params_cfg, "world_forward", 3, (1.0, 0.0, 0.0)),
        world_up=_parse_vector(params_cfg, "world_up", 3, (0.0, 0.0, 1.0)),
        max_forward_tilt=_parse_angle_value(
            params_cfg, "max_forward_tilt", np.deg2rad(75)
        ),
        v_max_pos=float(params_cfg.get("v_max_pos", 0.40)),
        v_max_rot=_parse_angle_value(params_cfg, "v_max_rot", np.deg2rad(150)),
        boundary_slowdown_dist=float(params_cfg.get("boundary_slowdown_dist", 0.03)),
        hold_on_violation=bool(params_cfg.get("hold_on_violation", True)),
        inter_arm_min_dist=float(params_cfg.get("inter_arm_min_dist", 0.14)),
    )

    return left_rect, right_rect, hband, left_pillars, right_pillars, params


def resolve_safety_config_path(config_path: Union[str, Path, None]) -> Path:
    if config_path is not None:
        return Path(config_path).expanduser().resolve()
    env_path = os.environ.get("BVP_SAFETY_CONFIG")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return DEFAULT_SAFETY_CONFIG_PATH


def _parse_rect(cfg: Optional[Dict[str, List[float]]], label: str) -> Rect2D:
    if cfg is None:
        raise SafetyConfigError(f"Missing rectangle config for {label}")
    x_range = _require_sequence(cfg.get("x"), 2, f"{label}.x")
    y_range = _require_sequence(cfg.get("y"), 2, f"{label}.y")
    return Rect2D(
        x_min=float(x_range[0]),
        x_max=float(x_range[1]),
        y_min=float(y_range[0]),
        y_max=float(y_range[1]),
    )


def _parse_pillars(cfg_list: List[Dict[str, List[float]]], label: str) -> List[Pillar]:
    pillars: List[Pillar] = []
    for idx, cfg in enumerate(cfg_list):
        center = _require_sequence(
            cfg.get("center"), 2, f"{label}[{idx}].center"
        )
        radius = cfg.get("radius")
        if radius is None:
            raise SafetyConfigError(f"Missing radius in {label}[{idx}]")
        margin = float(cfg.get("margin", 0.03))
        pillars.append(
            Pillar(center_xy=(float(center[0]), float(center[1])), radius=float(radius), margin=margin)
        )
    return pillars


def _parse_angle_range(
    cfg: Dict[str, Union[float, List[float]]],
    key: str,
    default: Tuple[float, float],
) -> Tuple[float, float]:
    if f"{key}_deg" in cfg:
        values = cfg[f"{key}_deg"]
        seq = _require_sequence(values, 2, f"params.{key}_deg")
        return tuple(float(np.deg2rad(float(v))) for v in seq)  # type: ignore[return-value]
    if key in cfg:
        seq = _require_sequence(cfg[key], 2, f"params.{key}")
        return tuple(float(v) for v in seq)
    return default


def _parse_angle_value(
    cfg: Dict[str, Union[float, List[float]]],
    key: str,
    default: float,
) -> float:
    deg_key = f"{key}_deg"
    if deg_key in cfg:
        return float(np.deg2rad(float(cfg[deg_key])))
    if key in cfg:
        return float(cfg[key])
    return default


def _parse_vector(
    cfg: Dict[str, Union[float, List[float]]],
    key: str,
    length: int,
    default: Tuple[float, ...],
) -> Tuple[float, ...]:
    if key not in cfg:
        return tuple(default)
    seq = _require_sequence(cfg[key], length, f"params.{key}")
    return tuple(float(v) for v in seq)


def _require_sequence(value, length: int, label: str):
    if value is None:
        raise SafetyConfigError(f"Missing {label}")
    if not isinstance(value, (list, tuple)):
        raise SafetyConfigError(f"{label} must be a sequence")
    if len(value) != length:
        raise SafetyConfigError(f"{label} must have length {length}")
    return value


def _clamp_angle(angle: float, bounds: Tuple[float, float]) -> float:
    wrapped = (angle + np.pi) % (2 * np.pi) - np.pi
    return float(np.clip(wrapped, bounds[0], bounds[1]))


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        return vec.copy()
    return vec / norm


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    a_u = _unit(a)
    b_u = _unit(b)
    return float(np.arccos(np.clip(a_u.dot(b_u), -1.0, 1.0)))


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    return rt.quaternion_slerp(q0, q1, t, shortest_path=True)


DEFAULT_SAFETY_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "safety_config"
    / "widowx.yaml"
)

__all__ = [
    "Rect2D",
    "HeightBand",
    "Pillar",
    "SafetyParams",
    "ArmState",
    "BimanualSafetyManager",
    "SafetyConfigError",
    "load_safety_config",
    "resolve_safety_config_path",
    "DEFAULT_SAFETY_CONFIG_PATH",
]
