import math
import os
from threading import Lock
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pinocchio as pin
import yaml

from bunny_teleop_server.control.base import BaseMotionControl

try:  # pragma: no cover - optional dependency for collision queries
    import hppfcl  # noqa: F401

    _HAVE_FCL = True
except Exception:  # pragma: no cover - allow runtime without FCL
    _HAVE_FCL = False


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_float_list(env_value: Optional[str], expected: int) -> Optional[np.ndarray]:
    if env_value is None:
        return None
    parts = [p for p in env_value.replace(";", ",").split(",") if p.strip()]
    if len(parts) != expected:
        raise ValueError(
            f"Expected {expected} comma-separated floats but got {len(parts)} in '{env_value}'"
        )
    return np.asarray([float(p) for p in parts], dtype=float)



def _parse_float_sequence(value, expected: int, label: str) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, str):
        return _parse_float_list(value, expected)
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except Exception as exc:
        raise ValueError(
            f"{label} must be a sequence of {expected} floats."
        ) from exc
    if arr.shape[0] != expected:
        raise ValueError(
            f"{label} must contain {expected} floats but received {arr.shape[0]}."
        )
    return arr


def _normalize_quat(wxyz: Sequence[float]) -> np.ndarray:
    quat = np.asarray(wxyz, dtype=float)
    norm = np.linalg.norm(quat)
    if norm <= 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return quat / norm


def _rot_z(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class PinocchioMotionControl(BaseMotionControl):
    def __init__(
        self,
        robot_name: str,
        robot_config_path: str,
    ):
        self.robot_name = robot_name
        self._qpos_lock = Lock()

        # Config
        robot_config_path = Path(robot_config_path)
        self.robot_config_path = robot_config_path
        if not robot_config_path.is_absolute():
            raise RuntimeError(
                f"Robot config path must be absolute: {robot_config_path}"
            )

        with robot_config_path.open("r") as f:
            cfg = yaml.safe_load(f)["robot_cfg"]
        self.robot_cfg = cfg
        ik_damping = cfg["kinematics"]["ik_damping"]
        ee_name = cfg["kinematics"]["ee_link"]
        self.ik_damping = float(ik_damping) * np.eye(6)
        self.ik_eps = float(cfg["kinematics"]["eps"])
        self.dt = float(cfg["dt"])
        self.ee_name = ee_name

        print("Pinocchio Configurations : ")
        print(f"End Effector Name: {self.ee_name}")
        print(f"IK Damping: {self.ik_damping}")
        print(f"IK Epsilon: {self.ik_eps}")
        print(f"Time Step: {self.dt}")

        # Build robot
        urdf_path = self.get_urdf_absolute_path(cfg, robot_config_path)
        self.robot_urdf_path = urdf_path
        self.model: pin.Model = pin.buildModelFromUrdf(str(urdf_path), mimic=False)
        self.data: pin.Data = self.model.createData()
        frame_mapping: Dict[str, int] = {}

        print("Pinocchio shape : ")

        for i in range(1, self.model.njoints):
            joint = self.model.joints[i]
            joint_name = self.model.names[i]
            print("Joint name : ", joint_name)
            print("Joint type : ", joint)



        # print(pin.neutral(self.model).shape[0])

        for i, frame in enumerate(self.model.frames):
            frame_mapping[frame.name] = i

        if self.ee_name not in frame_mapping:
            raise ValueError(
                f"End effector name {ee_name} not find in robot with path: {urdf_path}."
            )
        self.frame_mapping = frame_mapping
        self.ee_frame_id = frame_mapping[ee_name]

        # Current state
        self.qpos = pin.neutral(self.model)
        pin.forwardKinematics(self.model, self.data, self.qpos)
        self.ee_pose: pin.SE3 = pin.updateFramePlacement(
            self.model, self.data, self.ee_frame_id
        )

        # Safety configuration toggled via environment flags
        self.enable_safe_mode = _env_flag("BTP_SAFE_MODE")
        self.enable_orientation_gate = _env_flag(
            "BTP_SAFE_ORIENTATION_GATE", default=self.enable_safe_mode
        )
        self.collision_margin = float(os.getenv("BTP_SAFE_MARGIN", "0.02"))
        self.collision_substeps = max(1, int(os.getenv("BTP_SAFE_SUBSTEPS", "8")))
        cage_path_env = os.getenv("BTP_CAGE_URDF")
        self._cage_urdf_path = (
            Path(cage_path_env).expanduser().resolve() if cage_path_env else None
        )

        safety_cfg = cfg.get("safety")
        if not isinstance(safety_cfg, dict):
            safety_cfg = {}
        base_pose_cfg = safety_cfg.get("cage_base_pq")
        base_pose_env = os.getenv("BTP_CAGE_BASE_PQ")
        if base_pose_cfg is not None:
            self._base_in_cage_pq = _parse_float_sequence(
                base_pose_cfg, 7, "robot_cfg.safety.cage_base_pq"
            )
        else:
            self._base_in_cage_pq = _parse_float_list(base_pose_env, 7)
        self._collision_ready = False

        if self.enable_safe_mode and self._base_in_cage_pq is not None:
            self._collision_ready = self._setup_collision_environment()
            if not self._collision_ready:
                print(
                    "[Safety] Collision checks unavailable; continuing without cage guard."
                )

    # ------------------------------------------------------------------
    # Safety setup and helpers
    # ------------------------------------------------------------------
    def _setup_collision_environment(self) -> bool:
        if not _HAVE_FCL:
            print("[Safety] hppfcl not available; disable safe mode.")
            return False

        if self._cage_urdf_path is None or not self._cage_urdf_path.is_file():
            print(
                "[Safety] No cage URDF found. Set BTP_CAGE_URDF to an existing file."
            )
            return False

        if self._base_in_cage_pq is None:
            print(
                "[Safety] Robot base pose not provided; waiting before enabling collision guard."
            )
            return False

        try:
            self.collision_model = pin.buildGeomFromUrdf(
                self.model,
                str(self.robot_urdf_path),
                pin.GeometryType.COLLISION,
            )
            self.collision_data = self.collision_model.createData()
            self._robot_geom_indices = list(range(self.collision_model.ngeoms))
            self._robot_geom_count = self.collision_model.ngeoms
        except Exception as exc:
            print(f"[Safety] Failed to build robot collision geometry: {exc}")
            return False

        try:
            cage_model = pin.buildModelFromUrdf(
                str(self._cage_urdf_path), mimic=False
            )
            cage_collision = pin.buildGeomFromUrdf(
                cage_model,
                str(self._cage_urdf_path),
                pin.GeometryType.COLLISION,
            )
        except Exception as exc:
            print(f"[Safety] Failed to load cage geometry: {exc}")
            return False

        if self._base_in_cage_pq is None:
            T_base_in_cage = pin.SE3.Identity()
        else:
            xyzw = np.array(
                [
                    self._base_in_cage_pq[4],
                    self._base_in_cage_pq[5],
                    self._base_in_cage_pq[6],
                    self._base_in_cage_pq[3],
                ]
            )
            T_base_in_cage = pin.XYZQUATToSE3(
                np.concatenate([self._base_in_cage_pq[:3], xyzw])
            )

        self._cage_geom_indices: List[int] = []
        for geom in cage_collision.geometryObjects:
            geom_copy = pin.GeometryObject(geom)
            geom_copy.parentJoint = 0
            geom_copy.placement = T_base_in_cage.inverse() * geom_copy.placement
            self.collision_model.addGeometryObject(geom_copy)
            self._cage_geom_indices.append(self.collision_model.ngeoms - 1)

        for i in self._robot_geom_indices:
            for j in self._cage_geom_indices:
                self.collision_model.addCollisionPair(pin.CollisionPair(i, j))

        self.collision_data = self.collision_model.createData()
        return True

    def configure_environment_base_pose(
        self, base_pose_pq: Optional[Sequence[float]]
    ) -> None:
        super().configure_environment_base_pose(base_pose_pq)
        if not self.enable_safe_mode or base_pose_pq is None:
            return

        base_arr = np.asarray(base_pose_pq, dtype=float).reshape(-1)
        if base_arr.shape[0] != 7:
            raise ValueError(
                "Environment base pose must be length 7: [x, y, z, qw, qx, qy, qz]."
            )

        if (
            self._base_in_cage_pq is not None
            and np.allclose(self._base_in_cage_pq, base_arr, atol=1e-9)
            and self._collision_ready
        ):
            return

        self._base_in_cage_pq = base_arr
        self._collision_ready = self._setup_collision_environment()
        if not self._collision_ready:
            print(
                "[Safety] Failed to activate collision guard with provided base pose; running without cage constraints."
            )

    def _gate_orientation_forward_side(
        self, pos: np.ndarray, quat_wxyz: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.enable_orientation_gate:
            return pos, quat_wxyz

        R = pin.Quaternion(
            float(quat_wxyz[0]),
            float(quat_wxyz[1]),
            float(quat_wxyz[2]),
            float(quat_wxyz[3]),
        ).toRotationMatrix()
        forward = R[:, 0]

        if forward[0] < 0.0:
            R = _rot_z(math.pi) @ R
            forward = R[:, 0]

        lateral = forward[1]
        if lateral < 0.0 and self.robot_name.lower().find("right") != -1:
            yaw = math.atan2(-lateral, max(forward[0], 1e-9))
            R = _rot_z(yaw) @ R
        elif lateral > 0.0 and self.robot_name.lower().find("left") != -1:
            yaw = math.atan2(lateral, max(forward[0], 1e-9))
            R = _rot_z(-yaw) @ R

        quat_xyzw = pin.Quaternion(R).coeffs()
        gated = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        return pos, _normalize_quat(gated)

    def _min_distance_robot_vs_cage(self) -> float:
        if not (self.enable_safe_mode and self._collision_ready):
            return float("inf")

        try:
            pin.updateGeometryPlacements(
                self.model, self.data, self.collision_model, self.collision_data
            )
            pin.computeDistances(self.collision_model, self.collision_data)
        except Exception:
            return float("inf")

        min_distance = float("inf")
        for idx, _pair in enumerate(self.collision_model.collisionPairs):
            if idx >= len(self.collision_data.distanceResults):
                break
            result = self.collision_data.distanceResults[idx]
            distance = getattr(result, "min_distance", None)
            if distance is None:
                continue
            if distance < min_distance:
                min_distance = distance
        return min_distance

    def _safe_integrate(self, q: np.ndarray, v: np.ndarray, dt: float) -> np.ndarray:
        if not (self.enable_safe_mode and self._collision_ready):
            return pin.integrate(self.model, q, v * dt)

        if self._min_distance_robot_vs_cage() > 5.0 * self.collision_margin:
            return pin.integrate(self.model, q, v * dt)

        step = v * (dt / float(self.collision_substeps))
        q_new = q.copy()
        for _ in range(self.collision_substeps):
            alpha = 1.0
            while alpha > 1e-4:
                q_try = pin.integrate(self.model, q_new, step * alpha)
                pin.forwardKinematics(self.model, self.data, q_try)
                distance = self._min_distance_robot_vs_cage()
                if distance >= self.collision_margin:
                    q_new = q_try
                    break
                alpha *= 0.5
            else:
                return q_new
        return q_new

    def get_min_distance_to_environment(self) -> float:
        if not (self.enable_safe_mode and self._collision_ready):
            return float("inf")
        pin.forwardKinematics(self.model, self.data, self.qpos)
        return self._min_distance_robot_vs_cage()

    def step(self, pos: Optional[np.ndarray], quat: Optional[np.ndarray], repeat=1):
        if quat is None or pos is None:
            raise ValueError("Position and quaternion targets are required for IK step.")

        pos, quat = self._gate_orientation_forward_side(pos, quat)

        xyzw = np.array([quat[1], quat[2], quat[3], quat[0]])
        pose_vec = np.concatenate([pos, xyzw])
        oMdes = pin.XYZQUATToSE3(pose_vec)
        with self._qpos_lock:
            qpos = self.qpos.copy()

        for k in range(100 * repeat):
            pin.forwardKinematics(self.model, self.data, qpos)
            ee_pose = pin.updateFramePlacement(self.model, self.data, self.ee_frame_id)
            J = pin.computeFrameJacobian(self.model, self.data, qpos, self.ee_frame_id)
            iMd = ee_pose.actInv(oMdes)
            err = pin.log(iMd).vector
            if np.linalg.norm(err) < self.ik_eps:
                break

            # JLog = pin.Jlog6(iMd.inverse())
            # J = -JLog@J

            v = J.T.dot(np.linalg.solve(J.dot(J.T) + self.ik_damping, err))
            qpos = self._safe_integrate(qpos, v, self.dt)

        self.set_current_qpos(qpos)

    def compute_ee_pose(self, qpos: np.ndarray) -> np.ndarray:
        pin.forwardKinematics(self.model, self.data, qpos)
        oMf: pin.SE3 = pin.updateFramePlacement(self.model, self.data, self.ee_frame_id)
        xyzw_pose = pin.SE3ToXYZQUAT(oMf)

        return np.concatenate(
            [
                xyzw_pose[:3],
                np.array([xyzw_pose[6], xyzw_pose[3], xyzw_pose[4], xyzw_pose[5]]),
            ]
        )

    def get_current_qpos(self) -> np.ndarray:
        with self._qpos_lock:
            return self.qpos.copy()

    def set_current_qpos(self, qpos: np.ndarray):
        print("Pinocchio shape : ")

        print(pin.neutral(self.model).shape[0])
        with self._qpos_lock:
            self.qpos = qpos
            pin.forwardKinematics(self.model, self.data, self.qpos)
            self.ee_pose = pin.updateFramePlacement(
                self.model, self.data, self.ee_frame_id
            )

    def get_ee_name(self) -> str:
        return self.ee_name

    def get_dof(self) -> int:
        return pin.neutral(self.model).shape[0]

    def get_timestep(self) -> float:
        return self.dt

    def get_joint_names(self) -> List[str]:
        # Pinocchio by default add a dummy joint name called "universe"
        names = list(self.model.names)
        return names[1:]

    def is_use_gpu(self) -> bool:
        return False
