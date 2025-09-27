from threading import Lock
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pinocchio as pin
import yaml

from bunny_teleop_server.control.base import BaseMotionControl


# --- Simple link-vs-obstacle guard (planes & cylinders) ---

class _Plane:
    def __init__(self, point, normal, margin):
        self.point = np.asarray(point, float)
        n = np.asarray(normal, float)
        self.normal = n / (np.linalg.norm(n) + 1e-12)
        self.margin = float(margin)

    # Safe half-space is: (p - point)·normal >= radius + margin
    def violation_amount(self, p, radius):
        return (self.point - p).dot(self.normal) + radius + self.margin


class _CylinderZ:
    # Infinite cylinder along +Z with center in XY
    def __init__(self, center_xy, radius, margin):
        self.c = np.asarray(center_xy, float)
        self.R = float(radius)
        self.margin = float(margin)

    # Positive => violation (inside keepout)
    def violation_amount(self, p, radius):
        d = np.hypot(p[0] - self.c[0], p[1] - self.c[1]) - (self.R + radius + self.margin)
        return -d  # violate when d < 0


class CollisionGuard:
    def __init__(self, model, data, frame_mapping, cfg: dict):
        self.model = model
        self.data = data
        # Which frames to monitor (origins are good proxies for link bodies)
        names = cfg.get("frame_names", [])
        self.frame_ids = [frame_mapping[n] for n in names if n in frame_mapping]
        missing = [n for n in names if n not in frame_mapping]
        if missing:
            print("[CollisionGuard] Warning: frame(s) not found:", missing)

        self.frame_radius = float(cfg.get("frame_radius_m", 0.02))  # ~2 cm default
        self.max_backtrack = int(cfg.get("max_backtrack", 10))
        self.min_scale = float(cfg.get("min_scale", 1e-3))
        self.relief_eps = float(cfg.get("relief_eps", 1e-4))

        # Obstacles
        self.planes = []
        for pl in cfg.get("planes", []):
            self.planes.append(_Plane(pl["point"], pl["normal"], pl.get("margin", 0.0)))
        self.cyls = []
        for cy in cfg.get("cylinders", []):
            self.cyls.append(
                _CylinderZ(cy["center_xy"], cy["radius"], cy.get("margin", 0.0))
            )

    def _is_safe_qpos(self, qpos) -> bool:
        pin.forwardKinematics(self.model, self.data, qpos)
        for fid in self.frame_ids:
            oMf = pin.updateFramePlacement(self.model, self.data, fid)
            p = oMf.translation  # world position of the frame origin
            # Planes
            for pl in self.planes:
                if pl.violation_amount(p, self.frame_radius) > 0:
                    return False
            # Cylinders
            for cy in self.cyls:
                if cy.violation_amount(p, self.frame_radius) > 0:
                    return False
        return True

    def _violation_score(self, qpos) -> float:
        """Return the worst positive violation; <=0 means fully safe."""
        pin.forwardKinematics(self.model, self.data, qpos)
        worst = 0.0
        for fid in self.frame_ids:
            oMf = pin.updateFramePlacement(self.model, self.data, fid)
            p = oMf.translation
            for pl in self.planes:
                worst = max(worst, pl.violation_amount(p, self.frame_radius))
            for cy in self.cyls:
                worst = max(worst, cy.violation_amount(p, self.frame_radius))
        return worst

    def filter_step(self, qpos, step):
        """Backtrack joint step, allowing relief while already in violation."""
        v0 = self._violation_score(qpos)
        s = 1.0
        for _ in range(self.max_backtrack):
            qcand = pin.integrate(self.model, qpos, step * s)
            if v0 <= 0.0:
                if self._is_safe_qpos(qcand):
                    return step * s
            else:
                v1 = self._violation_score(qcand)
                if v1 <= 0.0 or v1 < v0 - self.relief_eps:
                    return step * s
            s *= 0.5
            if s < self.min_scale:
                return np.zeros_like(step)
        return np.zeros_like(step)


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
        if not robot_config_path.is_absolute():
            raise RuntimeError(
                f"Robot config path must be absolute: {robot_config_path}"
            )

        with robot_config_path.open("r") as f:
            cfg = yaml.safe_load(f)["robot_cfg"]
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
        self.model: pin.Model = pin.buildModelFromUrdf(str(urdf_path),mimic=False)
        self.data: pin.Data = self.model.createData()
        frame_mapping: Dict[str, int] = {}

        print("Pinocchio shape : ")

        for i in range(1,self.model.njoints):
         joint = self.model.joints[i]
         joint_name = self.model.names[i]
         print("Joint name : " , joint_name)
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

        # Optional collision/link guard configuration
        self.guard = None
        guard_cfg = cfg.get("collision_guard", {})
        guard_cfg_expanded = guard_cfg
        if guard_cfg.get("enabled", False):
            names_val = guard_cfg.get("frame_names", [])
            if names_val == "__ALL__" or names_val == ["__ALL__"]:
                guard_cfg_expanded = dict(guard_cfg)
                guard_cfg_expanded["frame_names"] = [
                    f.name for f in self.model.frames if f.name != "universe"
                ]
            self.guard = CollisionGuard(
                self.model, self.data, frame_mapping, guard_cfg_expanded
            )
            print("[CollisionGuard] enabled with", len(self.guard.frame_ids), "frames")

        # Current state
        self.qpos = pin.neutral(self.model)
        pin.forwardKinematics(self.model, self.data, self.qpos)
        self.ee_pose: pin.SE3 = pin.updateFramePlacement(
            self.model, self.data, self.ee_frame_id
        )

        self.lower = self.model.lowerPositionLimit.copy()
        self.upper = self.model.upperPositionLimit.copy()
        self.has_lower = np.isfinite(self.lower)
        self.has_upper = np.isfinite(self.upper)
        self.max_joint_step = np.deg2rad(10.0)
        self._hold_streak_s = 0.0
        self._retreat_timer_s = 0.0
        self._retreat_after_hold_s = float(guard_cfg.get("retreat_after_hold_s", 0.6))
        self._retreat_duration_s = float(guard_cfg.get("retreat_duration_s", 1.0))
        self._home_ee_pose = pin.SE3()
        home_q = pin.neutral(self.model)
        pin.forwardKinematics(self.model, self.data, home_q)
        self._home_ee_pose = pin.updateFramePlacement(
            self.model, self.data, self.ee_frame_id
        )

    def step(self, pos: Optional[np.ndarray], quat: Optional[np.ndarray], repeat=1):
        xyzw = np.array([quat[1], quat[2], quat[3], quat[0]])
        pose_vec = np.concatenate([pos, xyzw])
        commanded_pose = pin.XYZQUATToSE3(pose_vec)
        oMdes = self._home_ee_pose if self._retreat_timer_s > 0.0 else commanded_pose
        with self._qpos_lock:
            qpos = self.qpos.copy()
        prev_err_norm = None

        for k in range(100 * repeat):
            target_pose = self._home_ee_pose if self._retreat_timer_s > 0.0 else oMdes
            pin.forwardKinematics(self.model, self.data, qpos)
            ee_pose = pin.updateFramePlacement(self.model, self.data, self.ee_frame_id)
            J = pin.computeFrameJacobian(self.model, self.data, qpos, self.ee_frame_id)
            iMd = ee_pose.actInv(target_pose)
            err = pin.log(iMd).vector
            err_norm = np.linalg.norm(err)
            if err_norm < self.ik_eps:
                break

            # JLog = pin.Jlog6(iMd.inverse())
            # J = -JLog@J

            v = J.T.dot(np.linalg.solve(J.dot(J.T) + self.ik_damping, err))
            step = np.clip(v * self.dt, -self.max_joint_step, self.max_joint_step)
            if self.guard is not None:
                step = self.guard.filter_step(qpos, step)
            if k == 0 and self.guard is not None:
                held = np.allclose(step, 0.0)
                pushing_toward_wall = False
                if self._retreat_timer_s <= 0.0:
                    ee_p = ee_pose.translation
                    dcmd = commanded_pose.translation - ee_p
                    for pl in self.guard.planes:
                        if abs(pl.normal[1]) > 0.5:
                            signed_dist = (ee_p - pl.point).dot(pl.normal)
                            threshold = self.guard.frame_radius + pl.margin + 1e-4
                            if signed_dist < threshold and dcmd.dot(pl.normal) < 0.0:
                                pushing_toward_wall = True
                                break
                if held and pushing_toward_wall:
                    self._hold_streak_s += self.dt
                    if self._hold_streak_s >= self._retreat_after_hold_s:
                        self._retreat_timer_s = self._retreat_duration_s
                        self._hold_streak_s = 0.0
                        oMdes = self._home_ee_pose
                else:
                    self._hold_streak_s = 0.0
                if self._retreat_timer_s > 0.0:
                    self._retreat_timer_s = max(0.0, self._retreat_timer_s - self.dt)
            qpos = pin.integrate(self.model, qpos, step)

            if self.has_lower.any():
                qpos[self.has_lower] = np.maximum(
                    qpos[self.has_lower], self.lower[self.has_lower]
                )
            if self.has_upper.any():
                qpos[self.has_upper] = np.minimum(
                    qpos[self.has_upper], self.upper[self.has_upper]
                )

            if prev_err_norm is not None and err_norm >= prev_err_norm * 0.999:
                break
            prev_err_norm = err_norm

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
        if self.has_lower.any() or self.has_upper.any():
            qpos = qpos.copy()
            if self.has_lower.any():
                qpos[self.has_lower] = np.maximum(
                    qpos[self.has_lower], self.lower[self.has_lower]
                )
            if self.has_upper.any():
                qpos[self.has_upper] = np.minimum(
                    qpos[self.has_upper], self.upper[self.has_upper]
                )
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
