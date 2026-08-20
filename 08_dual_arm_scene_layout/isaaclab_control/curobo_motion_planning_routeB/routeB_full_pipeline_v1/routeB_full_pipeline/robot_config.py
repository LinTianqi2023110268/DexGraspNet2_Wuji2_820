from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


ATTACHED_LINK = "routeB_attached_object"
RIGHT_ARM_JOINTS = tuple(f"arm_r_joint_{i}" for i in range(1, 8))


def _kinematics_dict(robot_dict: dict) -> dict:
    if "robot_cfg" in robot_dict:
        return robot_dict["robot_cfg"]["kinematics"]
    return robot_dict["kinematics"]


def with_attachment_link(
    robot_source: Mapping[str, Any],
    *,
    parent_link: str = "arm_r_link_tf",
    sphere_slots: int = 48,
) -> dict:
    """Add a fixed collision-only attachment link to a copied cuRobo config.

    Uses the official KinematicsLoaderCfg `extra_links` +
    `extra_collision_spheres` mechanisms.  No URDF/USD file is edited.
    The link is identity-fixed under arm_r_link_tf so AttachmentManager's
    tool-frame-relative update contract remains valid.
    """
    data = deepcopy(dict(robot_source))
    kin = _kinematics_dict(data)

    extra_links = dict(kin.get("extra_links") or {})
    extra_links[ATTACHED_LINK] = {
        "link_name": ATTACHED_LINK,
        "parent_link_name": str(parent_link),
        "joint_name": ATTACHED_LINK + "_fixed_joint",
        "joint_type": "FIXED",
        "fixed_transform": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    }
    kin["extra_links"] = extra_links

    collision_links = list(kin.get("collision_link_names") or [])
    if ATTACHED_LINK not in collision_links:
        collision_links.append(ATTACHED_LINK)
    kin["collision_link_names"] = collision_links

    extra_spheres = dict(kin.get("extra_collision_spheres") or {})
    extra_spheres[ATTACHED_LINK] = max(1, int(sphere_slots))
    kin["extra_collision_spheres"] = extra_spheres
    return data


def build_locked_joint_values(
    *,
    full_joint_names: list[str],
    measured_by_name: dict[str, float],
    hand_joint_names: list[str],
    hand_q20,
) -> dict[str, float]:
    hand_map = {
        str(name): float(value)
        for name, value in zip(hand_joint_names, hand_q20)
    }
    active = set(RIGHT_ARM_JOINTS)
    lock: dict[str, float] = {}
    for name in full_joint_names:
        if name in active:
            continue
        if name in hand_map:
            lock[name] = hand_map[name]
        elif name in measured_by_name:
            lock[name] = float(measured_by_name[name])
        else:
            raise KeyError(f"no fixed value available for locked joint {name}")
    return lock
