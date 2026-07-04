# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Configuration for Boying quadruped robot.

The Boying robot uses ENCOS EC-A6408-P2-25 planetary gearbox actuators on all 12 joints.
Motor parameters are derived from the ENCOS datasheet V3.14 and back-drive damping
measurements.  See EncosActuatorCfg_A6408P225 in unitree_actuator.py for the full
derivation notes.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR
from robot_lab.assets.unitree_actuator import EncosActuatorCfg_A6408P225

BOYING_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/boying_description/urdf/boying_description_withouthm.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0, damping=0
            )
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.42),
        joint_pos={
            ".*R_hip_joint": -0.1,
            ".*L_hip_joint": 0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # EC-A6408-P2-25: 25:1 planetary, 60 Nm peak, 20 Nm rated @ 133 RPM
        "hip_thigh": EncosActuatorCfg_A6408P225(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint"],
            effort_limit=60.0,    # peak stall torque [Nm]
            velocity_limit=15.60, # no-load output speed [rad/s] = 149 RPM
            stiffness=60.0,       # PD position gain [Nm/rad]
            damping=4.5,          # PD velocity gain [Nm·s/rad]; 4.5 exceeded peak torque at rated speed
            friction=0.0,         # joint coulomb friction (handled by Fs/Fd in T-N model)
        ),
        "calf": EncosActuatorCfg_A6408P225(
            joint_names_expr=[".*_calf_joint"],
            effort_limit=60.0,    # peak stall torque [Nm]
            velocity_limit=10.4,  # calf URDF velocity limit [rad/s] (lower than hip/thigh)
            stiffness=60.0,       # PD position gain [Nm/rad]
            damping=4.5,          # PD velocity gain [Nm·s/rad]
            friction=0.0,
        ),
    },
)
