# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Boying 四足机器人配置文件。

Boying 机器人所有 12 个关节均使用 ENCOS EC-A6408-P2-25 行星减速电机。
电机参数来源于 ENCOS 数据手册 V3.14 及反驱阻尼实测数据。
完整推导说明见 unitree_actuator.py 中的 EncosActuatorCfg_A6408P225。
"""

import isaaclab.sim as sim_utils                               # Isaac Lab 仿真工具模块
from isaaclab.assets.articulation import ArticulationCfg       # 关节式机器人配置基类

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR          # 资源文件根目录（指向 resources/）
from robot_lab.assets.unitree_actuator import EncosActuatorCfg_A6408P225  # Boying 专用电机配置

BOYING_CFG = ArticulationCfg(
    # ── URDF 加载与物理属性 ──────────────────────────────────────────
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,                          # 不固定底座，允许自由漂浮（正常行走模式）
        merge_fixed_joints=True,                 # 将 fixed joint 子体合并到父体（head/hm/imu 并入 base）
        replace_cylinders_with_capsules=True,    # 将碰撞圆柱替换为胶囊体（PhysX 推荐，稳定性更好）
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/boying_description/urdf/boying_description_withouthm.urdf",
        # ^ URDF 文件路径：不含 hm 配重版本，已去除 head/hm 碰撞体，FR/RR calf 质量已修正
        activate_contact_sensors=True,           # 启用接触力传感器（用于 undesired_contacts 奖励和终止判断）
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,               # 开启重力
            retain_accelerations=False,          # 不缓存加速度（节省内存）
            linear_damping=0.0,                  # 线性阻尼（空气阻力模型，0 = 无）
            angular_damping=0.0,                 # 角阻尼（0 = 无）
            max_linear_velocity=1000.0,          # 最大线速度限制 [m/s]（防止数值爆炸）
            max_angular_velocity=1000.0,         # 最大角速度限制 [rad/s]
            max_depenetration_velocity=1.0,      # 碰撞穿透修正最大速度 [m/s]
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,        # 开启自碰撞（防止腿部穿透机身）
            solver_position_iteration_count=8,   # 位置求解迭代次数（8 次，适合复杂结构）
            solver_velocity_iteration_count=4,   # 速度求解迭代次数（4 次）
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0,   # URDF 自带 PD 增益置零（由 actuators 配置覆盖）
                damping=0      # 同上，由下方 actuators 字典中的 stiffness/damping 统一管理
            )
        ),
    ),

    # ── 初始状态 ────────────────────────────────────────────────────
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.42),           # 初始位置 [m]：x=0, y=0, z=0.42（站立高度，足端离地约 84mm）
        joint_pos={
            ".*R_hip_joint": -0.1,       # 右侧髋关节初始角度 -0.1 rad（轻微外展，与 Go2 对齐）
            ".*L_hip_joint":  0.1,       # 左侧髋关节初始角度 +0.1 rad（对称外展）
            "F[L,R]_thigh_joint": 0.8,  # 前腿大腿关节初始角度 0.8 rad（半蹲姿态）
            "R[L,R]_thigh_joint": 1.0,  # 后腿大腿关节初始角度 1.0 rad（后腿稍弯，与 Go2 对齐）
            ".*_calf_joint": -1.5,       # 所有小腿关节初始角度 -1.5 rad（calf 范围 [-2.75, -0.93]）
        },
        joint_vel={".*": 0.0},           # 所有关节初始速度为零
    ),

    # ── 关节软限位 ──────────────────────────────────────────────────
    soft_joint_pos_limit_factor=0.95,
    # ^ 软限位因子：关节在 URDF 硬限位的 95% 处开始受到惩罚
    # 例：calf 硬限位 [-2.7478, -0.9269] → 软限位 [-2.6104, -0.8806]

    # ── 执行器配置 ──────────────────────────────────────────────────
    actuators={
        # EC-A6408-P2-25 规格：减速比 25:1，峰值扭矩 60 Nm，额定 20 Nm @ 133 RPM
        "hip_thigh": EncosActuatorCfg_A6408P225(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint"],  # 覆盖全部 8 个髋/大腿关节
            effort_limit=60.0,    # 峰值堵转扭矩 [Nm]（电机输出轴）
            velocity_limit=15.60, # 空载输出转速 [rad/s] = 149 RPM（髋/大腿关节）
            stiffness=60.0,       # PD 位置增益 Kp [Nm/rad]
            damping=4.5,          # PD 速度增益 Kd [Nm·s/rad]（额定转速下 D 项 = 62.7 Nm，略超峰值，注意）
            friction=0.0,         # 库仑摩擦系数（摩擦已由 T-N 模型中的 Fs/Fd 处理，此处置零）
        ),
        "calf": EncosActuatorCfg_A6408P225(
            joint_names_expr=[".*_calf_joint"],  # 覆盖全部 4 个小腿关节
            effort_limit=90.0,    # 峰值堵转扭矩 [Nm]（与髋/大腿相同）
            velocity_limit=10.4,  # 小腿关节速度上限 [rad/s]（URDF 中 calf 限制，低于髋/大腿的 15.6）
            stiffness=60.0,       # PD 位置增益 Kp [Nm/rad]
            damping=4.5,          # PD 速度增益 Kd [Nm·s/rad]
            friction=0.0,         # 同上，由 T-N 模型统一处理
        ),
    },
)
