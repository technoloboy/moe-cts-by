import math                                                           # 数学库（未直接使用，保留兼容性）
import isaaclab.sim as sim_utils                                      # Isaac Lab 仿真工具（物理材质、灯光等）
from isaaclab.assets import ArticulationCfg, AssetBaseCfg             # 关节式机器人配置、通用资产配置基类
from isaaclab.envs import ManagerBasedRLEnvCfg                        # 基于管理器的强化学习环境配置基类
from isaaclab.managers import CurriculumTermCfg as CurrTerm           # 课程项配置（重命名为 CurrTerm）
from isaaclab.managers import EventTermCfg as EventTerm               # 事件项配置（重命名为 EventTerm）
from isaaclab.managers import ObservationGroupCfg as ObsGroup         # 观测组配置（重命名为 ObsGroup）
from isaaclab.managers import ObservationTermCfg as ObsTerm           # 单个观测项配置（重命名为 ObsTerm）
from isaaclab.managers import RewardTermCfg as RewTerm                # 奖励项配置（重命名为 RewTerm）
from isaaclab.managers import SceneEntityCfg                          # 场景实体引用（用于指定 robot/sensor）
from isaaclab.managers import TerminationTermCfg as DoneTerm          # 终止条件项配置（重命名为 DoneTerm）
from isaaclab.scene import InteractiveSceneCfg                        # 交互式场景配置基类
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns # 接触力传感器、射线传感器、扫描模式
from isaaclab.terrains import TerrainImporterCfg                      # 地形导入配置
from isaaclab.utils import configclass                                 # 配置类装饰器（自动生成 __init__ 等）
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR  # Nucleus 服务器资源路径常量
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise    # 加性均匀噪声配置（重命名为 Unoise）

import robot_lab.tasks.boying.mdp as mdp                              # Boying 任务的 MDP 模块（奖励/观测/事件等函数）
from robot_lab.assets.boying import BOYING_CFG                        # Boying 机器人关节及执行器配置
from robot_lab.tasks.boying.mdp.terrains import TERRAIN_CFG           # Boying 地形生成配置

# 关节名称列表（保序）：FL/FR/RL/RR × hip/thigh/calf，共12个关节
# 这个顺序决定观测和动作向量中每个维度对应哪个关节
JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",   # 左前腿：髋、大腿、小腿
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",   # 右前腿
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",   # 左后腿
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",   # 右后腿
]

BASE_LINK_NAME = "base"        # 机体主体 link 名称（用于 illegal_contact 终止和质量随机化）
FOOT_LINK_NAME = ".*_foot"     # 足部 link 正则（匹配 FL_foot/FR_foot/RL_foot/RR_foot）
BASE_HEIGHT_TARGET = 0.35      # 机体目标站立高度 [m]（用于 base_height_l2 奖励，比 Go2 的 0.38m 低）


@configclass
class BoyingSceneCfg(InteractiveSceneCfg):
    """Boying 机器人粗糙地形场景配置。"""

    # ── 地形 ─────────────────────────────────────────────────────
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",                   # 地形在 USD stage 中的路径
        terrain_type="generator",                    # 使用程序化地形生成器（非静态 mesh）
        terrain_generator=TERRAIN_CFG,               # 地形类型和比例见 terrains.py
        max_init_terrain_level=5,                    # 初始随机分配的最高难度行（0~9 共10行，越高越难）
        collision_group=-1,                          # -1 表示与所有碰撞组交互
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",         # 两个接触面摩擦系数取均值
            restitution_combine_mode="average",      # 弹性系数也取均值
            static_friction=1.0,                     # 静摩擦系数（地面）
            dynamic_friction=1.0,                    # 动摩擦系数（地面）
            restitution=0.0,                         # 弹性系数 0 = 无弹跳（完全非弹性）
        ),
        visual_material=sim_utils.MdlFileCfg(       # 地形视觉材质（仅影响渲染，不影响物理）
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,                        # 投影 UV 坐标（使纹理平铺）
            texture_scale=(0.25, 0.25),              # 纹理缩放比例
        ),
        debug_vis=False,                             # 不显示地形调试可视化
    )

    # ── 机器人 ────────────────────────────────────────────────────
    robot: ArticulationCfg = BOYING_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # 将 BOYING_CFG 中的 prim_path 替换为每个 env 实例的路径（ENV_REGEX_NS 是 env 级别的命名空间）

    # ── 大范围高度扫描传感器（用于 Critic 特权观测）──────────────────
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",       # 射线从 base link 发射
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),  # 从 base 上方 20m 垂直向下射（避免穿模）
        ray_alignment="yaw",                         # 射线网格随 yaw 旋转（跟随机器人朝向）
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        # 1.6m×1.0m 网格，间距 0.1m → 16×10+多1行 = 187 个射线点（Critic 特权地形信息）
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],           # 只对地形 mesh 进行射线检测
    )

    # ── 小范围高度扫描传感器（用于 base_height_l2 奖励估计地面高度）───
    height_scanner_small = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.4, 0.3]),
        # 0.4m×0.3m 小网格 → 4×3+1 = 13 个点，用于估计机体正下方的地面高度
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # ── 接触力传感器（用于 undesired_contacts 奖励和 illegal_contact 终止）
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",         # 监测机器人所有 link 的接触力
        history_length=3,                            # 保留最近 3 步的接触力历史（用于检测持续接触）
        track_air_time=True,                         # 记录足部离地时间（供步态奖励使用）
    )

    # ── 环境灯光 ──────────────────────────────────────────────────
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,                         # 天空光强度（流明），提供均匀环境光
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
            # 使用 HDR 天空贴图（晴天 4K）提供真实环境光照
        ),
    )


@configclass
class CommandsCfg:
    """速度指令配置：定义给机器人发送的目标速度命令格式。"""
    base_velocity = mdp.Go2RLGymCommandCfg()
    # 使用 Go2RLGymCommand 命令生成器：
    # - 初始范围: lin_vel_x/y = ±0.5 m/s, ang_vel_yaw = ±1.0 rad/s
    # - it=20000: 扩展到 ±1.0 m/s（与 Go2 对齐）
    # - it=50000: 扩展到 ±2.0 m/s（最终最大速度）
    # - 零命令概率: 0→10%, 0~1500it（让机器人学会静止站立）


@configclass
class ActionsCfg:
    """动作空间配置：Policy 输出 12 维关节位置目标偏移。"""
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",                          # 作用于名为 "robot" 的资产
        joint_names=JOINT_NAMES,                     # 覆盖所有 12 个关节
        scale={".*_hip_joint": 0.25, "^(?!.*_hip_joint).*": 0.25},
        # 动作缩放：policy 输出 [-1,1] → 实际关节偏移 [-0.25, 0.25] rad（所有关节统一 0.25）
        use_default_offset=True,                     # 动作是相对于 URDF 默认关节角的偏移（不是绝对值）
        clip={".*": (-100.0, 100.0)},               # 动作截断（实际不会触发，仅防止数值溢出）
        preserve_order=True,                         # 保持关节顺序与 JOINT_NAMES 一致
    )


@configclass
class ObservationsCfg:
    """观测空间配置：包含 Policy、Critic、SingleObs 三组观测。"""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy 观测组（学生/部署时使用）：只含可直接测量的量 + 历史。
        
        维度：6项 × (3+3+3+12+12+12)=45维，history_length=10 → 450维展平输入
        """
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,                   # 机体角速度 [rad/s]，3维（roll/pitch/yaw rate）
            noise=Unoise(n_min=-0.2, n_max=0.2),     # 加性均匀噪声 ±0.2 rad/s（模拟 IMU 噪声）
            clip=(-100.0, 100.0),                    # 截断（防溢出）
            scale=0.25,                              # 缩放：角速度 × 0.25（归一化到合理范围）
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,              # 重力在机体坐标系的投影，3维（感知姿态倾斜）
            noise=Unoise(n_min=-0.05, n_max=0.05),   # 加性噪声 ±0.05（模拟 IMU 重力测量误差）
            clip=(-100.0, 100.0),
            scale=1.0,                               # 不缩放（模值为 1，已归一化）
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,             # 当前速度命令 [vx, vy, ω_z]，3维
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,                               # 不缩放（命令直接传入 policy）
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,                  # 12个关节相对默认位置的角度偏差 [rad]
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-0.03, n_max=0.03),  # 加性噪声 ±0.03 rad（模拟编码器噪声）
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,                  # 12个关节速度 [rad/s]
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-2.0, n_max=2.0),    # 加性噪声 ±2.0 rad/s（关节速度噪声较大）
            clip=(-100.0, 100.0),
            scale=0.05,                              # 缩放：速度 × 0.05（降低数值量级，辅助网络训练）
        )
        actions = ObsTerm(
            func=mdp.last_action,                    # 上一步 policy 输出的动作，12维（提供动作历史）
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.history_length = 10        # 保留 10 帧历史 → 观测维度 = 45 × 10 = 450
            self.enable_corruption = True   # 训练时开启噪声（eval/play 时可关闭）
            self.concatenate_terms = True   # 将所有项拼接成单个向量
            self.flatten_history_dim = True # 将历史维度展平（[T, D] → [T×D]）

    @configclass
    class CriticCfg(ObsGroup):
        """Critic/Teacher 观测组（训练时使用）：包含特权信息（真实速度/接触力/高度扫描）。
        
        这些量在真实部署时无法直接获取，只在仿真训练中用于 Critic 和 Teacher Encoder。
        """
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,                   # 机体真实线速度 [m/s]，3维（特权：IMU无法直接测）
            clip=(-100.0, 100.0),
            scale=2.0,                               # 缩放 × 2（放大线速度信号，提高 Critic 敏感度）
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,                   # 机体角速度（Critic 用，无噪声版本）
            clip=(-100.0, 100.0),
            scale=0.25,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,              # 重力投影（Critic 用，无噪声）
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,             # 当前速度命令（Critic 知道目标）
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,                  # 关节位置（Critic 用，无噪声）
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,                  # 关节速度（Critic 用，无噪声）
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        actions = ObsTerm(
            func=mdp.last_action,                    # 上一步动作（Critic 用）
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_acc = ObsTerm(
            func=mdp.joint_acc,                      # 关节加速度 [rad/s²]，12维（特权：高频噪声大）
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1e-4,                              # 缩放极小（加速度数值很大，需压缩）
        )
        joint_torque = ObsTerm(
            func=mdp.joint_effort,                   # 关节力矩 [N·m]，12维（特权：电机内部量）
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=0.01,                              # 缩放 × 0.01（力矩最大 60Nm，压缩到 ±0.6 范围）
        )
        contact_force = ObsTerm(
            func=mdp.foot_contact_force_norm,        # 4个足部接触力范数 [N]（特权：区分摆动/支撑相）
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_LINK_NAME)},
            clip=(-100.0, 100.0),
            scale=1e-3,                              # 缩放 × 1e-3（接触力可达数百N，压缩到 ±0.5）
        )
        height_scan = ObsTerm(
            func=mdp.height_scan,                    # 大范围高度扫描，187维（特权：提前感知地形起伏）
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),                        # 截断到 ±1m（过深/过高的值不重要）
            scale=2.5,                               # 缩放 × 2.5（高度差通常 <0.4m，放大到接近 1）
        )

        def __post_init__(self):
            self.enable_corruption = False   # Critic 观测不加噪声（使用真实特权信息）
            self.concatenate_terms = True

    @configclass
    class SingleObsCfg(PolicyCfg):
        """单步观测组（history_length=1）：供 Actor 使用的当前时刻 Policy 观测。
        
        MoE-CTS 中，Actor 接受 [latent(32) + single_obs(45)] 作为输入，
        其中 latent 来自编码器（包含历史），single_obs 提供当前时刻的直接感知。
        """
        def __post_init__(self):
            super().__post_init__()
            self.history_length = 1          # 只保留当前帧（45维，无历史展开）

    policy: PolicyCfg = PolicyCfg()          # Policy 观测（学生/部署）：450维历史
    critic: CriticCfg = CriticCfg()          # Critic 观测（训练特权）：含地形扫描等特权信息
    single_obs: SingleObsCfg = SingleObsCfg()  # 单步观测（Actor 输入一部分）：45维当前帧


@configclass
class EventCfg:
    """域随机化事件配置：在训练中随机化机器人物理参数，提升 Sim-to-Real 迁移能力。"""

    # ── startup 类事件：仿真启动时执行一次，整个训练周期固定 ──────────

    randomize_rigid_body_mass_base = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",                              # 仿真启动时执行一次（每个 env 独立随机）
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_LINK_NAME),
            "mass_distribution_params": (-1.0, 1.0), # base 质量加减 ±1.0 kg（均匀分布）
            "operation": "add",                       # 加法操作（在原质量上增减）
            "recompute_inertia": True,                # 重新计算惯量（保持物理一致性）
        },
    )
    randomize_rigid_body_mass_others = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="^(?!.*base).*"),
            # 正则：匹配所有非 base 的 link（hip/thigh/calf/foot）
            "mass_distribution_params": (0.9, 1.1),  # 质量缩放至原来的 90%~110%（均匀分布）
            "operation": "scale",                     # 乘法操作（相对缩放）
            "recompute_inertia": True,
        },
    )
    randomize_com_positions = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_LINK_NAME),
            "com_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.03, 0.03)},
            # base 质心位置在 xyz 各偏移 ±30mm（模拟载荷偏心）
        },
    )
    randomize_rigid_body_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),  # 所有 link
            "static_friction_range": (0.0, 2.0),    # 静摩擦系数：0（光滑）~ 2.0（粗糙）
            "dynamic_friction_range": (0.0, 2.0),   # 动摩擦系数
            "restitution_range": (0.0, 0.5),         # 弹性系数：0（完全非弹性）~ 0.5
            "num_buckets": 64,                       # 将材质组合离散成 64 个桶（降低内存开销）
            "make_consistent": True,                  # 确保同一 env 中材质前后一致
        },
    )

    # ── reset 类事件：每次 episode reset 时执行 ──────────────────────

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",                                # 每次 episode reset 时执行
        params={
            "position_range": (0.8, 1.2),
            # 关节位置 = default × scale，scale ∈ [0.8, 1.2]
            # 避免 scale<0.617 时小腿超出上限 -0.9269 rad（已修复的 bug）
            "velocity_range": (0.0, 0.0),            # 关节速度始终初始化为 0
        },
    )
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),  # Kp 缩放至 90%~110%
            "damping_distribution_params": (0.9, 1.1),    # Kd 缩放至 90%~110%
            "operation": "scale",
            "distribution": "uniform",               # 均匀分布（非正态）
        },
    )
    randomize_motor_zero_offset = EventTerm(
        func=mdp.randomize_action_joint_pos_offset,
        mode="reset",
        params={
            "action_term_name": "joint_pos",         # 作用于名为 "joint_pos" 的动作项
            "offset_range": (-0.035, 0.035),
            # 给每个关节的动作加随机偏置 ±0.035 rad（~±2°）
            # 模拟电机零点误差，训练 policy 鲁棒性
            # 注意：play 时此项默认仍激活（可导致关节卡限位，见分析）
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5), "y": (-0.5, 0.5),  # 水平位置随机偏移 ±0.5m（在地形格内）
                "z": (0.0, 0.2),                       # 高度随机偏移 0~0.2m（从略高处落下）
                "yaw": (-3.14, 3.14),                  # 偏航角全范围随机（机器人朝向随机）
            },
            "velocity_range": {
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5),  # 初始线速度 ±0.5m/s
                "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5),  # 初始角速度 ±0.5rad/s
            },
        },
    )

    # ── interval 类事件：按时间间隔周期性执行 ────────────────────────

    randomize_push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 4.0),                 # 每 4 秒推一次（固定间隔，非随机）
        params={
            "velocity_range": {
                "x": (-0.4, 0.4), "y": (-0.4, 0.4),      # 水平冲击速度 ±0.4 m/s
                "roll": (-0.6, 0.6), "pitch": (-0.6, 0.6), "yaw": (-0.6, 0.6),
                # 旋转冲击 ±0.6 rad/s（模拟外力扰动，训练抗干扰能力）
            }
        },
    )


@configclass
class RewardsCfg:
    """奖励函数配置：所有奖励项的权重和参数。
    
    奖励 = Σ(weight × fn(env))，每步累加，episode 结束时记录总和。
    正奖励驱动目标行为，负奖励（惩罚）抑制不良行为。
    Boying 的权重基于与 Go2 的机械参数比例推导（扭矩/功率/足端速度等）。
    """

    # ── 正向奖励：跟踪速度命令 ───────────────────────────────────────
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.0,                                  # 最重要的正奖励，权重最高（线速度跟踪）
        params={"command_name": "base_velocity", "std": 0.35},
        # 奖励 = exp(-||v_cmd-v_actual||² / 0.35²) ∈ (0,1]
        # std=0.35（Go2 用 0.5）：更窄的高斯核，对速度误差更敏感
        # 基于 Boying 最大速度 ~1.5m/s 推导（σ应与可达速度匹配）
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,                                  # 偏航角速度跟踪，权重为线速度的一半
        params={"command_name": "base_velocity", "std": 0.5},
        # 奖励 = exp(-(ω_cmd-ω_actual)² / 0.5²)
    )

    # ── 稳定性惩罚 ────────────────────────────────────────────────
    lin_vel_z_l2 = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-2.0,
        # 惩罚垂直弹跳（vz² × weight）
        # 课程：it=0→1500 从 -2.0 线性衰减到 0（早期强约束，后期放开）
    )
    ang_vel_xy_l2 = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.1,
        # 惩罚俯仰/横滚角速度（ωx²+ωy²）
        # Boying 用 -0.1，Go2 用 -0.05：Boying 重心更高，需更强稳定约束
    )

    # ── 效率惩罚 ──────────────────────────────────────────────────
    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.0e-7,                              # 惩罚关节加速度（Σq̈²），极小权重
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES)},
    )
    joint_power = RewTerm(
        func=mdp.joint_power,
        weight=-1.0e-5,                              # 惩罚关节功率（Σ|τω|），Go2 用 -2e-5
        # 按峰值功率比缩放：Go2=315.9W, Boying=634.8W → 权重 × 0.5
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES)},
    )
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.5e-5,                              # 惩罚关节力矩平方和（Στ²），Go2 用 -1e-4
        # 按扭矩容量比缩放：(23.4/60)² = 0.152 → -1e-4 × 0.152 ≈ -1.5e-5
        # 避免过度惩罚 Boying 的大扭矩电机正常工作
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES)},
    )

    # ── 高度维持 ──────────────────────────────────────────────────
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-1.0,
        # 惩罚机体高度偏离目标（(h-0.35)²）
        # 课程：it=0→7000 从 -1.0 增强到 -8.5（比 Go2 的 -10.0 更温和）
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_LINK_NAME),
            "target_height": BASE_HEIGHT_TARGET,     # 目标高度 0.35m
            "sensor_cfg": SceneEntityCfg("height_scanner_small"),
            # 用小范围高度扫描估计地面高度（地形起伏下的真实离地高度）
        },
    )

    # ── 动作平滑惩罚 ──────────────────────────────────────────────
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,
        # 惩罚动作一阶变化率：Σ(aₜ-aₜ₋₁)²（防止动作急变）
    )
    action_smoothness_l2 = RewTerm(
        func=mdp.action_smoothness_l2,
        weight=-0.01,
        # 惩罚动作二阶变化（jerk）：Σ(aₜ-2aₜ₋₁+aₜ₋₂)²（防止动作抖动）
    )

    # ── 安全惩罚 ──────────────────────────────────────────────────
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        # 惩罚大腿/小腿触地（应该只有足部接触地面）
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh|.*_calf"), "threshold": 5.0},
        # 接触力 > 5N 才算（避免轻微擦碰误触发）
    )
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-2.0,
        # 惩罚关节超出软限位（soft_joint_pos_limit_factor=0.95 处开始惩罚）
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES)},
    )

    # ── 步态质量惩罚 ──────────────────────────────────────────────
    feet_regulation = RewTerm(
        func=mdp.feet_regulation,
        weight=-0.16,                                # Go2 用 -0.05；Boying 按足速比缩放 ×3.17
        # 惩罚足部在近地时的水平速度（防止拖地）
        # 公式：Σ ||v_foot_xy||² × exp(-h_foot / 0.025×h_target)
        # 足部越靠近地面，拖动惩罚越大；离地越高，惩罚趋近于零
        params={
            "base_height_target": BASE_HEIGHT_TARGET,
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_LINK_NAME),
            "sensor_cfg": SceneEntityCfg("height_scanner_small"),
        },
    )
    hip_pos_penalty_l1 = RewTerm(
        func=mdp.hip_pos_penalty_l1,
        weight=-0.077,                               # Go2 用 -0.05；按髋关节范围比缩放 ×1.54
        # Boying 髋关节范围 ±39°，比 Go2 ±60° 窄，每 rad 偏差代价更大
        # 低速/横向/转向命令时，惩罚髋关节偏离默认位置
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint"),
            "stand_still_scale": 1.0,               # 静止时惩罚系数（与运动时相同）
            "command_threshold": 0.1,               # |cmd_y|或|cmd_yaw|>0.1 rad/s 时认为在横/转
        },
    )
    joint_pos_penalty_l1 = RewTerm(
        func=mdp.joint_pos_penalty_l1,
        weight=-0.01,
        # 惩罚大腿/小腿关节偏离默认位置（L1范数），引导姿态整洁
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_(thigh|calf)_joint"),
            "stand_still_scale": 1.0,
            "velocity_threshold": 0.1,              # 机体速度 < 0.1 m/s 认为静止
            "command_threshold": 0.1,
        },
    )


@configclass
class TerminationsCfg:
    """终止条件配置：定义 episode 提前结束的条件。"""

    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True,
        # 超时终止：达到最大 episode 时长（episode_length_s=25s → 1250步）时终止
        # time_out=True 表示这是"自然结束"而非"失败"（Bootstrap 方式不同）
    )
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=BASE_LINK_NAME),
            # 监测 base（机体主体）的接触力
            "threshold": 5.0,
            # 机体接触力超过 5N 立即终止（base 不应触地）
            # Boying 用 5N，比 Go2 的 1N 宽松：重型机器人倒地冲击更大，避免误触发
        },
    )


@configclass
class CurriculumCfg:
    """课程学习配置：训练过程中动态调整难度和奖励权重。"""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel_gym)
    # 地形难度课程：每次 episode reset 时评估
    # - 升级条件：episode 内最大移动距离 > terrain_length/2 = 4m
    # - 降级条件：实际距离 < 期望距离的 50%
    # - 地形共 10 行（level 0~9），max_init_terrain_level=5 限制初始最高难度
    # - 返回值（均值地形等级）记录到日志 "Curriculum/terrain_levels"

    base_linear_velocity = CurrTerm(mdp.gradual_reward_weight_modification, params={
        "term_name": "lin_vel_z_l2",
        "initial_weight": -2.0,    # 初始权重：强力抑制垂直弹跳
        "final_weight": -0.0,      # 最终权重：完全关闭（由 base_height_l2 接管高度控制）
        "start_it": 0,
        "end_it": 1500,            # 1500 iteration 内线性衰减
        # 设计意图：训练早期需要抑制弹跳帮助机器人建立平衡；
        # 之后 base_height_l2 的高度惩罚已足够约束垂直稳定性
    })

    base_height_l2 = CurrTerm(mdp.gradual_reward_weight_modification, params={
        "term_name": "base_height_l2",
        "initial_weight": -1.0,    # 初始权重较小（避免早期与探索冲突）
        "final_weight": -8.5,      # 最终权重：较强的高度约束
        # Go2 用 -10.0，Boying 按高度目标比例缩放：-10 × (0.35/0.38)² ≈ -8.5
        "start_it": 0,
        "end_it": 7000,            # 7000 iteration 内线性增强（比 Go2 的 5000 更慢）
    })


@configclass
class BoyingEnvCfg(ManagerBasedRLEnvCfg):
    """Boying 机器人粗糙地形强化学习环境总配置。
    
    将场景、观测、动作、命令、奖励、终止、事件、课程各子配置组合为完整环境。
    """

    # ── 各子配置实例化 ────────────────────────────────────────────
    scene: BoyingSceneCfg = BoyingSceneCfg(num_envs=16384, env_spacing=0.5)
    # 16384 个并行环境，间距 0.5m（env 间距决定地形格大小分配）
    observations: ObservationsCfg = ObservationsCfg()   # Policy/Critic/SingleObs 三组观测
    actions: ActionsCfg = ActionsCfg()                   # 12维关节位置动作
    commands: CommandsCfg = CommandsCfg()                # 速度命令生成器
    rewards: RewardsCfg = RewardsCfg()                   # 所有奖励项
    terminations: TerminationsCfg = TerminationsCfg()    # 终止条件
    events: EventCfg = EventCfg()                        # 域随机化事件
    curriculum: CurriculumCfg = CurriculumCfg()          # 课程学习

    def __post_init__(self):
        """环境初始化后处理：设置仿真参数和传感器更新频率。"""

        # ── 控制频率 ──────────────────────────────────────────────
        self.decimation = 4          # 每 4 个物理步执行一次 Policy（控制频率 = 1/(0.005×4) = 50Hz）
        self.episode_length_s = 25.0  # 每个 episode 最长 25 秒 = 1250 步（policy 步）

        # ── 物理仿真参数 ──────────────────────────────────────────
        self.sim.dt = 0.005          # 物理步长 0.005s = 200Hz（高频物理，精确接触模拟）
        self.sim.render_interval = self.decimation  # 渲染间隔 = 控制间隔（每4物理步渲染一次）

        # 将地形物理材质应用到全局仿真（确保地面摩擦系数与地形一致）
        self.sim.physics_material = self.scene.terrain.physics_material

        # ── PhysX GPU 内存配置 ────────────────────────────────────
        self.sim.physx.gpu_max_rigid_patch_count = int(1 * 1024 * 1024)
        # GPU 碰撞 patch 数量上限（1M），足够 16384 × 多腿机器人的接触计算
        self.sim.physx.gpu_collision_stack_size = int(512 * 1024 * 1024)
        # GPU 碰撞栈大小（512MB），防止复杂接触时内存溢出
        self.sim.physx.enable_external_forces_every_iteration = True
        # 每次物理迭代都施加外力（确保推力随机化的实时性）

        # ── 传感器更新频率 ────────────────────────────────────────
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
            # 大范围高度扫描：每个 policy 步更新一次（与控制同频，0.02s）
        if self.scene.height_scanner_small is not None:
            self.scene.height_scanner_small.update_period = self.decimation * self.sim.dt
            # 小范围高度扫描：同上
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
            # 接触力传感器：每个物理步更新（0.005s，高频采样确保接触检测准确）

        # ── 地形课程开关 ──────────────────────────────────────────
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
                # 启用地形课程：地形生成时按难度分行排列（row 0 = 最简单）
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
                # 若课程被禁用（如 play 时），地形随机分布不按难度排列
