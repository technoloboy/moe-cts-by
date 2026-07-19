from isaaclab.utils import configclass  # IsaacLab 的配置类装饰器，提供 dataclass 风格的配置管理
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg  # RSL-RL 基础配置类：Runner、Actor-Critic网络、PPO算法


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):  # 标准 PPO 训练运行器配置，作为 MoE-CTS 的对照基线
    num_steps_per_env = 24          # 每个环境每次 rollout 收集 24 步数据，即 rollout buffer 长度
    max_iterations = 300000         # 最大训练迭代次数（每次迭代 = 一次 rollout + 一次策略更新）
    save_interval = 500             # 每 500 次迭代保存一次 checkpoint
    experiment_name = "boying_rough"  # 实验名称，用于 wandb/tensorboard 日志目录命名

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,             # 动作高斯噪声初始标准差，控制探索强度
        actor_obs_normalization=False,  # 不对 actor 的输入观测做运行时归一化（已在观测层处理）
        critic_obs_normalization=False, # 不对 critic 的输入观测做运行时归一化
        actor_hidden_dims=[512, 256, 128],  # actor MLP 各隐层宽度，逐层压缩特征
        critic_hidden_dims=[512, 256, 128], # critic MLP 各隐层宽度，与 actor 结构对称
        activation="elu",               # 激活函数，ELU 对负值有平滑梯度，优于 ReLU
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,            # value loss 相对于 policy loss 的权重系数
        use_clipped_value_loss=True,    # 使用 PPO-style 的 value function clipping，提升训练稳定性
        clip_param=0.2,                 # PPO clip 系数 ε，限制策略更新幅度，防止过大步长
        entropy_coef=0.01,             # 熵正则系数，鼓励探索，防止策略过早收敛
        num_learning_epochs=5,          # 每次 rollout 后对同一批数据做 5 轮梯度更新
        num_mini_batches=4,             # 每轮 epoch 将 rollout buffer 切成 4 个 mini-batch
        learning_rate=1.0e-3,          # Adam 优化器学习率
        schedule="adaptive",            # 自适应学习率调度：KL 超标时降低 lr，不足时升高
        gamma=0.99,                    # 折扣因子，控制未来奖励的权重
        lam=0.95,                      # GAE-λ 系数，平衡 bias 和 variance 的优势估计
        desired_kl=0.01,               # 自适应调度的目标 KL 散度阈值
        max_grad_norm=1.0,             # 梯度裁剪上限，防止梯度爆炸
    )


@configclass
class RslRlMoeCtsActorCriticCfg(RslRlPpoActorCriticCfg):  # MoE-CTS 的 Actor-Critic 网络配置，扩展标准 PPO 网络
    class_name = "ActorCriticMoECTS"            # 指定实例化的网络类名，由 RSL-RL 框架按名查找注册表
    init_noise_std = 1.0                        # 动作噪声初始标准差，与标准 PPO 保持一致
    expert_num = 8                              # MoE 专家数量，每个专家是一个独立的策略子网络
    latent_dim = 32                             # Teacher encoder 输出的隐空间维度，表征地形/环境特征
    norm_type = 'l2norm'                        # 专家门控权重的归一化方式，L2 归一化保持权重在单位球面
    teacher_encoder_hidden_dims = [512, 256]    # Teacher encoder MLP 结构（特权观测 → latent），训练时可见地形信息
    student_encoder_hidden_dims = [512, 256, 256]  # Student encoder MLP 结构（历史本体感觉 → latent），部署时使用
    actor_hidden_dims = [512, 256, 128]         # Actor（策略头）MLP 各隐层宽度
    critic_hidden_dims = [512, 256, 128]        # Critic（价值头）MLP 各隐层宽度
    activation = "elu"                          # 所有 MLP 的激活函数
    actor_obs_normalization = False             # 不对 actor 输入做归一化
    critic_obs_normalization = False            # 不对 critic 输入做归一化


@configclass
class RslRlMoeCtsAlgorithmCfg(RslRlPpoAlgorithmCfg):  # MoE-CTS 算法配置，在标准 PPO 基础上新增 MoE 专有超参
    class_name = "MoECTS"                      # 指定算法类名，RSL-RL 按名实例化对应的训练逻辑
    value_loss_coef = 1.0                      # value loss 权重，与标准 PPO 相同
    load_balance_coef = 0.01                   # 专家负载均衡损失系数，防止所有样本塌陷到少数专家
    use_clipped_value_loss = True              # 使用 PPO value clipping
    clip_param = 0.2                           # PPO clip 系数 ε
    entropy_coef = 0.01                        # 熵正则系数
    num_learning_epochs = 5                    # 每次 rollout 后的梯度更新轮数
    num_mini_batches = 4                       # mini-batch 数量
    learning_rate = 1e-3                       # Actor/Critic/Teacher encoder 的学习率
    student_encoder_learning_rate = 1e-3       # Student encoder 单独的学习率（蒸馏阶段可独立调节）
    schedule = "adaptive"                      # 自适应学习率调度策略
    gamma = 0.99                               # 折扣因子
    lam = 0.95                                 # GAE-λ 系数
    betas = (0.9, 0.999)                       # Adam 优化器的一阶/二阶矩衰减系数
    weight_decay = 0.0                         # Adam 权重衰减（L2 正则），当前不启用
    desired_kl = 0.01                          # 自适应调度的目标 KL 散度
    max_grad_norm = 1.0                        # 梯度裁剪上限
    teacher_env_ratio = 0.75                   # 每次 rollout 中使用 teacher policy（特权信息）的环境比例，其余用 student


@configclass
class MoECTSRunnerCfg(RslRlOnPolicyRunnerCfg):  # Boying 机器人的 MoE-CTS 主训练配置，实际训练时使用此类
    experiment_name = "boying_moe_cts"          # 实验名称，对应日志/checkpoint 目录
    class_name = "OnPolicyRunnerCTS"            # 指定 Runner 类名，包含 CTS（Curriculum Teacher-Student）训练流程
    num_steps_per_env = 24                      # 每个环境每次 rollout 的步数
    max_iterations = 300000                     # 最大训练迭代次数
    save_interval = 500                         # checkpoint 保存间隔
    policy = RslRlMoeCtsActorCriticCfg()        # 使用 MoE-CTS Actor-Critic 网络配置
    algorithm = RslRlMoeCtsAlgorithmCfg()       # 使用 MoE-CTS 算法配置


@configclass
class MoECTSCatELURunnerCfg(MoECTSRunnerCfg):  # MoECTSRunnerCfg 的变体，将激活函数替换为 CatELU
    def __post_init__(self):
        super().__post_init__()                 # 先执行父类的初始化，确保所有字段正确设置
        self.policy.activation = 'cat_elu'      # 覆盖激活函数为 CatELU：对正负值分别做 ELU 再拼接，扩大表示容量
