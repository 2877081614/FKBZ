# AirDefenseResourceAssignmentEnv v1.0 架构图

更新时间：2026-07-10

本文档根据 [air_defense_rl_environment_model_design.md](D:/huang/Programs/防空编组/docs/air_defense_rl_environment_model_design.md) 绘制，用于指导 `AirDefenseResourceAssignmentEnv v1.0` 的环境实现。

## 1. 总体分层架构

```mermaid
flowchart TB
    ALG["强化学习算法层<br/>DQN / PPO / Maskable PPO<br/>后续 MAPPO / MADDPG / QMIX"]
    API["环境 API 层<br/>v1.0: Gymnasium Env<br/>后续: PettingZoo ParallelEnv"]

    subgraph ENV["AirDefenseResourceAssignmentEnv v1.0 环境核心"]
        CONFIG["Scenario Config<br/>场景参数 / 随机种子 / 最大目标数"]
        STATE["State Manager<br/>全局真实状态 S_t"]
        OBS["Observation Builder<br/>观测 O_t / padding / masks"]
        ACTION["Action Encoder & Mask<br/>MultiDiscrete 联合动作<br/>action_mask"]
        TRANS["Transition Simulator<br/>状态转移 P"]
        REWARD["Reward Evaluator<br/>区域损伤 / 拦截收益 / 资源成本"]
        METRICS["Metrics & Info<br/>damage / intercept_rate / leak_rate"]
        RENDER["Render<br/>文本或可视化战场状态"]
    end

    subgraph ENTITIES["场景实体层"]
        ZONES["Protected Zones<br/>关键区域<br/>position / value / damage"]
        TARGETS["Hostile UAV Targets<br/>来袭目标<br/>position / payload / threat<br/>target_zone / time_to_impact"]
        UNITS["Defense Units<br/>防御单元<br/>missile / laser / jammer / sensor"]
        TRACKS["Tracks<br/>航迹<br/>confidence / covariance / AoI"]
        REL["Resource-Target Relations<br/>距离 / 射程 / 成功概率 / 期望收益"]
    end

    subgraph SIM["仿真模型层"]
        MOTION["Target Motion Model<br/>目标运动与突防"]
        SENSOR["Sensor & Track Model<br/>探测 / 跟踪 / 观测噪声"]
        EFFECTOR["Effector Model<br/>拦截 / 命中概率 / 冷却 / 弹药"]
        JAMMER["Jammer Model<br/>压制 / 航向扰动 / 速度下降"]
        DAMAGE["Damage Model<br/>区域损伤 / 任务成功判定"]
    end

    ALG --> API
    API --> ENV
    CONFIG --> STATE
    STATE --> OBS
    OBS --> API
    API --> ACTION
    ACTION --> TRANS
    TRANS --> REWARD
    REWARD --> METRICS
    METRICS --> API
    RENDER --> API

    STATE --> ZONES
    STATE --> TARGETS
    STATE --> UNITS
    STATE --> TRACKS
    STATE --> REL

    TRANS --> MOTION
    TRANS --> SENSOR
    TRANS --> EFFECTOR
    TRANS --> JAMMER
    TRANS --> DAMAGE

    MOTION --> TARGETS
    SENSOR --> TRACKS
    EFFECTOR --> UNITS
    EFFECTOR --> TARGETS
    JAMMER --> TARGETS
    DAMAGE --> ZONES
```

## 2. 单步交互流程

```mermaid
flowchart TD
    START["reset(seed)<br/>初始化场景"] --> INIT["Scenario Generator<br/>生成关键区域 / 来袭目标 / 防御单元"]
    INIT --> S0["全局真实状态 S_0"]
    S0 --> OBS0["Observation Builder<br/>构造 O_0"]
    OBS0 --> MASK0["Action Mask<br/>构造合法动作掩码"]
    MASK0 --> AGENT["Agent<br/>集中式防空编组指挥器"]

    AGENT --> ACT["联合动作 A_t<br/>[a_1, a_2, ..., a_M]"]
    ACT --> DEC["Action Decoder<br/>解析每个防御单元动作"]
    DEC --> CHECK["Legality & Conflict Check<br/>射程 / 冷却 / 弹药 / 目标状态 / 重复分配"]

    CHECK --> EXEC["Effect Execution<br/>engage / jam / track / no-op"]
    EXEC --> HIT["Intercept & Jam Result<br/>命中采样 / 干扰效果 / 资源消耗"]
    HIT --> MOVE["Target Motion<br/>目标运动 / time_to_impact 更新"]
    MOVE --> SENSOR["Sensor & Track Update<br/>探测置信度 / AoI / 航迹误差"]
    SENSOR --> LEAK["Leak & Damage Check<br/>目标突防 / 区域损伤"]
    LEAK --> REWARD["Reward Calculation<br/>r_t"]
    REWARD --> TERM["Termination Check<br/>成功 / 失败 / max_steps"]
    TERM --> NEXT["返回<br/>O_{t+1}, reward, terminated, truncated, info"]
    NEXT --> AGENT
```

## 3. 状态、动作、奖励三大接口

```mermaid
flowchart LR
    subgraph STATE["状态空间 O_t"]
        ZF["zone_features<br/>区域位置 / 价值 / 损伤"]
        TF["target_features<br/>位置 / 速度 / 威胁 / 载荷<br/>目标区域 / 到达时间 / 航迹置信度"]
        UF["unit_features<br/>类型 / 弹药 / 能量 / 冷却<br/>射程 / 成本 / 可用性"]
        EF["edge_features<br/>距离 / in_range / 命中概率<br/>time_to_intercept / expected_benefit"]
        GF["global_features<br/>时间进度 / 存活比例 / 弹药比例 / 总损伤"]
    end

    subgraph ACTION["动作空间 A_t"]
        MD["MultiDiscrete 联合动作<br/>每个防御单元选择一个动作"]
        MODE["mode_i<br/>noop / track / engage / jam / hold"]
        TARGET["target_i<br/>目标编号或 null"]
        MASK["action_mask<br/>物理约束和资源约束"]
    end

    subgraph REWARD["奖励函数 r_t"]
        RI["R_intercept<br/>拦截高威胁目标"]
        RJ["R_jam / R_track<br/>压制和航迹维护"]
        RP["R_protect<br/>保护关键区域"]
        CD["C_resource / C_time<br/>资源和时间成本"]
        PD["P_damage / P_leak<br/>区域损伤和突防惩罚"]
        PC["P_invalid / P_conflict / P_overkill<br/>非法动作 / 冲突 / 过度分配"]
    end

    STATE --> ACTION
    ACTION --> REWARD
```

## 4. 建议代码模块映射

```mermaid
flowchart TB
    subgraph ENVPKG["rein_learning/envs/air_defense_v1/"]
        CONFIG["config.py<br/>环境参数与默认场景"]
        ENTITIES["entities.py<br/>ProtectedZone / Target / DefenseUnit / Track"]
        CENV["centralized_env.py<br/>Gymnasium 单智能体集中式环境"]
        MAENV["multi_agent_env.py<br/>后续 PettingZoo 多智能体环境"]
        OBSB["observation_builder.py<br/>向量观测 / padding / masks"]
        ACTE["action_encoder.py<br/>MultiDiscrete 编码与解码"]
        MASKS["masks.py<br/>合法动作掩码"]
        REW["reward.py<br/>奖励分项与终局奖励"]
        SCENE["scenario_generator.py<br/>目标波次与区域生成"]
    end

    subgraph SIMPKG["rein_learning/simulators/air_defense/"]
        TM["target_motion.py<br/>目标运动与 time_to_impact"]
        SM["sensor_model.py<br/>探测概率与观测噪声"]
        TRM["track_model.py<br/>航迹置信度 / covariance / AoI"]
        EM["effector_model.py<br/>拦截概率 / 弹药 / 冷却"]
        JM["jammer_model.py<br/>干扰效果"]
        DM["damage_model.py<br/>突防与区域损伤"]
    end

    subgraph EXP["实验与测试"]
        BASE["baselines<br/>规则策略 / Hungarian / CBBA"]
        TRAIN["trainers<br/>PPO / Maskable PPO / 后续 MAPPO"]
        SCRIPTS["scripts<br/>训练 / 评估 / 方法对比"]
        TESTS["tests<br/>env / reward / masks / scenario"]
    end

    CENV --> OBSB
    CENV --> ACTE
    CENV --> MASKS
    CENV --> REW
    CENV --> SCENE
    CENV --> TM
    CENV --> SM
    CENV --> TRM
    CENV --> EM
    CENV --> JM
    CENV --> DM

    MAENV -. "v1.4 扩展" .-> OBSB
    MAENV -. "v1.4 扩展" .-> ACTE
    BASE --> CENV
    TRAIN --> CENV
    SCRIPTS --> TRAIN
    TESTS --> CENV
```

## 5. v1.0 最小实现边界

```mermaid
flowchart LR
    V0["当前 v0<br/>单个 unit-target 动作<br/>单保护目标<br/>拦截奖励"]
    V10["v1.0 MVP<br/>多保护区域<br/>目标绑定攻击区域<br/>payload / time_to_impact<br/>联合 MultiDiscrete 动作<br/>区域损伤奖励"]
    V12["v1.2<br/>探测 / 跟踪不确定性<br/>track_confidence / covariance / AoI"]
    V14["v1.4<br/>PettingZoo 多智能体<br/>局部观测 / 团队奖励 / MAPPO"]
    V15["v1.5<br/>图观测 / GNN<br/>资源-目标二部图"]

    V0 --> V10 --> V12 --> V14 --> V15
```

## 6. 一句话架构理解

`AirDefenseResourceAssignmentEnv v1.0` 的核心不是“模拟一枚导弹打一个目标”，而是把防空编组抽象为一个动态资源分配系统：环境维护目标、区域、资源和航迹状态；agent 输出多防御单元联合动作；仿真器执行运动、拦截、干扰和损伤转移；奖励函数以关键区域损伤最小化和资源效率为核心。
