# DST-08：DS-2 异质场景最小筛选

任务状态：`NOT_STARTED`  
训练授权：3×10k  
前置任务：DST-07=`PASSED`

## 1. 目标

判断 DS-TR 是否减少结构性塌缩并改善已冻结安全—资源表现，而不是仅降低策略
变化率。

## 2. 固定实验

```text
scenario: heterogeneity_pressure
budget: 10k
policy seeds: 8, 9, 10
baseline: 原 factorized engagement-target joint PPO
candidate: joint PPO + DS-TR
unit order: 012
```

优先复用同配置的冻结 baseline，不重复训练；只有配置或软件版本不可比时才允许
成对重跑，并在 manifest 中说明。

## 3. 系数冻结

- 仅用 DST-07 的冻结 batch 做损失量纲匹配；
- 主系数在正式运行前写入配置并锁定；
- 不根据 episode reward、安全或资源结果选择；
- 最多允许一个弱/强 smoke 检查数值稳定性；
- 不做网格搜索。

## 4. 主要判据

复用项目现有正式口径，并同时检查：

1. exact fallback 与结构合法性；
2. all-noop 不增加；
3. 高威胁泄漏不劣化，并有冻结安全指标改善；
4. 资源成本满足项目既有非劣门；
5. target 排序不下降；
6. 策略仍产生有益翻转，不是完全冻结；
7. 至少 2/3 种子方向一致，且不能由单一种子决定总体结论。

同时报告：

```text
KL
普通 flip rate
DS-weighted flip mass
engage→no-op / no-op→engage
engagement count distribution
prefix position breakdown
```

## 5. 失败解释

以下均为 `STOPPED`，不是调参理由：

- churn 降低但安全/资源无改善；
- all-noop 转为持续高交战；
- 只在一个种子有效；
- target 排序或资源成本显著恶化；
- 必须改变 DS 定义或新增模块才能展示收益。

## 6. 交付物

```text
results/air_defense_v1/dynamic_support_trust_region/dst_08_ds2_screening/
  experiment_config.json
  source_model_manifest.json
  runs.csv
  episodes.csv
  update_metrics.csv
  decision_summary.csv
  gate_summary.json
docs/experiments/air_defense_v1_ds_tr_v0_screening.md
```

## 7. 阶段出口

- `PASS`：只表示 DS-TR v0 值得做增量控制，不代表正式算法成立；
- `STOPPED`：保留“诊断阳性、简单干预阴性”的结果并停止扩展；
- 不得直接从 3×10k 跳到论文效果主张。

