# DST-07：DS-TR v0 实现与精确回退

任务状态：`NOT_STARTED`  
训练授权：仅 unit/integration smoke  
前置任务：DST-06=`PASSED`

## 1. 目标

在现有 factorized engagement-target joint PPO 上实现一个且仅一个新干预：
限制早期前缀策略更新造成的动态支持域扰动。

## 2. 冻结算法形式

保持原 joint PPO surrogate 不变。对旧策略 \(\pi_{\text{old}}\) 和新策略
\(\pi_\theta\)，在 rollout 早期前缀上计算支持域变化量
\(D_{\mathrm{DS}}(\pi_\theta,\pi_{\text{old}})\)，优化：

\[
\max_\theta L_{\mathrm{PPO}}(\theta)
\quad\text{s.t.}\quad
D_{\mathrm{DS}}\le\delta_{\mathrm{DS}}.
\]

v0 必须使用 DST-01 冻结的“Jaccard 动作对代价 + 加权总变差”策略距离，可用
固定惩罚实现约束，但必须满足：

- \(D_{\mathrm{DS}}(\pi_{\text{old}},\pi_{\text{old}})=0\)；
- 只依赖动作概率与精确后缀集合；
- 不读取 reward、advantage、Q 或 BPCE 标签来定义距离；
- 只作用于存在下游级联的前缀位置；
- 不修改 target 条件分布结构；
- DS 系数为 0 时逐项退化为原 joint PPO。

DST-01 必须已冻结 v0 的精确离散计算形式。实现阶段不得在多个公式间按 smoke
表现选择。

## 3. 建议代码

```text
rein_learning/algorithms/policy_gradient/dynamic_support_trust_region_ppo.py
tests/test_dynamic_support_trust_region_ppo.py
```

尽量继承或薄包装：

```text
rein_learning/algorithms/policy_gradient/factorized_engagement_ppo.py
```

不得复制整套 PPO 训练器形成长期分叉。

## 4. 必须测试

### 精确回退

`lambda_ds=0` 时：

- 相同 batch 的总 loss、policy loss、value loss、entropy 一致；
- 梯度与更新后参数在冻结数值容差内一致；
- 动作分布、joint log-prob 和评估动作一致；
- 随机数消费顺序一致。

### DS 梯度

- identical policy 的 DS 为 0；
- 人工提高高 DS 动作概率时惩罚增大；
- 相同后缀集合的动作交换不产生 DS 惩罚；
- 最后位置不产生 DS 梯度；
- DS 梯度有限，无 NaN/Inf。

### 结构

- 非法动作、冲突、过杀仍为零；
- 动态 mask 和 joint ratio 未被替换；
- 保存/加载模型保持一致。

## 5. 禁止

- 新 Critic；
- DS 值按威胁或 Q 加权；
- 自适应学习乘子；
- 新 entropy schedule；
- margin loss；
- BPCE、MCH、GradS；
- 为 smoke 结果改公式。

## 6. 交付物

```text
rein_learning/algorithms/policy_gradient/dynamic_support_trust_region_ppo.py
tests/test_dynamic_support_trust_region_ppo.py
results/air_defense_v1/dynamic_support_trust_region/dst_07_implementation/
  fallback_equivalence.json
  gradient_sanity.csv
  smoke_summary.json
  implementation_manifest.json
```

## 7. 验收

所有精确回退、梯度和结构测试通过后才进入 DST-08。工程测试失败属于
`BLOCKED`，不能用训练结果覆盖实现错误。
