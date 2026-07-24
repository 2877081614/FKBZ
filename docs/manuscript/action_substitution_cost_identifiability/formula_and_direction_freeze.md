# N/E 方向与成本公式冻结

更新时间：2026-07-24  
状态：W1-01 冻结  
实现锚点：`rein_learning/common/action_substitution_confirmation.py::_cost_ledger`

## 1. 分支身份与方向

| 分支 | 当前探针动作 | 固定身份 |
| --- | --- | --- |
| \(N\) | no-op / 不交战 | 对照分支 |
| \(E\) | 指定合法交战动作 | 干预分支 |

全文只允许以下方向：

\[
\Delta C_{\mathrm{episode}}
:= C_{\mathrm{episode}}(E)-C_{\mathrm{episode}}(N).
\]

正的 \(\Delta C_{\mathrm{episode}}\) 表示当前交战使回合累计资源成本更高。所有
“替代量”采用相反的节省方向 \(N-E\)：正值表示 E 分支替代了 N 分支中原本
会发生的消耗。

## 2. 当前步与未来成本

设干预发生在决策步 \(t\)，探针单元为 \(i\)。当前步成本拆分为探针和其他
单元：

\[
C_t(B)=C_{t,i}(B)+C_{t,-i}(B),\qquad B\in\{N,E\}.
\]

严格晚于当前步的累计成本记为：

\[
C_{>t}(B)=C_{>t,i}(B)+C_{>t,-i}(B).
\]

探针直接成本定义为：

\[
C_{\mathrm{direct}}
:=C_{t,i}(E)-C_{t,i}(N).
\]

## 3. 三类替代成本

同一步其他单元替代：

\[
Sub_{\mathrm{cost,same}}
:=C_{t,-i}(N)-C_{t,-i}(E).
\]

未来探针替代：

\[
Sub_{\mathrm{cost,future,probe}}
:=C_{>t,i}(N)-C_{>t,i}(E).
\]

未来其他单元替代：

\[
Sub_{\mathrm{cost,future,other}}
:=C_{>t,-i}(N)-C_{>t,-i}(E).
\]

未来替代成本与总替代成本分别为：

\[
Sub_{\mathrm{cost,future}}
=Sub_{\mathrm{cost,future,probe}}
+Sub_{\mathrm{cost,future,other}},
\]

\[
\boxed{
Sub_{\mathrm{cost,total}}
=Sub_{\mathrm{cost,same}}
+Sub_{\mathrm{cost,future,probe}}
+Sub_{\mathrm{cost,future,other}}
}.
\]

## 4. 主恒等式

由上述定义直接得到：

\[
\boxed{
\Delta C_{\mathrm{episode}}
=C_{\mathrm{direct}}-Sub_{\mathrm{cost,total}}
}.
\]

展开形式为：

\[
\Delta C_{\mathrm{episode}}
=C_{\mathrm{direct}}
-Sub_{\mathrm{cost,same}}
-Sub_{\mathrm{cost,future,probe}}
-Sub_{\mathrm{cost,future,other}}.
\]

R2 的 `7,776` 条目标成本账本在该完整公式下最大分解误差为
\(8.88\times10^{-16}\)。仅使用未来替代项会漏掉同一步后缀替代，首轮有
`287/7,776` 条受影响，最大残差为 `2.0`。

## 5. 射击数、比率与符号掩盖

未来替代射击数只统计严格晚于干预步的射击：

\[
Sub_{\mathrm{shot}}
:=Shots_{>t}(N)-Shots_{>t}(E).
\]

它不含同一步其他单元动作，因此不得与总替代成本互换。

当 \(C_{\mathrm{direct}}>0\) 时：

\[
\rho_{\mathrm{sub}}
:=\frac{Sub_{\mathrm{cost,total}}}{C_{\mathrm{direct}}}.
\]

成本符号掩盖事件定义为：

\[
I_{\mathrm{mask}}
:=\mathbb{1}
\left[
C_{\mathrm{direct}}>0
\land
\Delta C_{\mathrm{episode}}\le 0
\right].
\]

解释规则：

| 条件 | 解释 |
| --- | --- |
| \(\rho_{\mathrm{sub}}<1\) | 替代成本不足以完全抵消当前直接成本 |
| \(\rho_{\mathrm{sub}}=1\) | 替代成本恰好抵消当前直接成本 |
| \(\rho_{\mathrm{sub}}>1\) | 替代成本超过当前直接成本，累计成本符号可被掩盖 |
| \(I_{\mathrm{mask}}=1\) | 累计成本差未显露正的当前直接成本 |

## 6. 统计单位冻结

- `ledger row`：一个上下文、重复、目标和目标条件概率下的一条精确成本账。
- `repeat`：同一上下文的一次共同随机数 N/E 配对。
- `context`：一个冻结策略状态、探针单元和干预槽位组合；上下文区间以重复为样本。
- `block`：场景、策略种子和槽位组合；block 区间以 context 为样本。
- `seed`：独立训练策略种子；R2 的新策略种子固定为 `17/18/19`。
- 目标动作采用条件于“发生交战”的目标概率进行精确边缘化；不得把目标账本行数当作独立 context 数。

## 7. 实现字段冻结

R2 代码中的 `sub_cost` 对应论文符号
\(Sub_{\mathrm{cost,total}}\)，`future_sub_cost` 对应
\(Sub_{\mathrm{cost,future}}\)。后续论文和图表必须使用长名称，不能依赖代码
中的历史短字段名。

任何公式变更必须同时：

1. 更新术语账本和本文件；
2. 复核 `_cost_ledger` 字段映射；
3. 在 `evidence_conflict_log.md` 登记原因；
4. 不得通过新增实验绕过定义冲突。
