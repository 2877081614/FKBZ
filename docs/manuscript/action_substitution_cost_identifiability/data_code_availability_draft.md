# W1-10 Data and Code Availability 草稿

更新时间：2026-07-28  
目标期刊：未指定  
公共仓库标识：无  
许可证：未指定  
状态：M2 内部真实声明，可用于后续投稿准备，但不是公共发布承诺

## 1. 材料清单

| 材料 | 当前项目位置 | 访问路线 | 状态 |
| --- | --- | --- | --- |
| AirDefense v1 环境和 wrapper | `rein_learning/envs/air_defense_v1/` | 本地项目仓库 | 已存在 |
| factorized joint PPO | `rein_learning/algorithms/`、`rein_learning/models/` | 本地项目仓库 | 已存在 |
| 成对反事实审计实现 | `rein_learning/common/action_substitution_confirmation.py` | 本地项目仓库 | 已存在 |
| 正式运行入口 | `scripts/run_air_defense_v1_action_substitution_confirmation.py` | 本地项目仓库 | 已存在 |
| 配置和门控 | `results/air_defense_v1/action_substitution_confirmation/*.json` | 本地项目仓库 | 已存在 |
| 来源模型和哈希 | `results/.../source_models/`、`source_model_manifest.json` | 本地项目仓库 | 已存在 |
| 账本和派生表 | `results/.../*.csv` | 本地项目仓库 | 已存在 |
| 首轮无效账本归档 | `results/.../pre_ledger_correction/` | 本地项目仓库 | 已存在 |
| 图表 source data | `docs/manuscript/.../figures/source/` | 本地项目仓库 | 已存在 |
| 图表导出和 metadata | `docs/manuscript/.../figures/`、`tables/` | 本地项目仓库 | 已存在 |
| 复现映射 | `reproducibility_map.md` | 本地项目仓库 | 已存在 |

本研究未使用需要另行申请的个人、临床或第三方受限数据。现有数据均为项目仿真
产生的配置、模型、轨迹账本和确定性派生图表数据。

## 2. Current truthful statement (English)

### Data Availability

The simulation configurations, model manifests, paired counterfactual ledger records,
derived summary tables, and figure source data supporting this study are maintained in the
project repository and are mapped to the manuscript through the accompanying reproducibility
and traceability files. At the W1-10 manuscript freeze, these materials have not been deposited
in a public repository and no DOI or accession identifier has been assigned. This manuscript
therefore does not claim public data availability. A public or controlled-access release route,
versioned archive, and licence must be specified before external submission.

### Code Availability

The AirDefense v1 environment, factorized joint PPO implementation, paired counterfactual audit,
analysis entry points, and deterministic figure-generation code are maintained in the project
repository. At the W1-10 manuscript freeze, the code has no public repository identifier or
declared software licence. This manuscript therefore does not claim public code availability.
The release scope, version, licence, and persistent identifier must be fixed before external
submission.

## 3. 中文核对

### 数据可用性

支撑本研究的仿真配置、模型清单、成对反事实账本、派生汇总表和图表源数据均保存在
当前项目仓库，并通过复现映射和稿件追溯文件与正文对应。截至 W1-10 冻结时，
这些材料尚未存入公共仓库，也没有 DOI 或 accession identifier。因此当前稿件
不声称数据已公开。外部投稿前必须确定公开或受控访问路线、版本化归档和许可证。

### 代码可用性

AirDefense v1 环境、factorized joint PPO、成对反事实审计、分析入口和确定性
制图代码均保存在当前项目仓库。截至 W1-10 冻结时，代码尚无公共仓库标识或正式
软件许可证，因此当前稿件不声称代码已公开。外部投稿前必须冻结发布范围、版本、
许可证和持久标识。

## 4. 外部投稿前的仓库动作

1. 决定代码、模型、账本和图表数据的公开范围；
2. 清除机器绝对路径、缓存、临时模型和无关历史实验；
3. 为代码和数据分别选择适当许可证；
4. 建立版本化发布包，并记录 commit/tag；
5. 选择持久仓库并获取真实 DOI 或 accession；
6. 在干净环境中复核安装、R3 审计和图表重建；
7. 将最终标识填入 Data/Code Availability 和数据引用。

## 5. FAIR 与风险审计

| 维度 | 当前状态 | 风险 |
| --- | --- | --- |
| Findable | 无公共标识 | 外部读者无法发现 |
| Accessible | 仅本地项目 | 不能声称公开 |
| Interoperable | JSON/CSV/SVG/PDF 等开放格式为主 | 模型 zip 和环境依赖需说明 |
| Reusable | 有配置、哈希、追溯和复现映射 | 缺许可证、版本和独立发布 README |

不采用无理由的 “available upon reasonable request”，也不虚构未来 DOI、仓库名、
发布日期或访问审批流程。
