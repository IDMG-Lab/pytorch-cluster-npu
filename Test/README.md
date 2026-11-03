# 测试与验收

在项目根目录执行 `source env_cann9.sh`，使用已配置的 910B 环境。

## 精度与性能结果

设备为 Ascend910B1，PyTorch 2.7.1+cpu、torch_npu 2.7.1.post4。八算子主矩阵 **4,315/4,315** 按既定规则通过，完整性能 **84/84** 达到相对性能 **≥0.1** 的要求。

| 算子 | 主矩阵通过/总数 | Float32 严格差异项 | 性能达标/总数 | 最低相对性能 |
|---|---:|---:|---:|---:|
| Nearest | [312/312](precision/results/nearest.json) | 1 | 11/11 | 0.1046 |
| Graclus | [145/145](precision/results/graclus.json) | 0 | 11/11 | 0.1094 |
| KNN | [936/936](precision/results/knn.json) | 6 | 10/10 | 0.3274 |
| FPS | [936/936](precision/results/fps.json) | 32 | 11/11 | 0.4394 |
| Radius | [880/880](precision/results/radius.json) | 16 | 9/9 | 0.3960 |
| Grid | [888/888](precision/results/grid_cluster.json) | 0 | 13/13 | 0.1813 |
| Random Walk | [82/82](precision/results/random_walk.json) | 0 | 11/11 | 0.1058 |
| Subgraph | [136/136](precision/results/subgraph.json) | 0 | 8/8 | 0.5899 |
| 合计 | **4,315/4,315** | **55** | **84/84** | **0.1046** |

Nearest 的 312 项包含 288 项精度比较和 24 项输入校验。最低相对性能取该算子各场景中 `pyg_ms / npu_ms` 的最小值。

| 补充验证 | 通过/总数 | 结果位置 |
|---|---:|---|
| Radius Int32 完整补测 | 220/220 | [radius_int32.json](precision/results/radius_int32.json) |
| 精度及边界回归 | 327/327 | [regression.xml](precision/results/regression.xml) |
| 同输入与结构控制记录 | 7/7 | [controls.json](precision/results/controls.json) |
| 八算子冒烟检查 | 8/8 | [verification.json](verification.json) |
| 九个工作进程的用例探测 | 150/150 | [verification.json](verification.json) |
| 参考实现及接口测试 | 14/14 | [verification.json](verification.json) |

**结果来源**：完整精度主矩阵和 Int32 补测沿用已验收记录；目录迁移后重新执行其余补充检查，并在原项目通过 `Test/performance/run.py` 重新测量全部 84 个性能场景。当前运行实现及原生二进制与已验收版本核对一致。

**精度口径**：55 项 Float32 严格索引、序列或边界差异按距离质量与整数结构业务规则通过，对应距离指标的 matched ratio 均为 100%；不代表所有索引与独立 CPU 参考完全一致。详细记录见 [strict_differences.json](precision/results/strict_differences.json)。当前范围不宣称 CANN 所有有效组合全覆盖。

**性能口径**：相对性能为 `pyg_ms / npu_ms`，逐场景要求 ≥0.1，即 NPU 耗时至多为基线的 10 倍；20 次预热、100 次迭代，计时前后同步，包含公开接口开销。本轮最低场景为 Nearest：`x=[4096,3]`、`y=[1024,3]`，CUDA 基线 **0.044901 ms**，NPU **0.429166 ms**，相对性能 **0.104624**。

完整记录：[精度汇总](precision/results/summary.json)、[性能汇总](performance/results/summary.json)、[84 项性能明细](performance/results/cases.json)。

## 目录与入口

```text
Test/
├── unit/                 通用单元测试、接口测试
├── precision/            精度入口、用例、回归、参考输入和结果
├── performance/          性能入口和结果
├── benchmark/            pyg-lib 对照脚本与基线数据
└── specification.docx    原始验收规格
```

| 验收内容 | 入口 | 结果 |
|---|---|---|
| 八算子主矩阵、Int32 补测 | `precision/run.py --suite full` | `precision/results/summary.json`、各算子 JSON |
| 327 项回归及 7 条控制记录 | `precision/run.py --suite regression` | `precision/results/regression.xml`、`controls.json` |
| 全部 84 个性能场景 | `performance/run.py` | `performance/results/summary.json`、`cases.json` |

## benchmark 的含义

`benchmark` 保存性能对照所需的**测试定义和 pyg-lib 基线**：

- `geometry.py`、`subgraph.py`：用户提供的 pyg-lib 脚本，定义 shape、参数和输入生成方式。性能入口复用这些定义调用 NPU 算子。
- `baseline/geometry_cuda.tsv`：pyg-lib 几何算子的 CUDA 耗时；`baseline/subgraph_cpu.tsv`：pyg-lib Subgraph 的 CPU 耗时，单位均为毫秒。
- `reference.md`：原始基准结果说明。

NPU 实际测量结果写入 `performance/results/`。几何算子与 CUDA 基线比较，Subgraph 与 CPU 基线比较；本轮未重新运行 pyg-lib。历史基线没有种子和输入哈希，无法确认随机输入字节相同。

## 运行命令

```bash
python -m pytest Test/unit
python Test/precision/run.py --suite regression --output Test/precision/results/regression_run
python Test/precision/run.py --suite full --output Test/precision/results/full_run
python Test/performance/run.py --precision-gate Test/precision/results/gate.json --output Test/performance/results/performance_run.json
```

完整精度输出目录须为空；脚本会核验当前源码和构建产物。`precision/cases/` 保存算子用例，`precision/regression/` 保存回归，`precision/scripts/` 保存判定与覆盖核验。

