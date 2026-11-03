# pytorch-cluster-npu

本项目为 pytorch_cluster 提供昇腾 NPU 适配，包括 Python 接口、PyTorch C++ 扩展和 Ascend C 内核。

```text
csrc/              CPU、CUDA、NPU 分发与算子实现
torch_cluster/     Python 接口
Test/              单元测试、精度验收、性能验收与基准数据
setup.py           Python 扩展构建
env_cann9.sh        当前服务器 CANN 9 环境入口
```

当前 910B 验收覆盖 `fps`、`grid_cluster`、`nearest`、`knn`、`radius`、`graclus`、`random_walk`、`subgraph`。完整范围、脚本和结果见 [Test/README.md](Test/README.md)。

## 使用已有环境

```bash
source env_cann9.sh
python -m pytest Test/unit
python Test/precision/run.py --suite regression --output Test/precision/results/regression_run
```

服务器环境为 CANN 9.0.0、PyTorch 2.7.1、torch_npu 2.7.1.post4。环境入口加载项目内的已验收扩展和原生库；运行产物位于被 Git 忽略的 `.precision_910b_runtime/`。

## 构建

在匹配的 PyTorch、torch_npu 和 CANN 开发环境中：

```bash
cd csrc/npu/impl/op_project
bash build.sh
# 安装生成的 .run 算子包，并 source 对应 vendors/.../bin/set_env.bash
cd ../../../..
pip install -e . --no-build-isolation
```

`csrc/npu/impl/op_project/` 将原生算子打包到同一个 OPP 包，提供 `libcust_opapi.so` 中的 aclnn 接口。修改原生内核后须重新构建算子包和相关扩展。

Subgraph 的接口分别为 `subgraph_910b` 和 `subgraph_950`，`subgraph` 按设备选择后端。950 使用独立 SIMT 内核，构建与接口说明见 [csrc/npu/950/README.md](csrc/npu/950/README.md)。910B 上的验收不代表已执行 950 硬件测试。

## 精度与性能验收

`Test/precision/` 保存完整精度入口、用例和结果；`Test/performance/` 保存完整性能入口和结果。

`Test/benchmark/` 保存 pyg-lib 的**对照测试定义及基线耗时**：`geometry.py`、`subgraph.py` 定义输入与参数，`baseline/` 中的 TSV 提供 CUDA 几何算子及 CPU Subgraph 基线。NPU 实测数据写入 `Test/performance/results/`，按 `pyg_ms / npu_ms >= 0.1` 逐项验收。

完整命令、结果范围及浮点判定说明见 [测试说明](Test/README.md)。
