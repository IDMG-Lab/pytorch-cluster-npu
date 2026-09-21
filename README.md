# pytorch-cluster-npu

**pytorch-cluster-npu** 是 [pytorch_cluster](https://github.com/rusty1s/pytorch_cluster) 的昇腾 NPU 适配版本。

在原有 CPU/CUDA 实现之外，本项目新增：

- **NPU 中间层**：PyTorch C++ Extension，按设备分发（CPU / CUDA / NPU），通过 `EXEC_NPU_CMD` 调用 CANN aclnn 算子；
- **Ascend C 算子内核**：将多个算子的 `op_host`（原型 + tiling）与 `op_kernel`（内核）合并到一个算子工程 `op_project` 中，编译后产出包含全部 aclnn 接口的算子包。

---

## 项目结构

```
pytorch-cluster-npu/
├── csrc/
│   ├── *.cpp                       # 算子分发层（CPU/CUDA/NPU 三路分发）
│   ├── cpu/                        # 原始 CPU 实现（来自 pytorch_cluster）
│   ├── cuda/                       # 原始 CUDA 实现（来自 pytorch_cluster）
│   └── npu/                        # NPU 适配层（本项目新增）
│       ├── include/
│       │   └── pytorch_npu_helper.hpp      # EXEC_NPU_CMD 宏与类型转换
│       ├── fps_npu.{h,cpp}                 # 各算子 NPU 中间层
│       ├── graclus_npu.{h,cpp}
│       ├── grid_npu.{h,cpp}
│       ├── knn_npu.{h,cpp}
│       ├── nearest_npu.{h,cpp}
│       ├── radius_npu.{h,cpp}
│       ├── rw_npu.{h,cpp}
│       ├── sampler_npu.{h,cpp}
│       └── impl/
│           ├── *_npu.json                  # 算子原型定义（msOpGen 输入）
│           └── op_project/                 # 唯一算子工程（4 个内核合并打包）
│               ├── build.sh
│               ├── CMakeLists.txt
│               ├── CMakePresets.json
│               ├── cmake/  framework/  scripts/
│               ├── op_host/                # 原型定义 + tiling
│               │   ├── farthest_point_sampling.cpp / _tiling.h
│               │   ├── voxel_grid.cpp / _tiling.h
│               │   ├── nearest_neighbor.cpp / _tiling.h
│               │   └── radius.cpp / radius_tiling.h
│               └── op_kernel/              # Ascend C 内核
│                   ├── farthest_point_sampling.cpp
│                   ├── voxel_grid.cpp
│                   ├── nearest_neighbor.cpp
│                   └── radius.cpp
├── torch_cluster/                  # Python 接口（与原始库一致）
├── test/                           # 测试脚本（含 test_all_npu.py）
├── CMakeLists.txt
├── setup.py
└── README.md
```

---

## 算子支持状态

| pytorch_cluster 算子 | NPU aclnn 算子 | 中间层 | Ascend C 内核 | 说明 |
|---|---|:---:|:---:|---|
| `fps` (Farthest Point Sampling) | `aclnnFarthestPointSampling` | ✅ | ✅ | 批并行 / 40 核协同双模式，dim=2/3 特化 |
| `grid_cluster` (VoxelGrid) | `aclnnVoxelGrid` | ✅ | ✅ | SoA `[D,N]`，支持 fp16/fp32 |
| `nearest` (Nearest Neighbor) | `aclnnNearestNeighbor` | ✅ | ✅ | dim=2 特化 + 任意维向量化 |
| `radius` (Radius Search) | `aclnnRadius` | ✅ | ✅ | 多核，DMA(`DataCopyPad`) 写回 |
| `graclus` | `aclnnGraclus` | ✅ | ❌ | 仅中间层 |
| `knn` | `aclnnKNNSearch` | ✅ | ❌ | 仅中间层 |
| `random_walk` | `aclnnRandomWalk` | ✅ | ❌ | 仅中间层 |
| `neighbor_sampler` | `aclnnNeighborSampler` | ✅ | ❌ | 仅中间层 |

> 只有 `fps`、`grid_cluster`、`nearest`、`radius` 提供了 Ascend C 内核，可在 NPU 上运行；其余算子仅有中间层，需补齐内核后才能运行。
> `radius` 不使用 `radiusdriving` 实现。

---

## 架构说明

```
Python 调用层            torch_cluster/*.py
    ↓  torch.ops.torch_cluster.xxx
csrc/*.cpp  (设备分发：CPU / CUDA / NPU)
    ↓  [WITH_NPU]
csrc/npu/*_npu.cpp  (NPU 中间层)
    ↓  EXEC_NPU_CMD 宏
aclnn 算子 API  (由 csrc/npu/impl/op_project 编译出的算子包提供)
    ↓
昇腾 NPU 硬件
```

### EXEC_NPU_CMD 宏

`csrc/npu/include/pytorch_npu_helper.hpp` 中的 `EXEC_NPU_CMD` 封装了调用 aclnn 算子的完整流程：

1. 动态查找算子函数地址（`dlsym`，结果缓存）
2. 调用 `GetWorkspaceSize` 确定工作空间大小
3. 在 NPU 上分配工作空间
4. 将 ATen 张量转换为 ACL 类型
5. 在当前 NPU stream 上异步执行算子
6. 销毁临时 ACL 对象

```cpp
// 示例：调用 radius NPU 算子
EXEC_NPU_CMD(aclnnRadius, x, y, ptr_x, ptr_y, r, max_num_neighbors,
             ignore_same_index, out);
```

### 算子工程 op_project

`csrc/npu/impl/op_project/` 是**唯一的算子工程**，把已实现内核的算子（fps / grid / nearest / radius）合并在一起：

- `op_host/`：算子原型（输入/输出/属性）与 tiling；
- `op_kernel/`：Ascend C 内核；
- 编译后产出单个 `.run` 算子包，其中 `libcust_opapi.so` 同时包含全部 `aclnn*` 接口。

> 之所以合并为一个工程：每个算子包都会覆盖同一个 `libcust_opapi.so`，分开安装会互相冲掉，导致中间层在运行时找不到部分算子。

`csrc/npu/impl/*.json` 为各算子的原型定义，可作为 `msOpGen` 生成工程的输入（原型以 `op_project/op_host` 为准）：

```bash
msopgen gen -i csrc/npu/impl/radius_npu.json \
            -f pytorch \
            -c ai_core-Ascend910B \
            -lan cpp \
            -out ./csrc/npu/impl/op_project
```

---

## 编译安装

### 前提条件

- Python >= 3.8
- PyTorch + torch_npu（昇腾 NPU 环境）
- CANN Toolkit（已安装 `msopgen`/Ascend C 编译工具链）
- 昇腾 NPU 硬件（Atlas 训练/推理系列，算子工程默认 `ascend910b`）

### 步骤 1：编译并安装 NPU 算子包

```bash
export ASCEND_HOME_PATH=/home/ma-user/Ascend/ascend-toolkit/latest   # 按实际环境修改

cd csrc/npu/impl/op_project
bash build.sh
./build_out/custom_opp_*.run --quiet          # 安装到默认 OPP 目录
# 指定目录安装： ./build_out/custom_opp_*.run --quiet --install-path=/your/opp
```

安装后按需 `source` 环境（非默认路径安装时）：

```bash
source <install_path>/vendors/customize/bin/set_env.bash
```

### 步骤 2：编译 torch_cluster Python 扩展

```bash
cd /home/ma-user/work/pytorch-cluster-npu
pip install -e . --no-build-isolation
```

`setup.py` 在 `torch_npu.npu.is_available()` 为真时定义 `WITH_NPU`，编译出 `torch_cluster/_*_npu.so`。

---

## 运行测试

统一测试脚本会依次运行所有已实现内核的 NPU 算子：

```bash
source <opp_install_path>/vendors/customize/bin/set_env.bash
python test/test_all_npu.py
```

预期输出：

```
[ OK ] fps
[ OK ] grid_cluster
[ OK ] nearest
[ OK ] radius
passed: ['fps', 'grid_cluster', 'nearest', 'radius']   failed: []
skipped (no kernel implemented yet): knn, random_walk (rw), neighbor_sampler (sampler), graclus, radiusdriving
```

也可单独运行各算子测试（仓库保留了原始测试，默认在 CPU/CUDA 上运行；NPU 覆盖以 `test_all_npu.py` 为准）：

```bash
pytest test/test_fps.py test/test_grid.py test/test_nearest.py test/test_radius.py
```

---

## 算子开发指引

1. 参考 `csrc/npu/impl/*.json` 的原型定义，用 `msOpGen` 生成工程骨架；
2. 在 `csrc/npu/impl/op_project/op_host/` 增加算子原型与 tiling，在 `op_kernel/` 增加 Ascend C 内核；
3. 在 `csrc/npu/` 增加对应中间层 `xxx_npu.cpp/.h`，并在 `torch_cluster/` 中接入 Python 接口；
4. 重新 `bash build.sh` 生成算子包、`pip install -e .` 重编扩展，用 `test/test_all_npu.py` 验证。

> 多核内核注意：跨核结果写回 **不要直接对 GM 使用标量 `SetValue`**（非本核写入可能丢失），应先在 UB 组装后用 `DataCopy`/`DataCopyPad` 写回。`radius` 内核即采用该方式（`DataCopyPad` 支持非 32B 对齐地址与长度）。

---

## 参考项目

- [pytorch_cluster](https://github.com/rusty1s/pytorch_cluster) — 原始算子实现
- [pytorch-sparse-npu](https://github.com/IDMG-Lab/pytorch-sparse-npu) — 参考 NPU 适配架构
- [torch_npu](https://github.com/Ascend/pytorch) — 昇腾 PyTorch 适配插件
- CANN 算子开发工具用户指南 — JSON 配置与 Ascend C 参考
