# pytorch-cluster-npu

**pytorch-cluster-npu** 是 [pytorch_cluster](https://github.com/rusty1s/pytorch_cluster) 的昇腾 NPU 适配版本，实现了 PyTorch C++ Extension 中间层，使图聚类算子能够在华为昇腾 NPU 上运行。

---

## 项目结构

```
pytorch-cluster-npu/
├── csrc/
│   ├── fps.cpp            # FPS算子分发（CPU/CUDA/NPU三路分发）
│   ├── graclus.cpp        # Graclus算子分发
│   ├── grid.cpp           # VoxelGrid算子分发
│   ├── knn.cpp            # KNN算子分发
│   ├── nearest.cpp        # Nearest算子分发
│   ├── radius.cpp         # Radius算子分发
│   ├── rw.cpp             # RandomWalk算子分发
│   ├── sampler.cpp        # NeighborSampler算子分发
│   ├── cpu/               # 原始CPU实现（来自pytorch_cluster）
│   ├── cuda/              # 原始CUDA实现（来自pytorch_cluster）
│   └── npu/               # NPU适配层（本项目新增）
│       ├── include/
│       │   └── pytorch_npu_helper.hpp   # EXEC_NPU_CMD宏及类型转换工具
│       ├── fps_npu.h / fps_npu.cpp      # FPS NPU中间层
│       ├── graclus_npu.h / graclus_npu.cpp
│       ├── grid_npu.h / grid_npu.cpp
│       ├── knn_npu.h / knn_npu.cpp
│       ├── nearest_npu.h / nearest_npu.cpp
│       ├── radius_npu.h / radius_npu.cpp
│       ├── rw_npu.h / rw_npu.cpp
│       ├── sampler_npu.h / sampler_npu.cpp
│       ├── nearest_npu/                 # nearest NPU算子相关实现
│       └── impl/                        # 算子原型定义（JSON）
│           ├── fps_npu.json
│           ├── graclus_npu.json
│           ├── grid_npu.json
│           ├── knn_npu.json
│           ├── nearest_npu.json
│           ├── radius_npu.json
│           ├── rw_npu.json
│           └── sampler_npu.json
├── torch_cluster/         # Python包（复用原始接口）
├── CMakeLists.txt
├── setup.py
└── README.md
```

---

## 算子映射

| pytorch_cluster 算子 | NPU CANN 算子名 | JSON 定义文件 |
|---|---|---|
| `fps` (Farthest Point Sampling) | `aclnnFarthestPointSampling` | `impl/fps_npu.json` |
| `graclus` (Graclus Clustering) | `aclnnGraclus` | `impl/graclus_npu.json` |
| `grid` (VoxelGrid Clustering) | `aclnnVoxelGrid` | `impl/grid_npu.json` |
| `knn` (K-Nearest Neighbors) | `aclnnKNNSearch` | `impl/knn_npu.json` |
| `nearest` (Nearest Neighbor Assignment) | `aclnnNearestNeighbor` | `impl/nearest_npu.json` |
| `radius` (Radius Search) | `aclnnRadiusSearch` | `impl/radius_npu.json` |
| `random_walk` (Node2Vec Random Walk) | `aclnnRandomWalk` | `impl/rw_npu.json` |
| `neighbor_sampler` (Neighbor Sampler) | `aclnnNeighborSampler` | `impl/sampler_npu.json` |

---

## 架构说明

### 中间层适配设计

本项目仅实现 **PyTorch C++ Extension 中间层**，不包含具体的算子内核实现。架构如下：

```
Python 调用层
    ↓
csrc/*.cpp  (设备分发：CPU / CUDA / NPU)
    ↓  [WITH_NPU]
csrc/npu/*_npu.cpp  (NPU中间层)
    ↓  EXEC_NPU_CMD 宏
aclnn算子 API  (CANN运行时，由具体算子工程提供)
    ↓
昇腾 NPU 硬件
```

### EXEC_NPU_CMD 宏

`pytorch_npu_helper.hpp` 中定义的 `EXEC_NPU_CMD` 宏封装了调用 CANN aclnn 算子的完整流程：

1. 动态查找算子函数地址（dlsym，结果缓存）
2. 调用 `GetWorkspaceSize` 确定工作空间大小
3. 在 NPU 上分配工作空间
4. 将 ATen 张量转换为 ACL 类型
5. 在当前 NPU stream 上异步执行算子
6. 销毁临时 ACL 对象

```cpp
// 示例：调用 FPS NPU 算子
EXEC_NPU_CMD(aclnnFarthestPointSampling, src, ptr, ratio, random_start, out);
```

### 算子工程 JSON 定义

`csrc/npu/impl/` 目录下每个 `.json` 文件对应一个算子的原型定义，用于通过 `msOpGen` 工具生成算子开发工程：

```bash
msopgen gen -i csrc/npu/impl/fps_npu.json \
            -f pytorch \
            -c ai_core-Ascend910B \
            -lan cpp \
            -out ./op_projects/fps
```

---

## 编译安装

### 前提条件

- Python >= 3.8
- PyTorch >= 1.13
- torch_npu（昇腾 NPU 环境）
- CANN Toolkit >= 7.0
- 昇腾 NPU 硬件（Atlas 训练系列或推理系列）

### 安装步骤

```bash
# 1. 克隆本仓库
git clone https://github.com/IDMG-Lab/pytorch-cluster-npu.git
cd pytorch-cluster-npu


# 2. 安装
pip install -e . --no-build-isolation
```

---

## 算子开发指引

1. 使用 `csrc/npu/impl/` 目录下对应的 JSON 文件，通过 `msOpGen` 生成算子工程
2. 在生成的工程 `op_kernel/` 目录中实现 Ascend C 内核代码
3. 编译部署算子包（`.run` 文件）到 NPU 运行环境

---

## 参考项目

- [pytorch_cluster](https://github.com/rusty1s/pytorch_cluster) — 原始算子实现
- [pytorch-sparse-npu](https://github.com/IDMG-Lab/pytorch-sparse-npu) — 参考 NPU 适配架构
- [torch_npu](https://github.com/Ascend/pytorch) — 昇腾 PyTorch 适配插件
- CANN 社区版 8.5.0 算子开发工具用户指南 — JSON 配置格式参考
