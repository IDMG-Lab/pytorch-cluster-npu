#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified smoke test for all NPU kernels implemented in pytorch-cluster-npu.

Currently implemented NPU operators (op_host + op_kernel):
    * fps           (aclnnFarthestPointSampling) -> csrc/npu/impl/op_project
    * grid_cluster  (aclnnVoxelGrid)             -> csrc/npu/impl/op_project
    * nearest       (aclnnNearestNeighbor)       -> csrc/npu/impl/op_project
    * radius        (aclnnRadius)                -> csrc/npu/impl/op_project

Operators that only have a C++ middle layer (no Ascend C kernel yet) are
listed at the bottom and intentionally skipped:
    knn, random_walk, neighbor_sampler, graclus, radiusdriving

Prerequisites
-------------
1. Build & install the custom operator package.  All four kernels live in the
   single combined project ``csrc/npu/impl/op_project`` (fps + grid + nearest +
   radius), which produces one package containing every ``aclnn`` kernel::

       export ASCEND_HOME_PATH=/home/ma-user/Ascend/ascend-toolkit/latest
       cd csrc/npu/impl/op_project
       bash build.sh
       ./build_out/custom_opp_*.run --quiet          # installs into the OPP dir

2. Build the torch_cluster Python extension with NPU support::

       cd /home/ma-user/work/pytorch-cluster-npu
       pip install -e . --no-build-isolation

   This script inserts the repository root into ``sys.path`` first, so it uses
   the freshly built ``torch_cluster/_*_npu.so`` even if a CPU-only
   ``torch_cluster`` is installed in site-packages.

3. Make sure the installed custom op_api lib is on the library path::

       source <opp_install_path>/vendors/customize/bin/set_env.bash

Run
---
    python test/test_all_npu.py
"""

import os
import sys
import traceback

# Make sure the repository's torch_cluster (with the freshly built *_npu.so) is
# imported ahead of any CPU-only installation in site-packages.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.isdir(os.path.join(_REPO_ROOT, "torch_cluster")):
    sys.path.insert(0, _REPO_ROOT)

import torch

try:
    import torch_npu  # noqa: F401
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"torch_npu is required to run this test: {exc}")

from torch_cluster import fps, grid_cluster, nearest, radius

DEVICE = torch.device("npu:0")

# Operators that still have no Ascend C kernel implementation.
UNIMPLEMENTED = [
    "knn",
    "random_walk (rw)",
    "neighbor_sampler (sampler)",
    "graclus",
    "radiusdriving",
]


# ============================================================================
# reference implementations (pure python / cpu)
# ============================================================================
def cpu_grid_cluster(point, size, start, end):
    """Mirror the linearised VoxelGrid index computation."""
    cluster = 0
    stride = 1
    for d in range(len(point)):
        grid = int((point[d] - start[d]) // size[d])
        cluster += grid * stride
        grid_size = int((end[d] - start[d]) // size[d]) + 1
        stride *= grid_size
    return cluster


def cpu_radius_edges(x, y, r, max_num_neighbors):
    """Brute-force radius search.

    Returns a set of (y_index, x_index) tuples, matching the pytorch_cluster
    edge_index convention [y_index; x_index].
    """
    r2 = r * r
    edges = set()
    for j, yj in enumerate(y):
        count = 0
        for i, xi in enumerate(x):
            dist2 = sum((a - b) ** 2 for a, b in zip(xi, yj))
            if dist2 <= r2:
                edges.add((j, i))
                count += 1
                if count >= max_num_neighbors:
                    break
    return edges


def to_edge_set(edge_index):
    edge_index = edge_index.detach().cpu().to(torch.long)
    return set((int(i), int(j)) for i, j in edge_index.t().tolist())


# ============================================================================
# individual operator tests
# ============================================================================
def test_fps():
    src = torch.tensor(
        [
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, 1.0],
            [1.0, -1.0],
            [-2.0, -2.0],
            [-2.0, 2.0],
            [2.0, 2.0],
            [2.0, -2.0],
        ],
        dtype=torch.float32,
        device=DEVICE,
    )
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long, device=DEVICE)

    out = fps(src, batch, ratio=0.5, random_start=False)
    torch.npu.synchronize()

    values = out.cpu().tolist()
    assert len(values) > 0, "fps returned an empty result"
    assert all(0 <= v < src.size(0) for v in values), f"fps index out of range: {values}"
    return f"out={values}"


def test_grid_cluster():
    points = [[float(i * 3 % 97), float(i * 3 % 97 + 1), float(i * 3 % 97 + 2)]
              for i in range(64)]
    size = [5.0, 5.0, 5.0]
    start = [0.0, 0.0, 0.0]
    end = [100.0, 100.0, 100.0]

    pos = torch.tensor(points, dtype=torch.float32, device=DEVICE)
    size_t = torch.tensor(size, dtype=torch.float32, device=DEVICE)
    start_t = torch.tensor(start, dtype=torch.float32, device=DEVICE)
    end_t = torch.tensor(end, dtype=torch.float32, device=DEVICE)

    cluster = grid_cluster(pos, size_t, start_t, end_t)
    torch.npu.synchronize()

    expected = [cpu_grid_cluster(p, size, start, end) for p in points]
    got = cluster.cpu().tolist()
    assert got == expected, "grid_cluster mismatch vs cpu reference"
    return f"{len(got)} voxel ids, first={got[:4]}"


def test_nearest():
    x = torch.tensor(
        [
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, 1.0],
            [1.0, -1.0],
            [-2.0, -2.0],
            [-2.0, 2.0],
            [2.0, 2.0],
            [2.0, -2.0],
        ],
        dtype=torch.float32,
        device=DEVICE,
    )
    y = torch.tensor(
        [[-1.0, 0.0], [1.0, 0.0], [-2.0, 0.0], [2.0, 0.0]],
        dtype=torch.float32,
        device=DEVICE,
    )
    batch_x = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long, device=DEVICE)
    batch_y = torch.tensor([0, 0, 1, 1], dtype=torch.long, device=DEVICE)

    out = nearest(x, y, batch_x, batch_y)
    torch.npu.synchronize()

    got = out.cpu().tolist()
    assert got == [0, 0, 1, 1, 2, 2, 3, 3], f"nearest mismatch: {got}"
    return f"cluster={got}"


def test_radius():
    x = torch.tensor(
        [
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, 1.0],
            [1.0, -1.0],
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, 1.0],
            [1.0, -1.0],
        ],
        dtype=torch.float32,
        device=DEVICE,
    )
    y = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.float32, device=DEVICE)

    r = 2.0
    max_num_neighbors = 4

    edge_index = radius(x, y, r, max_num_neighbors=max_num_neighbors)
    torch.npu.synchronize()

    got = to_edge_set(edge_index)
    assert len(got) > 0, "radius returned no edges"

    # Every returned edge must be a genuine neighbour.
    r2 = r * r
    for j, i in got:
        dist2 = sum((a - b) ** 2 for a, b in zip(x[i].cpu().tolist(), y[j].cpu().tolist()))
        assert dist2 <= r2 + 1e-5, f"radius edge ({j},{i}) is out of range"

    # Number of edges per query must respect max_num_neighbors.
    per_query = {}
    for j, _ in got:
        per_query[j] = per_query.get(j, 0) + 1
    assert all(v <= max_num_neighbors for v in per_query.values()), \
        "radius exceeded max_num_neighbors"

    # Sanity check against the brute-force reference for the first query.
    ref = cpu_radius_edges(x.cpu().tolist(), y.cpu().tolist(), r, max_num_neighbors)
    assert got.issubset(ref), "radius returned a non-existent edge"

    return f"{len(got)} edges, sample={sorted(got)[:4]}"


TESTS = [
    ("fps", test_fps),
    ("grid_cluster", test_grid_cluster),
    ("nearest", test_nearest),
    ("radius", test_radius),
]


# ============================================================================
# runner
# ============================================================================
def main():
    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        raise SystemExit("NPU is not available in this environment.")

    print("=" * 70)
    print("pytorch-cluster-npu unified smoke test")
    print(f"device       : {DEVICE}")
    print(f"torch        : {torch.__version__}")
    print("=" * 70)

    passed, failed = [], []
    for name, fn in TESTS:
        print(f"\n[ RUN ] {name}")
        try:
            detail = fn()
            passed.append(name)
            print(f"[ OK  ] {name}: {detail}")
        except Exception as exc:  # noqa: BLE001
            failed.append(name)
            print(f"[FAIL ] {name}: {exc}")
            traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"passed: {passed}")
    print(f"failed: {failed}")
    print("skipped (no kernel implemented yet): " + ", ".join(UNIMPLEMENTED))
    print("=" * 70)

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
