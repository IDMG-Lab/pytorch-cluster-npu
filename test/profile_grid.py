import torch
from torch_cluster import grid_cluster


def calc_grid_cluster(point, size, start, end):
    """
    和 kernel 中多维线性展开逻辑保持一致：

    grid[d] = floor((point[d] - start[d]) / size[d])
    grid_size[d] = floor((end[d] - start[d]) / size[d]) + 1

    cluster = grid[0]
            + grid[1] * grid_size[0]
            + grid[2] * grid_size[0] * grid_size[1]
            + ...
    """
    cluster = 0
    stride = 1

    for d in range(len(point)):
        grid = int((point[d] - start[d]) // size[d])
        cluster += grid * stride

        grid_size = int((end[d] - start[d]) // size[d]) + 1
        stride *= grid_size

    return cluster


def create_multidim_pos(num_points, dim, device, dtype):
    """
    构造 [N, D] 多维输入。

    使用取模控制坐标范围，避免坐标过大。
    """
    idx = torch.arange(num_points, dtype=dtype, device=device)

    cols = []
    for d in range(dim):
        # 每一维使用不同步长，构造不同坐标分布
        col = ((idx * (d + 1)) % 1000).reshape(-1, 1)
        cols.append(col)

    pos = torch.cat(cols, dim=1).contiguous()
    return pos


def expected_for_indices(indices, dim, size, start, end):
    """
    只计算少量 head / tail 的 expected，避免大规模 CPU 计算。
    """
    expected = []

    for i in indices:
        point = [
            float((i * (d + 1)) % 1000)
            for d in range(dim)
        ]

        expected.append(
            calc_grid_cluster(
                point,
                size,
                start,
                end,
            )
        )

    return expected


def main():
    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        raise RuntimeError("NPU not available")

    device = torch.device("npu:0")
    dtype = torch.float32

    num_points = 1000000
    dim = 3

    # pos: [N, D]
    pos = create_multidim_pos(
        num_points=num_points,
        dim=dim,
        device=device,
        dtype=dtype,
    )

    size_list = [5.0] * dim
    start_list = [0.0] * dim
    end_list = [1000.0] * dim

    size = torch.tensor(size_list, dtype=dtype, device=device)
    start = torch.tensor(start_list, dtype=dtype, device=device)
    end = torch.tensor(end_list, dtype=dtype, device=device)

    print("===== Multi-Dim VoxelGrid Profile =====")
    print("pos.shape:", pos.shape)
    print("size:", size)
    print("start:", start)
    print("end:", end)

    # warmup
    for _ in range(5):
        cluster = grid_cluster(pos, size, start, end)

    torch.npu.synchronize()

    # profiling target
    for _ in range(20):
        cluster = grid_cluster(pos, size, start, end)

    torch.npu.synchronize()

    assert cluster.dtype == torch.int64
    assert cluster.numel() == num_points

    # 只检查 head / tail，避免全量拷贝
    head_len = 16
    tail_len = 16

    head_indices = list(range(head_len))
    tail_indices = list(range(num_points - tail_len, num_points))

    head_output = cluster[:head_len].cpu().tolist()
    tail_output = cluster[-tail_len:].cpu().tolist()

    head_expected = expected_for_indices(
        head_indices,
        dim,
        size_list,
        start_list,
        end_list,
    )

    tail_expected = expected_for_indices(
        tail_indices,
        dim,
        size_list,
        start_list,
        end_list,
    )

    print("head_output:", head_output)
    print("head_expected:", head_expected)

    print("tail_output:", tail_output)
    print("tail_expected:", tail_expected)

    assert head_output == head_expected
    assert tail_output == tail_expected

    print("multi-dim large test passed")
    print("done")


if __name__ == "__main__":
    main()