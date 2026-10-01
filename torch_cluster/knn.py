from typing import Optional

import torch


def _knn_tiled_npu(x: torch.Tensor, y: torch.Tensor, k: int) -> torch.Tensor:
    # Center before the norm expansion to avoid cancellation from a common
    # translation. Refine a shortlist with direct differences, and fall back
    # to the scalar kernel when the exclusion margin is numerically unsafe.
    # Keep the distance tile near 32 MiB; larger workspaces increase
    # Add/Mul/TopK memory traffic on large reference sets.
    tile = max(64, min(1024, 8_388_608 // x.size(0)))
    shortlist = min(x.size(0), k + 8)
    origin = x[:1]
    reference_points = x - origin
    query_points = y - origin
    reference_norm = (reference_points * reference_points).sum(1)
    query_norm = (query_points * query_points).sum(1)
    transposed = reference_points.t().contiguous()
    references = []
    boundaries = []
    for first in range(0, y.size(0), tile):
        distances = torch.mm(query_points[first:first + tile], transposed) * -2
        distances = distances + reference_norm.unsqueeze(0)
        distances = distances + query_norm[first:first + tile].unsqueeze(1)
        values, ids = distances.topk(shortlist, dim=1, largest=False, sorted=True)
        references.append(ids)
        boundaries.append(values[:, -1])
    candidates = references[0] if len(references) == 1 else torch.cat(references)
    # Index ordering followed by a stable distance sort handles exact ties.
    # FP32 represents the permitted fast-path index range exactly and keeps
    # this sort on AIV instead of the int64 AICPU sorting fallback.
    candidates = candidates.to(torch.float32).sort(dim=1).values.to(torch.long)
    # A flat row gather avoids the broadcasted advanced-indexing path.
    # index_select owns this buffer, so refinement cannot mutate inputs.
    neighbors = torch.index_select(x, 0, candidates.reshape(-1)).view(
        y.size(0), shortlist, x.size(1))
    neighbors.sub_(y.unsqueeze(1)).square_()
    exact = neighbors.sum(2)
    order = exact.argsort(dim=1, stable=True)
    selected = candidates.gather(1, order[:, :k])
    if shortlist < x.size(0):
        boundary = boundaries[0] if len(boundaries) == 1 else torch.cat(boundaries)
        # Conservative FP32 reduction/matmul error estimate. This also sends
        # nonfinite norm expansions to the direct-distance baseline.
        error = (4 * x.size(1) * 1.1920928955078125e-7) * (
            1 + reference_norm.max() + query_norm)
        kth = exact.gather(1, order[:, k - 1:k]).reshape(-1)
        unsafe = ~torch.isfinite(boundary) | (kth >= boundary - error)
        unsafe_ids = unsafe.nonzero().reshape(-1)
        if unsafe_ids.numel():
            fallback = torch.ops.torch_cluster.knn(
                x, y[unsafe_ids].contiguous(), None, None, k, False, 1)
            selected[unsafe_ids] = fallback[1].view(-1, k)
    output = torch.empty((2, y.size(0) * k), dtype=torch.long, device=y.device)
    output[0].view(-1, k).copy_(
        torch.arange(y.size(0), device=y.device).view(-1, 1))
    output[1].copy_(selected.reshape(-1))
    return output


def knn(
    x: torch.Tensor,
    y: torch.Tensor,
    k: int,
    batch_x: Optional[torch.Tensor] = None,
    batch_y: Optional[torch.Tensor] = None,
    cosine: bool = False,
    num_workers: int = 1,
    batch_size: Optional[int] = None,
) -> torch.Tensor:
    r"""Finds for each element in :obj:`y` the :obj:`k` nearest points in
    :obj:`x`.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        y (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{M \times F}`.
        k (int): The number of neighbors.
        batch_x (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. :obj:`batch_x` needs to be sorted.
            (default: :obj:`None`)
        batch_y (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^M`, which assigns each
            node to a specific example. :obj:`batch_y` needs to be sorted.
            (default: :obj:`None`)
        cosine (boolean, optional): If :obj:`True`, will use the Cosine
            distance instead of the Euclidean distance to find nearest
            neighbors. (default: :obj:`False`)
        num_workers (int): Number of workers to use for computation. Has no
            effect in case :obj:`batch_x` or :obj:`batch_y` is not
            :obj:`None`, or the input lies on the GPU. (default: :obj:`1`)
        batch_size (int, optional): The number of examples :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)

    :rtype: :class:`LongTensor`

    .. code-block:: python

        import torch
        from torch_cluster import knn

        x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        batch_x = torch.tensor([0, 0, 0, 0])
        y = torch.Tensor([[-1, 0], [1, 0]])
        batch_y = torch.tensor([0, 0])
        assign_index = knn(x, y, 2, batch_x, batch_y)
    """
    if x.numel() == 0 or y.numel() == 0:
        return torch.empty(2, 0, dtype=torch.long, device=x.device)

    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y
    x, y = x.contiguous(), y.contiguous()

    if batch_size is None:
        batch_size = 1
        if batch_x is not None:
            assert x.size(0) == batch_x.numel()
            batch_size = int(batch_x.max()) + 1
        if batch_y is not None:
            assert y.size(0) == batch_y.numel()
            batch_size = max(batch_size, int(batch_y.max()) + 1)
    assert batch_size > 0

    ptr_x: Optional[torch.Tensor] = None
    ptr_y: Optional[torch.Tensor] = None
    if batch_size > 1:
        assert batch_x is not None
        assert batch_y is not None
        arange = torch.arange(batch_size + 1, device=x.device)
        ptr_x = torch.bucketize(arange, batch_x)
        ptr_y = torch.bucketize(arange, batch_y)

    if (x.device.type == 'npu' and ptr_x is None and not cosine
            and x.dtype == torch.float32 and y.dtype == torch.float32
            and k <= x.size(0) <= 16_777_216 and 0 < k <= 100
            and x.size(0) * y.size(0) * x.size(1) >= 1_048_576):
        return _knn_tiled_npu(x, y, k)

    out = torch.ops.torch_cluster.knn(x, y, ptr_x, ptr_y, k, cosine,
                                      num_workers)
    if x.device.type == 'npu' and (ptr_x is not None or x.size(0) < k):
        # The fixed-capacity Ascend C output uses -1 for missing neighbors.
        # A query's batch may contain fewer than k reference points.
        return out[:, out[1] >= 0]
    return out


def knn_graph(
    x: torch.Tensor,
    k: int,
    batch: Optional[torch.Tensor] = None,
    loop: bool = False,
    flow: str = 'source_to_target',
    cosine: bool = False,
    num_workers: int = 1,
    batch_size: Optional[int] = None,
) -> torch.Tensor:
    r"""Computes graph edges to the nearest :obj:`k` points.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        k (int): The number of neighbors.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. :obj:`batch` needs to be sorted.
            (default: :obj:`None`)
        loop (bool, optional): If :obj:`True`, the graph will contain
            self-loops. (default: :obj:`False`)
        flow (string, optional): The flow direction when used in combination
            with message passing (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)
        cosine (boolean, optional): If :obj:`True`, will use the Cosine
            distance instead of Euclidean distance to find nearest neighbors.
            (default: :obj:`False`)
        num_workers (int): Number of workers to use for computation. Has no
            effect in case :obj:`batch` is not :obj:`None`, or the input lies
            on the GPU. (default: :obj:`1`)
        batch_size (int, optional): The number of examples :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)

    :rtype: :class:`LongTensor`

    .. code-block:: python

        import torch
        from torch_cluster import knn_graph

        x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        batch = torch.tensor([0, 0, 0, 0])
        edge_index = knn_graph(x, k=2, batch=batch, loop=False)
    """

    assert flow in ['source_to_target', 'target_to_source']
    edge_index = knn(x, x, k if loop else k + 1, batch, batch, cosine,
                     num_workers, batch_size)

    if flow == 'source_to_target':
        row, col = edge_index[1], edge_index[0]
    else:
        row, col = edge_index[0], edge_index[1]

    if not loop:
        mask = row != col
        row, col = row[mask], col[mask]

    return torch.stack([row, col], dim=0)
