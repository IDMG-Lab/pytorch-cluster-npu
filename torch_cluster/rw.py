from typing import Optional, Tuple, Union

import torch
from torch import Tensor


def random_walk(
    row: Tensor,
    col: Tensor,
    start: Tensor,
    walk_length: int,
    p: float = 1,
    q: float = 1,
    coalesced: bool = True,
    num_nodes: Optional[int] = None,
    return_edge_indices: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Samples random walks of length :obj:`walk_length` from all node indices
    in :obj:`start` in the graph given by :obj:`(row, col)` as described in the
    `"node2vec: Scalable Feature Learning for Networks"
    <https://arxiv.org/abs/1607.00653>`_ paper.
    Edge indices :obj:`(row, col)` need to be coalesced/sorted according to
    :obj:`row` (use the :obj:`coalesced` attribute to force).

    Args:
        row (Tensor): Source nodes, with int32 or int64 dtype.
        col (Tensor): Target nodes, with the same dtype as row.
        start (Tensor): Starting nodes, with the same dtype as row.
        walk_length (int): The walk length.
        p (float, optional): Likelihood of immediately revisiting a node in the
            walk. (default: :obj:`1`)
        q (float, optional): Control parameter to interpolate between
            breadth-first strategy and depth-first strategy (default: :obj:`1`)
        coalesced (bool, optional): If set to :obj:`True`, will coalesce/sort
            the graph given by :obj:`(row, col)` according to :obj:`row`.
            (default: :obj:`True`)
        num_nodes (int, optional): The number of nodes. (default: :obj:`None`)
        return_edge_indices (bool, optional): Whether to additionally return
            the indices of edges traversed during the random walk.
            (default: :obj:`False`)

    Output indices retain the input index dtype.
    """
    if (row.dtype not in (torch.int32, torch.int64)
            or col.dtype != row.dtype or start.dtype != row.dtype):
        raise TypeError("row, col and start must have the same int32 or int64 dtype")
    node_seq, edge_seq = torch.ops.torch_cluster.random_walk_coo(
        row, col, start, walk_length, p, q, coalesced, num_nodes)

    if return_edge_indices:
        return node_seq, edge_seq

    return node_seq
