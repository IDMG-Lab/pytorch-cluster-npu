from typing import Optional, Tuple, Union

import torch


def subgraph(
    rowptr: torch.Tensor,
    col: torch.Tensor,
    nodes: torch.Tensor,
    return_edge_id: bool = True,
) -> Union[Tuple[torch.Tensor, torch.Tensor],
           Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Induced CSR subgraph in ``nodes`` order.

    ``rowptr`` and ``col`` describe a CSR graph. Output columns are relabelled
    to positions in ``nodes``. The optional edge IDs address the original
    ``col`` array. Duplicate edges retain their input order within each row.
    """
    if (rowptr.dtype not in (torch.int32, torch.int64)
            or col.dtype != rowptr.dtype or nodes.dtype != rowptr.dtype):
        raise TypeError("rowptr, col and nodes must have the same int32 or int64 dtype")
    if rowptr.device != col.device or rowptr.device != nodes.device:
        raise ValueError("rowptr, col and nodes must share a device")
    out_rowptr, out_col, out_eid, count = torch.ops.torch_cluster.subgraph(
        rowptr, col, nodes)
    edge_count = int(count.item())
    out_col = out_col.narrow(0, 0, edge_count)
    if return_edge_id:
        return out_rowptr, out_col, out_eid.narrow(0, 0, edge_count)
    return out_rowptr, out_col
