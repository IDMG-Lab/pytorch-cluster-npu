# Ascend 950 SIMT subgraph

`subgraph_950` and `subgraph_910b` are separately registered operators. The
existing `subgraph` interface chooses the backend for the current device.
The 950 backend loads its own `libsubgraph950.so`; the 910B backend continues
to use `aclnnClusterSubgraph` and its existing OPP package.

## Build and use

Source the CANN and torch_npu environment on an Ascend 950 machine, then run
from the repository root:

```bash
TORCH_CLUSTER_ONLY_OP=subgraph python setup.py build_ext --inplace
```

The build invokes `build.sh` with the `dav-3510` architecture and copies the
kernel library alongside the Python extension. An editable installation also
builds this library automatically on 950 hardware:

```bash
pip install -e . --no-build-isolation
```

```python
from torch_cluster import subgraph_950, subgraph_910b

out_rowptr, out_col, edge_id = subgraph_950(rowptr, col, nodes)
out_rowptr, out_col = subgraph_950(rowptr, col, nodes, return_edge_id=False)
# On 910B hardware, call subgraph_910b with the same arguments.
```

Inputs must be one-dimensional tensors on the same NPU with a matching
int32 or int64 dtype. They must describe valid CSR data and unique, in-range
selected nodes. N, E and Q must each fit in int32, including with int64 input.
Outputs follow the selected-node order, retain duplicate-edge order and
address the original column array with edge IDs. Explicit backend interfaces
reject hardware of the other type.

## Implementation

Six ordered launches implement mapping initialization, mapping writes,
edge counts, two levels of prefix scan and stable edge scattering.
`__simt_vf__` and `Simt::VF_CALL` execute the device work in the SIMT paradigm.
Short rows use individual threads; long rows and hubs use warp cooperation.
Warp shuffles implement prefix scans. The caller's current stream is captured
before dispatch to the OpCommand worker.

## Validation

```bash
python -m pytest test/test_subgraph.py test/test_subgraph_backends.py -q
python test/acceptance/subgraph950.py --accuracy --performance \
  --profile-kernels --output subgraph950-results.json
```

The acceptance runner checks 32 document shapes with int32/int64 and uniform
or skewed degrees, plus empty/single-node cases. It checks both edge-ID return
modes against an independent NumPy CSR reference: 136 inputs, 272 calls.
Backend tests also cover duplicate edges, partial warps, hubs, changing inputs
and nondefault streams.

Performance follows the eight uploaded pyg-lib benchmark shapes with int32,
20 warmups and 100 measured iterations. The threshold is
`historical_CPU_ms / NPU_public_wall_ms >= 0.2`. The recorded 950 run achieved
a mean ratio of 18.68 and a minimum of 3.46. Hardware profiling separately
counts six device kernels per invocation. Historical CPU logs lack input
seeds/hashes, so they do not establish identical random inputs on both sides.

The implementation was tested on Ascend950PR_9579 with CANN 9.0.0 beta.2,
PyTorch 2.7.1 and torch_npu 2.7.1.post8. A 910B hardware regression has not
been run. Remaining costs include host synchronization for dynamic output
lengths, output capacity proportional to the original E, and load balance
for extreme single-hub graphs.
