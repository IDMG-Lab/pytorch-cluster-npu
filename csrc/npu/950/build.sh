#!/usr/bin/env bash
set -euo pipefail
: "${ASCEND_HOME_PATH:?Source the CANN environment first}"
here=$(cd -- "$(dirname -- "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
archroot="$ASCEND_HOME_PATH/$(uname -m)-linux"
"$archroot/ccec_compiler/bin/bisheng" --asc-aicore-lang --npu-arch=dav-3510 \
  -O2 -fPIC -shared "$here/subgraph950.cpp" \
  -I"$ASCEND_HOME_PATH/include" -I"$ASCEND_HOME_PATH/include/ascendc" \
  -I"$ASCEND_HOME_PATH/include/ascendc/basic_api" \
  -I"$archroot/asc/include" -I"$archroot/asc/include/simt_api" \
  -I"$archroot/asc/include/basic_api" -I"$archroot/asc/include/tiling" \
  -I"$archroot/asc/include/tiling/platform" \
  -L"$ASCEND_HOME_PATH/lib64" -L"$archroot/lib64" \
  -lplatform -lascendcl -lruntime -lascendc_runtime -ltiling_api -lunified_dlog \
  -o "$root/torch_cluster/libsubgraph950.so"
