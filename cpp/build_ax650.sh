#!/usr/bin/env bash
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; B="$D/build/ax650"
TC="${TOOLCHAIN_ROOT:-/data/shared/huyuan/toolchains/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu}"
BSP="${BSP_MSP_DIR:-/data/shared/huyuan/toolchains/ax650n_bsp_sdk/msp/out}"
rm -rf "$B"; mkdir -p "$B"
cmake -S "$D" -B "$B" -DCMAKE_BUILD_TYPE=Release -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 -DCMAKE_C_COMPILER="$TC/bin/aarch64-none-linux-gnu-gcc" -DCMAKE_CXX_COMPILER="$TC/bin/aarch64-none-linux-gnu-g++" -DAX650=ON -DBSP_MSP_DIR="$BSP"
cmake --build "$B" -j"$(nproc)"; mkdir -p "$D/bin"; cp "$B/universr_ax650" "$D/bin/universr_ax650"; "$TC/bin/aarch64-none-linux-gnu-strip" --strip-unneeded "$D/bin/universr_ax650"; echo built "$D/bin/universr_ax650"
