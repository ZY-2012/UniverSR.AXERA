import sys
from pathlib import Path

import numpy as np


def metrics(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    d = a - b
    max_abs = float(np.max(np.abs(d)))
    rmse = float(np.sqrt(np.mean(d * d)))
    denom = float(np.sum(a * a))
    snr = float(10.0 * np.log10(denom / max(float(np.sum(d * d)), 1e-30))) if denom > 0 else float("-inf")
    corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) if a.size > 1 else 1.0
    return max_abs, rmse, snr, corr


def main():
    name = sys.argv[1]
    root = Path(__file__).resolve().parents[1] / "build" / "compile" / f"{name}_work" / "quant" / "debug" / "io"
    fdir, qdir = root / "float", root / "quant"
    print(f"\n=== quantized-graph accuracy: {name} (float vs QuantAxModel, host reference) ===")
    if not fdir.is_dir() or not qdir.is_dir():
        print("no quant debug tensors")
        return
    for f in sorted(fdir.glob("*.npy")):
        q = qdir / f.name
        if not q.exists():
            continue
        a = np.load(f)
        b = np.load(q)
        if a.shape != b.shape:
            print(f"{f.stem}: shape {a.shape} != {b.shape}")
            continue
        mx, rmse, snr, corr = metrics(a, b)
        print(f"{f.stem:16s} max_abs={mx:.6g} rmse={rmse:.6g} snr={snr:.2f} dB corr={corr:.8f}")
    print("NOTE: QuantAxModel is the host fake-quant reference, NOT the compiled NPU axmodel. Board accuracy is reported separately.")
    print("=== end quantized-graph accuracy ===\n")


if __name__ == "__main__":
    main()
