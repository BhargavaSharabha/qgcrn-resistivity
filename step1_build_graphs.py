"""
Step 1 — Build PTE graphs for DPHI and GRD features.

For each feature (DPHI, GRD):
  1. Pivot raw CSV → (142 wells × N_depth) matrices
  2. Compute real pTE matrix (142 × 142) using IAAFT surrogates
  3. Build directed edge list: real_pTE[i→j] > max(surrogate_pTE[i→j])
  4. Save edge_index + edge_attr as .npy files

Output files (graph_data/)
  well_ids.npy              — ordered list of 142 well IDs
  {dphi,grd,ild}_pte_real.npy     — full pTE matrices (142×142)
  {dphi,grd,ild}_pte_iaaft.npy   — surrogate pTE matrices
  edge_index_{dphi,grd,ild}.npy  — (2, E) source/dest node indices
  edge_attr_{dphi,grd,ild}.npy   — (E,) edge weights (real pTE values)

Usage:
  python step1_build_graphs.py
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from pte import pTE

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(BASE_DIR, "aligned_wells.csv")
OUT_DIR  = os.path.join(BASE_DIR, "graph_data")
os.makedirs(OUT_DIR, exist_ok=True)


# ── Helpers ─────────────────────────────────────────────────────────────────

def compute_pte(mat, feature_name, tau=1, dimEmb=1, Nsurr=19):
    """Compute real + IAAFT surrogate pTE matrix; apply post-processing."""
    print(f"\n  Computing pTE for {feature_name}  shape={mat.shape} …")
    pte, pte_surr = pTE(mat, tau=tau, dimEmb=dimEmb, surr='iaaft', Nsurr=Nsurr)

    # Post-processing (from original notebook)
    np.fill_diagonal(pte, 0)
    np.fill_diagonal(pte_surr, 0)
    pte[~np.isfinite(pte)] = 0
    pte_surr[~np.isfinite(pte_surr)] = 0
    pte[pte < 0] = 0
    pte_surr[pte_surr < 0] = 0
    return pte, pte_surr


def build_edges(real_pte, surr_pte):
    """Build directed edge list from PTE vs surrogate comparison.

    Edge i→j is kept iff  real_pTE[i,j]  >  max_surrogate_pTE[i,j]
    This ensures the directed information flow is statistically significant.

    Returns
    -------
    edge_index : (2, E) int64  — source, dest node indices
    edge_attr  : (E,) float32 — edge weights (real pTE values)
    """
    N = real_pte.shape[0]
    src, dst, w = [], [], []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if real_pte[i, j] > surr_pte[i, j]:   # significance threshold
                src.append(i)
                dst.append(j)
                w.append(real_pte[i, j])
    return (np.array([src, dst], dtype=np.int64),
            np.array(w, dtype=np.float32))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("STEP 1 — Build PTE Graphs")
    print("=" * 60)

    # 1. Load & pivot
    print("\n[1/4] Loading data …")
    df = pd.read_csv(DATA_CSV)
    df = df.sort_values(["WELL_ID", "DEPT"]).reset_index(drop=True)
    n_wells = df["WELL_ID"].nunique()
    print(f"  {len(df)} rows × {n_wells} wells")

    print("[2/4] Pivoting …")
    dphi_df = df.pivot(index="WELL_ID", columns="DEPT", values="DPHI")
    grd_df  = df.pivot(index="WELL_ID", columns="DEPT", values="GRD")
    ild_df  = df.pivot(index="WELL_ID", columns="DEPT", values="ILD")

    well_ids = dphi_df.index.tolist()
    dphi_mat = dphi_df.to_numpy(dtype=np.float64)
    grd_mat  = grd_df.to_numpy(dtype=np.float64)
    ild_mat  = ild_df.to_numpy(dtype=np.float64)
    print(f"  Matrix shape: {dphi_mat.shape}  (wells × depth-points)")

    # Save well ordering
    np.save(os.path.join(OUT_DIR, "well_ids.npy"), np.array(well_ids, dtype=str))

    # 2. Compute pTE & build graphs
    print("[3/4] Computing pTE matrices & building graphs …")
    results = [
        (dphi_mat, "dphi"),
        (grd_mat,  "grd"),
        (ild_mat,  "ild"),   # ILD = target (optional, for analysis)
    ]

    for mat, name in results:
        pte, pte_surr = compute_pte(mat, name.upper())

        # Save full matrices
        np.save(os.path.join(OUT_DIR, f"{name}_pte_real.npy"),   pte)
        np.save(os.path.join(OUT_DIR, f"{name}_pte_iaaft.npy"), pte_surr)

        # Build & save edge list
        edge_index, edge_attr = build_edges(pte, pte_surr)
        np.save(os.path.join(OUT_DIR, f"edge_index_{name}.npy"), edge_index)
        np.save(os.path.join(OUT_DIR, f"edge_attr_{name}.npy"),  edge_attr)

        n_edges = edge_index.shape[1]
        density = n_edges / (n_wells * (n_wells - 1))
        print(f"  {name.upper()} graph → {n_edges} edges  density={density:.4f}")

    # 3. Summary
    print(f"\n[4/4] Files saved to  {OUT_DIR}/")
    for f in sorted(os.listdir(OUT_DIR)):
        sz = os.path.getsize(os.path.join(OUT_DIR, f)) / 1024
        print(f"   {f:<45s}  {sz:7.1f} KB")

    print("\n✓ Step 1 complete — run Step 2:  python step2_qgcrn_model.py")


if __name__ == "__main__":
    main()
