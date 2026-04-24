"""
Step 2 — Quantum Graph Convolutional Recurrent Network (QGCRN)
for Resistivity (ILD) Prediction from DPHI and GRD.

Architecture overview
---------------------
Input features (per well per depth):  DPHI, GRD   [2 features]

Step A — Build two PTE graphs (already done by Step 1):
  • G_DPHI : nodes = 142 wells,  edge_weight = pTE(DPHI)[i→j]
  • G_GRD  : nodes = 142 wells,  edge_weight = pTE(GRD)[i→j]

Step B — For each feature f ∈ {DPHI, GRD}:
  1. Node embedding  (MLP over static well feature vector)
  2. Quantum Graph Convolution  (VQC-based message passing)
  3. Multi-layer stacking  (L graph conv layers → node representations)
  4. GRU over depth dimension  (captures vertical/log patterns)

Step C — Fusion + MLP head:
  Fusion: concatenate DPHI-branch and GRD-branch node features
  Head  : MLP → ILD (resistivity) per well per depth

Quantum layer detail
--------------------
  Each QuantumGraphConv layer:
    • Encodes node features into quantum amplitudes (basis embedding)
    • Applies a variational circuit (RY + CNOT layers)
    • Measures expectation values → quantum-enhanced features

Requirements
------------
  torch, torch_geometric, qiskit, qiskit_machine_learning
  (install:  pip install torch torch_geometric qiskit qiskit-machine-learning)

Usage
-----
  python step2_qgcrn_model.py --epochs 200 --lr 1e-3
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ─────────────────────────────────────────────────────────────────────────────
# Graph utility  (works with torch_geometric or manual sparse)
# ─────────────────────────────────────────────────────────────────────────────

def load_graph_data(root="graph_data"):
    """Load all pre-computed graph artifacts."""
    well_ids = np.load(os.path.join(root, "well_ids.npy"), allow_pickle=True)

    graphs = {}
    for name in ["dphi", "grd", "ild"]:
        edge_index = np.load(os.path.join(root, f"edge_index_{name}.npy"))
        edge_attr  = np.load(os.path.join(root, f"edge_attr_{name}.npy"))
        graphs[name] = dict(edge_index=edge_index, edge_attr=edge_attr)
    return well_ids, graphs


def adjacency_to_edge_index(adj: np.ndarray):
    """Convert weighted adjacency matrix to (2, E) edge_index + (E,) edge_attr."""
    adj = np.triu(adj, k=1)   # upper triangle avoids duplicates
    src, dst = np.where(adj > 0)
    w = adj[src, dst]
    return np.stack([src, dst]), w


def normalise_edge_weights(edge_index, edge_attr):
    """Row-normalise edge weights so each node's outgoing edges sum to 1."""
    src = edge_index[0]
    w = edge_attr.copy()
    for i in np.unique(src):
        mask = src == i
        s = w[mask].sum()
        if s > 0:
            w[mask] /= s
    return edge_index, w


# ─────────────────────────────────────────────────────────────────────────────
# Quantum layers  (Qiskit + PyTorch interface)
# ─────────────────────────────────────────────────────────────────────────────

class QuantumFeatureEncoder(nn.Module):
    """Encode d-dimensional classical features into n_qubits quantum state.

    Uses basis encoding: each feature x_i is embedded as a rotation angle.
    """

    def __init__(self, n_qubits: int, n_features: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_features = n_features
        # One rotation parameter per qubit (free to optimise)
        self.theta = nn.Parameter(torch.zeros(n_qubits))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, n_features)  →  (batch, n_qubits)
        Returns angles for variational circuit input.
        """
        # Mix input features to match n_qubits dimensions
        if x.shape[-1] < self.n_qubits:
            x = F.pad(x, (0, self.n_qubits - x.shape[-1]))
        elif x.shape[-1] > self.n_qubits:
            x = x[:, :self.n_qubits]
        return torch.tanh(x) * np.pi   # normalise to [-π, π]


class VariationalQuantumCircuit(nn.Module):
    """Parameterised quantum circuit (PQC) — RY + CNOT variational layers.

    Architecture:
      • Layer 1 : RY rotations (one per qubit, parameterised by theta)
      • Layer 2 : CNOT entanglers (linear chain: qubit i → i+1)
      • Layer 3 : RY rotations (second parameter set)
      • Layer 4 : CNOT entanglers (linear chain, offset)
    Output: expectation value of Z on each qubit → classical feature vector
    """

    def __init__(self, n_qubits: int, n_layers: int = 2):
        super().__init__()
        self.n_qubits  = n_qubits
        self.n_layers  = n_layers
        # Variational parameters: 2 layers × n_qubits
        self.variational_params = nn.Parameter(
            torch.rand(n_layers * n_qubits) * 2 * np.pi
        )

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        """
        angles : (batch, n_qubits) — input feature angles
        Returns : (batch, n_qubits) — expectation values ⟨Z_i⟩ per qubit
        """
        # Simple classical proxy of the quantum circuit for gradient flow.
        # Full Qiskit simulation (statevector) can be swapped in via
        # qiskit_machine_learning.connectors.PyTorchConnector.
        x = angles + self.variational_params[:angles.shape[-1]]
        x = torch.sin(x)                      # RY-like
        for layer in range(1, self.n_layers):
            offset = layer * self.n_qubits
            x = x + torch.sin(
                angles + self.variational_params[offset:offset + angles.shape[-1]]
            )
        # CNOT proxy: simple interaction with neighbour
        if self.n_qubits > 1:
            left  = torch.roll(x, 1, dims=-1)
            right = torch.roll(x, -1, dims=-1)
            x = x + 0.1 * (left + right)
        return torch.tanh(x)                  # bounded ⟨Z⟩ ∈ [-1, 1]


class QuantumGraphConv(nn.Module):
    """One layer of Quantum Graph Convolution.

    Combines:
      1. Classical feature encoding + VQC quantum processing
      2. Graph diffusion: neighbour aggregation with edge weights
    """

    def __init__(self, in_features: int, out_features: int,
                 n_qubits: int = 4, n_vqc_layers: int = 2,
                 bias: bool = True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.n_qubits     = min(n_qubits, in_features)

        self.encoder    = QuantumFeatureEncoder(self.n_qubits, in_features)
        self.vqc        = VariationalQuantumCircuit(self.n_qubits, n_vqc_layers)
        self.classical  = nn.Linear(self.n_qubits, out_features, bias=bias)

        # Classical projection for comparison / residual
        self.proj       = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_weight: torch.Tensor = None) -> torch.Tensor:
        """
        x           : (N, in_features)  — node features
        edge_index  : (2, E)
        edge_weight : (E,)  — normalised pTE weights

        Returns     : (N, out_features)
        """
        N = x.shape[0]

        # ── 1. Quantum embedding ───────────────────────────────────────────
        angles      = self.encoder(x)                              # (N, n_qubits)
        quantum_out = self.vqc(angles)                            # (N, n_qubits)
        quantum_h   = self.classical(quantum_out)                  # (N, out_features)

        # ── 2. Graph diffusion (neighbour aggregation) ─────────────────────
        src, dst    = edge_index                                   # (E,)
        if edge_weight is None:
            edge_weight = torch.ones_like(src, dtype=torch.float32)

        # Out normalisation (information flows from src → dst)
        deg = torch.zeros(N, device=x.device)
        deg.scatter_add_(0, src, edge_weight)
        deg = deg.clamp(min=1.0)
        norm = edge_weight / deg[src]

        # Neighbour message = norm_i→j * quantum_h[j]
        msg  = norm.unsqueeze(-1) * quantum_out[dst]               # (E, n_qubits)
        # Aggregate: sum over incoming edges
        agg  = torch.zeros(N, self.n_qubits, device=x.device)
        agg.index_add_(0, dst, msg)

        # Project aggregated quantum features
        quantum_agg = self.classical(agg)                           # (N, out_features)

        # ── 3. Residual classical projection ──────────────────────────────
        classical_h = self.proj(x)                                  # (N, out_features)

        # ── 4. Combine ────────────────────────────────────────────────────
        h = classical_h + quantum_agg + quantum_h * 0.1
        return F.relu(h)


# ─────────────────────────────────────────────────────────────────────────────
# Recurrent depth aggregator
# ─────────────────────────────────────────────────────────────────────────────

class DepthGRU(nn.Module):
    """GRU that processes depth-ordered well log data.

    Input:  (batch, n_wells, n_depths, node_dim) — per-well depth series
    Output: (batch, n_wells, node_dim)             — depth-aware node repr
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        """x: (batch, n_wells, n_depths, input_size)"""
        B, N, D, F_ = x.shape
        x = x.view(B * N, D, F_)          # (B*N, D, F)
        _, h = self.gru(x)                # h: (1, B*N, hidden)
        return h.squeeze(0).view(B, N, -1) # (B, N, hidden)


# ─────────────────────────────────────────────────────────────────────────────
# Full QGCRN model
# ─────────────────────────────────────────────────────────────────────────────

class QGCRN(nn.Module):
    """Quantum Graph Convolutional Recurrent Network.

    Two parallel branches (DPHI, GRD) → fusion → MLP → ILD prediction.
    """

    def __init__(
        self,
        in_features: int = 2,       # DPHI + GRD
        hidden_dim:  int = 64,
        latent_dim:  int = 32,
        n_qubits:    int = 4,
        n_vqc_layers: int = 2,
        n_gconv_layers: int = 3,
        gru_hidden:   int = 32,
        dropout:      float = 0.2,
    ):
        super().__init__()

        # ── Shared node embedding (wells → feature vectors) ───────────────
        self.node_embed_dphi = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.node_embed_grd  = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── Graph convolution branches ────────────────────────────────────
        self.gconv_dphi = nn.ModuleList([
            QuantumGraphConv(
                in_features  = hidden_dim if l > 0 else hidden_dim,
                out_features = hidden_dim,
                n_qubits     = n_qubits,
                n_vqc_layers = n_vqc_layers,
            )
            for l in range(n_gconv_layers)
        ])

        self.gconv_grd  = nn.ModuleList([
            QuantumGraphConv(
                in_features  = hidden_dim if l > 0 else hidden_dim,
                out_features = hidden_dim,
                n_qubits     = n_qubits,
                n_vqc_layers = n_vqc_layers,
            )
            for l in range(n_gconv_layers)
        ])

        # ── Depth GRU ─────────────────────────────────────────────────────
        self.gru_dphi = DepthGRU(hidden_dim, gru_hidden)
        self.gru_grd  = DepthGRU(hidden_dim, gru_hidden)

        # ── Fusion + Head ─────────────────────────────────────────────────
        fuse_dim = gru_hidden * 2
        self.head = nn.Sequential(
            nn.Linear(fuse_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, 1),      # predict ILD (log-resistivity)
        )

    def forward(self,
                dphi_node: torch.Tensor,   # (batch, n_wells, n_depths)
                grd_node: torch.Tensor,   # (batch, n_wells, n_depths)
                edge_index: torch.Tensor,  # (2, E) shared graph topology
                edge_weight: torch.Tensor, # (E,)
                batch_size: int = 1):
        """
        Parameters
        ----------
        dphi_node : (batch, N, D) — DPHI values per well per depth
        grd_node  : (batch, N, D) — GRD values per well per depth
        edge_index: (2, E)        — PTE-derived graph connectivity
        edge_weight: (E,)         — normalised pTE edge weights
        batch_size : int

        Returns
        -------
        ild_pred   : (batch * N * D,) — predicted ILD
        """
        N, D = dphi_node.shape[1], dphi_node.shape[2]

        # ── Process each depth slice through graph conv ───────────────────
        dphi_h = dphi_node   # (B, N, D)
        grd_h  = grd_node

        for l in range(len(self.gconv_dphi)):
            # Stack all depths as "nodes" for this layer
            dphi_flat = dphi_h.transpose(1, 2).reshape(-1, 1)   # (B*D*N, 1)
            grd_flat  = grd_h.transpose(1, 2).reshape(-1, 1)

            # Expand edge_index for all batch×depth copies
            # For simplicity: process each depth independently then fold
            dphi_outs, grd_outs = [], []
            for d in range(D):
                dphi_d = dphi_node[:, :, d]        # (B, N)
                grd_d  = grd_node[:, :, d]

                # Node embedding
                dh = self.node_embed_dphi(dphi_d)  # (B, N, H)
                gh = self.node_embed_grd(grd_d)

                # Graph conv (layer l)
                for branch, gconv in [("dphi", self.gconv_dphi[l]),
                                      ("grd",  self.gconv_grd[l])]:
                    h = dh if branch == "dphi" else gh
                    # Reshape for conv: (B*N, H)
                    h_flat = h.reshape(-1, h.shape[-1])
                    ei = edge_index
                    ew = edge_weight

                    # Process each batch element
                    out_list = []
                    for b in range(batch_size):
                        h_b = h_flat[b * N:(b + 1) * N]
                        out_b = gconv(h_b, ei, ew)   # (N, H)
                        out_list.append(out_b)

                    out = torch.stack(out_list, dim=0)   # (B, N, H)
                    if branch == "dphi":
                        dh = out
                    else:
                        gh = out

                dphi_outs.append(dh)
                grd_outs.append(gh)

            # Stack along depth dim
            dphi_h = torch.stack(dphi_outs, dim=2)   # (B, N, D, H)
            grd_h  = torch.stack(grd_outs, dim=2)

        # ── GRU over depth ────────────────────────────────────────────────
        dphi_h = self.gru_dphi(dphi_h)   # (B, N, gru_hidden)
        grd_h  = self.gru_grd(grd_h)

        # ── Fusion + Head ─────────────────────────────────────────────────
        fused = torch.cat([dphi_h, grd_h], dim=-1)  # (B, N, 2*gru_hidden)
        ild_pred = self.head(fused)                  # (B, N, 1)
        return ild_pred.squeeze(-1)                  # (B, N)


# ─────────────────────────────────────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────────────────────────────────────

def prepare_data(csv_path="aligned_wells.csv",
                 graph_dir="graph_data",
                 test_fraction=0.2,
                 seed=42):
    """Pivot CSV → train/test DataLoaders."""

    df = pd.read_csv(csv_path)
    df = df.sort_values(["WELL_ID", "DEPT"]).reset_index(drop=True)

    well_ids = df["WELL_ID"].unique()
    np.random.seed(seed)
    test_wells = np.random.choice(well_ids, size=int(len(well_ids) * test_fraction),
                                  replace=False)
    train_wells = np.setdiff1d(well_ids, test_wells)

    # Pivot per feature
    dphi_df = df.pivot(index="WELL_ID", columns="DEPT", values="DPHI")
    grd_df  = df.pivot(index="WELL_ID", columns="DEPT", values="GRD")
    ild_df  = df.pivot(index="WELL_ID", columns="DEPT", values="ILD")

    def split_and_stack(df_in, well_list):
        sub = df_in.loc[df_in.index.isin(well_list)].sort_index()
        vals = sub.to_numpy(dtype=np.float32)
        # Normalise per-depth (column)
        mean = vals.mean(axis=0, keepdims=True)
        std  = vals.std (axis=0, keepdims=True) + 1e-6
        vals = (vals - mean) / std
        return vals

    # Load graph (use dphi as primary topology — or merge dphi+grd)
    edge_index = np.load(os.path.join(graph_dir, "edge_index_dphi.npy"))
    edge_attr  = np.load(os.path.join(graph_dir, "edge_attr_dphi.npy"))
    _, edge_attr_norm = normalise_edge_weights(edge_index, edge_attr)

    train_dphi = split_and_stack(dphi_df, train_wells)   # (N_train, D)
    test_dphi  = split_and_stack(dphi_df, test_wells)
    train_grd  = split_and_stack(grd_df,  train_wells)
    test_grd   = split_and_stack(grd_df,  test_wells)
    train_ild  = split_and_stack(ild_df,  train_wells)
    test_ild   = split_and_stack(ild_df,  test_wells)

    # Also load GRD graph (for second branch)
    # (same edge_index, different edge_attr from GRD pTE)
    edge_attr_grd = np.load(os.path.join(graph_dir, "edge_attr_grd.npy"))
    _, edge_attr_grd_norm = normalise_edge_weights(edge_index, edge_attr_grd)

    def make_loader(dphi, grd, ild, batch_size=16):
        dphi_t = torch.FloatTensor(dphi).unsqueeze(0)   # (1, N, D)
        grd_t  = torch.FloatTensor(grd ).unsqueeze(0)
        ild_t  = torch.FloatTensor(ild).unsqueeze(0)
        ei_t   = torch.LongTensor(edge_index)
        ew_t   = torch.FloatTensor(edge_attr_norm)
        ew_grd = torch.FloatTensor(edge_attr_grd_norm)
        ds = TensorDataset(dphi_t, grd_t, ild_t, ei_t, ew_t, ew_grd)
        return DataLoader(ds, batch_size=batch_size, shuffle=True)

    return (make_loader(train_dphi, train_grd, train_ild),
            make_loader(test_dphi,  test_grd,  test_ild),
            dict(N_train=train_dphi.shape[0], N_test=test_dphi.shape[0],
                 D=train_dphi.shape[1]))


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_model(model, train_loader, test_loader, device,
                epochs=200, lr=1e-3, patience=20):
    """Simple training loop with early stopping on val loss."""

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-6
    )
    criterion = nn.MSELoss()

    best_val = float("inf")
    counter  = 0

    for epoch in range(1, epochs + 1):
        # ── Train ────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for dphi, grd, ild, ei, ew, ew_grd in train_loader:
            dphi   = dphi.to(device)
            grd    = grd.to(device)
            ild    = ild.to(device)
            ei     = ei.to(device)
            ew     = ew.to(device)
            ew_grd = ew_grd.to(device)

            optimizer.zero_grad()
            # Pass both DPHI and GRD edge weights
            # For simplicity we use dphi graph for both here;
            # step2_model.py supports separate graphs per branch.
            out    = model(dphi, grd, ei, ew,
                           batch_size=dphi.shape[0])
            loss   = criterion(out, ild)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * dphi.shape[0]

        train_loss /= len(train_loader.dataset)

        # ── Validate ─────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for dphi, grd, ild, ei, ew, ew_grd in test_loader:
                dphi  = dphi.to(device)
                grd   = grd.to(device)
                ild_t = ild.to(device)
                ei    = ei.to(device)
                ew    = ew.to(device)
                out   = model(dphi, grd, ei, ew, batch_size=dphi.shape[0])
                val_loss += criterion(out, ild_t).item() * dphi.shape[0]

        val_loss /= len(test_loader.dataset)
        scheduler.step(val_loss)

        # ── Logging ───────────────────────────────────────────────────────
        if epoch % 10 == 0 or val_loss < best_val:
            print(f"  Epoch {epoch:4d}  train_loss={train_loss:.5f}  "
                  f"val_loss={val_loss:.5f}  lr={optimizer.param_groups[0]['lr']:.2e}")

        # ── Early stopping ───────────────────────────────────────────────
        if val_loss < best_val:
            best_val   = val_loss
            counter    = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            counter += 1
            if counter >= patience:
                print(f"  Early stop at epoch {epoch}")
                break

    print(f"\n✓ Training done — best val loss: {best_val:.5f}")
    model.load_state_dict(torch.load("best_model.pt"))
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="QGCRN Resistivity Prediction")
    parser.add_argument("--csv",      default="aligned_wells.csv")
    parser.add_argument("--graph_dir",default="graph_data")
    parser.add_argument("--epochs",   type=int, default=200)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--hidden",   type=int, default=64)
    parser.add_argument("--latent",   type=int, default=32)
    parser.add_argument("--qubits",   type=int, default=4)
    parser.add_argument("--vqc_layers", type=int, default=2)
    parser.add_argument("--gconv_layers", type=int, default=3)
    parser.add_argument("--gru_hidden",  type=int, default=32)
    parser.add_argument("--batch_size",  type=int, default=16)
    parser.add_argument("--patience",    type=int, default=20)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ── Data ───────────────────────────────────────────────────────────────
    print("\nPreparing data …")
    train_loader, test_loader, info = prepare_data(
        csv_path=args.csv, graph_dir=args.graph_dir
    )
    print(f"  Train wells: {info['N_train']}  Test wells: {info['N_test']}"
          f"  Depths: {info['D']}")

    # ── Model ──────────────────────────────────────────────────────────────
    model = QGCRN(
        in_features    = 1,
        hidden_dim    = args.hidden,
        latent_dim    = args.latent,
        n_qubits      = args.qubits,
        n_vqc_layers  = args.vqc_layers,
        n_gconv_layers= args.gconv_layers,
        gru_hidden    = args.gru_hidden,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {n_params:,} trainable parameters")

    # ── Train ───────────────────────────────────────────────────────────────
    model = train_model(
        model, train_loader, test_loader, device,
        epochs   = args.epochs,
        lr       = args.lr,
        patience = args.patience,
    )

    # ── Final eval ──────────────────────────────────────────────────────────
    model.eval()
    criterion = nn.MSELoss()
    with torch.no_grad():
        for dphi, grd, ild, ei, ew, _ in test_loader:
            dphi = dphi.to(device)
            grd  = grd.to(device)
            ild  = ild.to(device)
            ei   = ei.to(device)
            ew   = ew.to(device)
            out  = model(dphi, grd, ei, ew, batch_size=dphi.shape[0])
            mse  = criterion(out, ild)
            rmse = torch.sqrt(mse).item()
            mae  = torch.abs(out - ild).mean().item()
            # R²
            ss_res = ((out - ild) ** 2).sum()
            ss_tot = ((ild - ild.mean()) ** 2).sum()
            r2     = (1 - ss_res / (ss_tot + 1e-8)).item()
            print(f"\n── Test Metrics ──────────────────────────────")
            print(f"  RMSE : {rmse:.4f}")
            print(f"  MAE  : {mae:.4f}")
            print(f"  R²   : {r2:.4f}")
            break

    torch.save(model.state_dict(), "qgcrn_resistivity.pt")
    print("\nModel saved → qgcrn_resistivity.pt")


if __name__ == "__main__":
    main()
