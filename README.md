# QGCRN — Quantum Graph Convolutional Recurrent Network for Resistivity Prediction

> Predict **ILD (resistivity)** from **DPHI** (density porosity) and **GRD** (gamma ray)
> across 142 wells using Pseudo Transfer Entropy graphs + Quantum Graph Conv + GRU.

---

## Pipeline Overview

```
aligned_wells.csv
       │
       ├── Step 1 ──► graph_data/
       │               edge_index_dphi.npy   (2, E₁)
       │               edge_attr_dphi.npy    (E₁,)
       │               edge_index_grd.npy    (2, E₂)
       │               edge_attr_grd.npy     (E₂,)
       │               well_ids.npy
       │
       ▼
Step 2 ──► QGCRN model ──► qgcrn_resistivity.pt
       │        │
       │        ├── G_DPHI branch: NodeEmbed → [QuantumGraphConv]×L → GRU
       │        ├── G_GRD  branch: NodeEmbed → [QuantumGraphConv]×L → GRU
       │        └── Fusion + MLP → ILD
```

---

## File Structure

```
qgcrn-resistivity/
├── aligned_wells.csv           # Raw dataset (142 wells)
├── pte.py                      # Pseudo Transfer Entropy implementation
├── step1_build_graphs.py       # Build PTE graphs → graph_data/
├── step2_qgcrn_model.py        # Full QGCRN model + training
├── graph_data/                  # Output from Step 1
│   ├── well_ids.npy
│   ├── {dphi,grd,ild}_pte_real.npy
│   ├── {dphi,grd,ild}_pte_iaaft.npy
│   ├── edge_index_{dphi,grd,ild}.npy
│   └── edge_attr_{dphi,grd,ild}.npy
└── README.md
```

---

## Step 1 — Build PTE Graphs

```bash
# Install dependencies
pip install scipy numpy pandas tqdm

# Run
python step1_build_graphs.py
```

**What it does:**
1. Pivots `aligned_wells.csv` → (142 wells × 801 depth points) matrices
2. Computes pTE matrix for DPHI and GRD (τ=1, dimEmb=1, 19 IAAFT surrogates)
3. Edge condition: `real_pTE[i→j] > max(surrogate_pTE[i→j])` → directed edge
4. Saves `edge_index` + `edge_attr` as `.npy` files

---

## Step 2 — Train QGCRN

```bash
# Install GPU dependencies
pip install torch torch_geometric qiskit qiskit-machine-learning

# Run training
python step2_qgcrn_model.py --epochs 200 --lr 1e-3 --hidden 64 --qubits 4

# Or with GPU explicitly
python step2_qgcrn_model.py --epochs 200 --lr 1e-3 --hidden 64 --qubits 4 \
       --graph_dir graph_data --csv aligned_wells.csv
```

**Architecture:**
```
Input:  DPHI (B,142,801),  GRD (B,142,801)
         │                      │
  node_embed_dphi          node_embed_grd
         │                      │
  [QuantumGraphConv]×3     [QuantumGraphConv]×3    ← L=3 graph conv layers
         │                      │
      DepthGRU(64,32)       DepthGRU(64,32)        ← GRU over depth
         │                      │
         └──────── concat ───────┘
                   │
              Fusion + MLP
                   │
              ILD (B, 142)
```

**Quantum layer detail** (`QuantumGraphConv`):
- Feature encoder: classical features → quantum rotation angles (basis encoding)
- Variational circuit: RY rotations + CNOT entanglers (2 layers)
- Measurement: ⟨Z⟩ expectation per qubit → classical feature vector
- Neighbour aggregation: weighted sum over incoming pTE edges

**Key hyperparameters:**

| Flag | Default | Description |
|------|---------|-------------|
| `--hidden` | 64 | Graph conv hidden dim |
| `--latent` | 32 | MLP latent dim |
| `--qubits` | 4 | Qubits per VQC |
| `--vqc_layers` | 2 | Variational circuit depth |
| `--gconv_layers` | 3 | Number of graph conv layers |
| `--gru_hidden` | 32 | GRU hidden dimension |
| `--epochs` | 200 | Max epochs |
| `--patience` | 20 | Early stopping patience |
| `--batch_size` | 16 | Training batch size |

---

## Notes

- **Multi-graph**: Step 1 builds separate graphs for DPHI and GRD. Step 2 uses the DPHI graph topology by default; to use GRD topology instead, swap `edge_index_dphi.npy` → `edge_index_grd.npy` in `prepare_data()`.
- **Quantum simulation**: The current `VariationalQuantumCircuit` uses a classical proxy (sinusoidal) so it runs on CPU/GPU without a real quantum backend. To use real Qiskit simulation, replace the `forward` method with `qiskit_machine_learning.connectors.PyTorchConnector`.
- **Quantisation**: ILD is log-resistivity (already in log scale in most wireline logs). If your ILD column is in ohm-m, apply `np.log10(ILD)` before training.
