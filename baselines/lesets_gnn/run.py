"""
LESets-inspired GNN baseline for HEA dataset.

Reference: Zhang et al. (2024-25), "Do Graph Neural Networks Work for High Entropy
Alloys?" arXiv:2408.16337  (https://github.com/Henrium/LESets)

Original LESets architecture (per-composition):
    - For each composition, build a *set* of local-environment (LE) graphs
      derived from atomic configurations.
    - phi(GNN) embeds each LE graph -> graph embedding.
    - Aggregate set embeddings weighted by composition fractions.
    - rho(MLP) -> scalar property.

Our HEA dataset (CoCrCuFeNi 696, 232 compositions x 3 temperatures) has *only*
composition + temperature, with no per-atom structural information.  We
therefore implement a *simplified* LESets that preserves the architectural
spirit (GNN embedding -> composition-fraction-weighted set aggregation ->
MLP head), but uses a single composition-level graph per sample:

    - Nodes: 5 element nodes (Co, Cr, Cu, Fe, Ni).
    - Node features: composition fraction + elemental descriptors
      (atomic number, atomic mass, atomic radius, electronegativity, VEC,
       melting T, density, cohesive energy).
    - Edges: fully-connected (4 neighbours each, no self-loops).
    - Edge attributes: pair-wise descriptors:
        * Miedema-style mixing-enthalpy proxy from electronegativity diff
        * radius mismatch
        * VEC difference
        * fraction product (xi * xj)
    - Global feature: temperature T (concatenated to graph embedding before MLP).

GNN: 2 x CGConv layers (LESets uses CGConv among others; CGConv consumes edge
attributes directly).  Aggregation: composition-fraction-weighted node sum
(LESets-style "phi -> weighted sum -> rho").  Head: 3-layer MLP -> 1 scalar.

Protocol:
    - 5-fold composition-grouped CV (GroupKFold over comp_id).
    - 5 seeds (0..4).
    - 12 targets.
    - Reports per-target mean R^2 across folds*seeds.

NOTE: This is the *simplified* LESets variant (composition-graph version).
The exact LESets-on-LE-graphs is not applicable without atomic structure data
for our compositions, so we report results as "LESets-inspired (simplified)".
"""

from pathlib import Path
import json
import os
import time
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data, Batch
from torch_geometric.nn import CGConv

warnings.filterwarnings("ignore")

# ---- Reproducibility / threading -------------------------------------------
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
torch.set_num_threads(2)
DEVICE = torch.device("cpu")


# ---- Paths ------------------------------------------------------------------
DATA_FILE = str(Path(__file__).resolve().parent.parent / "data" / "raw" / "CoCrCuFeNi_684.csv")
OUT_DIR = str(Path(__file__).resolve().parent.parent / "results" / "generated" / "lesets_gnn")
RESULTS_PATH = os.path.join(OUT_DIR, "results.json")
SUMMARY_PATH = os.path.join(OUT_DIR, "summary.md")
CKPT_DIR = os.path.join(OUT_DIR, "ckpts")
os.makedirs(CKPT_DIR, exist_ok=True)


# ---- Data definitions -------------------------------------------------------
TEMPS = {"80K": 80, "300K": 300, "1100K": 1100}
ELEMENTS = ["Co", "Cr", "Cu", "Fe", "Ni"]
COMP_COLS = [f"{e}(%)" for e in ELEMENTS]

TARGETS = {
    "Youngs_modulus": "Young's modulus(Gpa)",
    "UTS": "UTS(Gpa)",
    "Disloc_0pct": "Dislocation density 0%(×10¹⁷ m⁻²)",
    "Disloc_20pct": "Dislocation density 20%(×10¹⁷ m⁻²)",
    "FCC_0pct": "FCC 0%(%)",
    "HCP_0pct": "HCP 0%(%)",
    "BCC_0pct": "BCC 0%(%)",
    "FCC_20pct": "FCC 20%(%)",
    "HCP_20pct": "HCP 20%(%)",
    "BCC_20pct": "BCC 20%(%)",
    "Other_0pct": "Other 0%(%)",
    "Other_20pct": "Other 20%(%)",
}

# Elemental descriptors (Co, Cr, Cu, Fe, Ni)
# atomic number, atomic mass (u), atomic radius (pm), Pauling EN,
# valence electron count (VEC for d-block: s + d), melting T (K), density (g/cm^3),
# cohesive energy (eV/atom)
ELEM_DESC = {
    # element : [Z, mass, r_pm, EN, VEC, Tm_K, rho, Ecoh_eV]
    "Co": [27, 58.93, 125.0, 1.88, 9.0, 1768.0, 8.90, 4.39],
    "Cr": [24, 52.00, 128.0, 1.66, 6.0, 2180.0, 7.19, 4.10],
    "Cu": [29, 63.55, 128.0, 1.90, 11.0, 1357.8, 8.96, 3.49],
    "Fe": [26, 55.85, 126.0, 1.83, 8.0, 1811.0, 7.87, 4.28],
    "Ni": [28, 58.69, 124.0, 1.91, 10.0, 1728.0, 8.90, 4.44],
}
DESC_KEYS = ["Z", "mass", "r_pm", "EN", "VEC", "Tm", "rho", "Ecoh"]


# ---- Load data --------------------------------------------------------------
def load_data():
    dfs = []
    for sheet, t in TEMPS.items():
        df = pd.read_excel(DATA_FILE, sheet_name=sheet)
        df["T"] = t
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    data["comp_id"] = (
        data["Co(%)"].astype(str)
        + "_" + data["Cr(%)"].astype(str)
        + "_" + data["Cu(%)"].astype(str)
        + "_" + data["Fe(%)"].astype(str)
        + "_" + data["Ni(%)"].astype(str)
    )
    return data


# ---- Graph construction -----------------------------------------------------
NODE_FEAT_DIM = 1 + len(DESC_KEYS)  # frac + 8 descriptors
EDGE_FEAT_DIM = 4                    # |dEN|, |dr|, |dVEC|, xi*xj


def build_graph(comp_at_pct, T_K):
    """Build a 5-node fully-connected graph for one (composition, T) sample."""
    fracs = np.asarray(comp_at_pct, dtype=np.float64) / 100.0
    fracs = fracs / max(fracs.sum(), 1e-12)

    # Node features: [frac, Z, mass, r, EN, VEC, Tm, rho, Ecoh]
    desc_arr = np.array([ELEM_DESC[e] for e in ELEMENTS], dtype=np.float64)
    node_feats = np.concatenate([fracs[:, None], desc_arr], axis=1)
    x = torch.tensor(node_feats, dtype=torch.float32)

    # Fully-connected edges (no self-loops)
    src, dst, ea = [], [], []
    for i in range(5):
        for j in range(5):
            if i == j:
                continue
            src.append(i)
            dst.append(j)
            d_en = abs(desc_arr[i, 3] - desc_arr[j, 3])
            d_r = abs(desc_arr[i, 2] - desc_arr[j, 2])
            d_vec = abs(desc_arr[i, 4] - desc_arr[j, 4])
            xij = fracs[i] * fracs[j]
            ea.append([d_en, d_r, d_vec, xij])
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(ea, dtype=torch.float32)

    # Composition fractions used for LESets-style aggregation
    frac_t = torch.tensor(fracs, dtype=torch.float32)

    # Global temperature
    g = torch.tensor([T_K], dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                frac=frac_t, T=g)


# ---- Model ------------------------------------------------------------------
class LESetsSimpleGNN(nn.Module):
    """
    Simplified LESets:
      - 2x CGConv message-passing on the 5-element composition graph.
      - LESets-style aggregation: per-node embeddings weighted by composition
        fraction, summed -> graph embedding.
      - Concatenate global feature T -> 3-layer MLP -> scalar.
    """
    def __init__(self, node_dim=NODE_FEAT_DIM, edge_dim=EDGE_FEAT_DIM,
                 hidden=64, mlp_hidden=64):
        super().__init__()
        self.node_encoder = nn.Linear(node_dim, hidden)
        self.conv1 = CGConv(channels=hidden, dim=edge_dim, batch_norm=False)
        self.conv2 = CGConv(channels=hidden, dim=edge_dim, batch_norm=False)
        self.act = nn.SiLU()
        # rho: 3-layer MLP head as in LESets
        self.head = nn.Sequential(
            nn.Linear(hidden + 1, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, data):
        x, ei, ea = data.x, data.edge_index, data.edge_attr
        h = self.node_encoder(x)
        h = self.act(self.conv1(h, ei, ea))
        h = self.act(self.conv2(h, ei, ea))

        # LESets-style: per-graph weighted sum by composition fraction.
        # data.batch maps each node to its sample index.
        frac = data.frac.unsqueeze(-1)            # [N_nodes, 1]
        weighted = h * frac                         # [N_nodes, hidden]
        # scatter-sum into per-graph embeddings
        n_graphs = int(data.batch.max().item()) + 1
        g_emb = torch.zeros(n_graphs, h.size(1), device=h.device)
        g_emb.index_add_(0, data.batch, weighted)

        # Concatenate global T (one scalar per graph)
        # data.T is shape [n_graphs] (one per graph) since each graph has 1 T entry
        T = data.T.view(n_graphs, 1)
        gT = torch.cat([g_emb, T], dim=-1)
        out = self.head(gT).squeeze(-1)
        return out


# ---- Training utilities -----------------------------------------------------
def make_loader(graphs, batch_size=64, shuffle=False):
    """Simple manual batching using PyG Batch.from_data_list."""
    idx = np.arange(len(graphs))
    if shuffle:
        np.random.shuffle(idx)
    batches = []
    for s in range(0, len(idx), batch_size):
        sub = [graphs[i] for i in idx[s:s + batch_size]]
        batches.append(Batch.from_data_list(sub))
    return batches


def train_one(model, train_graphs, val_graphs, y_train, y_val,
              max_epochs=400, patience=40, lr=1e-3, wd=1e-4,
              batch_size=64, verbose=False):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()

    # Prebuild val batch (single batch)
    val_batch = Batch.from_data_list(val_graphs).to(DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=DEVICE)

    best_val = float("inf")
    best_state = None
    bad = 0
    for ep in range(max_epochs):
        model.train()
        # Shuffle indices
        idx = np.random.permutation(len(train_graphs))
        running = 0.0
        n_seen = 0
        for s in range(0, len(idx), batch_size):
            sub = [train_graphs[i] for i in idx[s:s + batch_size]]
            yb = torch.tensor(y_train[idx[s:s + batch_size]],
                              dtype=torch.float32, device=DEVICE)
            batch = Batch.from_data_list(sub).to(DEVICE)
            opt.zero_grad()
            pred = model(batch)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            running += loss.item() * len(sub)
            n_seen += len(sub)
        train_loss = running / max(n_seen, 1)

        model.eval()
        with torch.no_grad():
            vp = model(val_batch)
            vloss = loss_fn(vp, y_val_t).item()

        if vloss < best_val - 1e-6:
            best_val = vloss
            best_state = deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

        if verbose and (ep % 25 == 0):
            print(f"   ep {ep:03d} train {train_loss:.4f} val {vloss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict(model, graphs):
    model.eval()
    batch = Batch.from_data_list(graphs).to(DEVICE)
    with torch.no_grad():
        p = model(batch).cpu().numpy()
    return p


# ---- Main pipeline ----------------------------------------------------------
def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)


def main():
    print("=" * 70)
    print("LESets-inspired GNN baseline (simplified, composition-graph)")
    print("=" * 70)

    data = load_data()
    print(f"Total samples: {len(data)} (compositions x T)")
    print(f"Unique compositions: {data['comp_id'].nunique()}")

    # Build all graphs once (no T-dependent node features, only T as global)
    all_graphs = []
    for _, row in data.iterrows():
        comp = [row[c] for c in COMP_COLS]
        g = build_graph(comp, float(row["T"]))
        all_graphs.append(g)
    print(f"Built {len(all_graphs)} graphs.")

    groups = data["comp_id"].values
    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(np.zeros(len(data)), np.zeros(len(data)), groups))

    SEEDS = [0, 1, 2, 3, 4]

    results = {
        "method": "LESets-inspired GNN (simplified, composition-graph; CGConv x2 + LESets aggregation)",
        "lesets_variant": "simplified (composition-graph, no LE graphs available)",
        "framework": "PyTorch Geometric",
        "n_folds": 5,
        "seeds": SEEDS,
        "targets": {},
    }

    overall_t0 = time.time()
    for tname, tcol in TARGETS.items():
        print("\n" + "=" * 60)
        print(f"Target: {tname} ({tcol})")
        print("=" * 60)

        y_all = data[tcol].values.astype(np.float64)
        # Standardise targets per-target globally for stable training; we
        # invert before computing R^2.
        y_scaler = StandardScaler()
        y_std = y_scaler.fit_transform(y_all.reshape(-1, 1)).ravel()

        per_seed_fold_r2 = []   # (seed, fold) -> r2 (on original scale)
        per_seed_fold_mae = []
        t_target = time.time()

        for seed in SEEDS:
            set_seed(seed)
            seed_r2s, seed_maes = [], []
            for fi, (tr_idx, te_idx) in enumerate(folds):
                # Inner val from train (last 15% by composition)
                tr_groups = groups[tr_idx]
                uniq = np.unique(tr_groups)
                rng = np.random.RandomState(seed * 100 + fi)
                rng.shuffle(uniq)
                n_val_g = max(1, int(0.15 * len(uniq)))
                val_g = set(uniq[:n_val_g].tolist())
                in_val = np.array([g in val_g for g in tr_groups])
                tr_only = tr_idx[~in_val]
                vl_only = tr_idx[in_val]

                tr_graphs = [all_graphs[i] for i in tr_only]
                vl_graphs = [all_graphs[i] for i in vl_only]
                te_graphs = [all_graphs[i] for i in te_idx]

                y_tr = y_std[tr_only]
                y_vl = y_std[vl_only]

                model = LESetsSimpleGNN().to(DEVICE)
                model = train_one(
                    model, tr_graphs, vl_graphs, y_tr, y_vl,
                    max_epochs=400, patience=40, lr=1e-3, wd=1e-4,
                    batch_size=64,
                )

                # Predict test in original scale
                p_std = predict(model, te_graphs)
                p_orig = y_scaler.inverse_transform(p_std.reshape(-1, 1)).ravel()
                y_te = y_all[te_idx]
                r2 = r2_score(y_te, p_orig)
                mae = mean_absolute_error(y_te, p_orig)
                seed_r2s.append(r2)
                seed_maes.append(mae)

                # Save the very first fold/seed checkpoint per target
                if seed == SEEDS[0] and fi == 0:
                    ck = os.path.join(CKPT_DIR, f"{tname}_seed{seed}_fold{fi}.pt")
                    torch.save(model.state_dict(), ck)

            per_seed_fold_r2.append(seed_r2s)
            per_seed_fold_mae.append(seed_maes)
            print(
                f"  seed {seed}: fold R2={np.round(seed_r2s, 3).tolist()} "
                f"mean={np.mean(seed_r2s):.3f}"
            )

        arr = np.array(per_seed_fold_r2)            # [n_seeds, n_folds]
        arr_mae = np.array(per_seed_fold_mae)
        results["targets"][tname] = {
            "r2_mean": float(arr.mean()),
            "r2_std": float(arr.std()),
            "r2_per_seed_mean": [float(x) for x in arr.mean(axis=1)],
            "r2_per_seed_fold": arr.tolist(),
            "mae_mean": float(arr_mae.mean()),
            "mae_std": float(arr_mae.std()),
            "n_seeds": len(SEEDS),
            "n_folds": 5,
        }
        print(
            f"  -> mean R2 = {arr.mean():.3f} +/- {arr.std():.3f} "
            f"(MAE {arr_mae.mean():.3g})  [{time.time()-t_target:.1f}s]"
        )

        # Persist incremental results after every target
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)

    overall = np.mean([v["r2_mean"] for v in results["targets"].values()])
    results["overall_r2_mean"] = float(overall)
    results["wall_time_sec"] = float(time.time() - overall_t0)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    # Summary md
    lines = []
    lines.append("# LESets-inspired GNN (simplified) — HEA CoCrCuFeNi 696\n")
    lines.append(
        "Baseline implementation of Zhang et al. (arXiv:2408.16337) LESets-style "
        "GNN, *simplified* for composition-only data (no atomic structure):\n"
    )
    lines.append(
        "- Single 5-element fully-connected graph per (composition, T).\n"
        "- Node features: composition fraction + 8 elemental descriptors "
        "(Z, mass, r, EN, VEC, Tm, rho, Ecoh).\n"
        "- Edge attributes: |dEN|, |dr|, |dVEC|, xi*xj.\n"
        "- Two CGConv layers (LESets uses CGConv as one option), SiLU activations.\n"
        "- LESets-style aggregation: per-node embedding * composition fraction, "
        "summed to graph embedding (rho-input).\n"
        "- Concatenate temperature T, then 3-layer MLP head -> scalar.\n"
    )
    lines.append("\nProtocol: 5-fold composition-grouped CV x 5 seeds (0..4).\n")
    lines.append("\n| Target | R^2 mean | R^2 std | MAE mean |")
    lines.append("|---|---:|---:|---:|")
    for tname, v in results["targets"].items():
        lines.append(
            f"| {tname} | {v['r2_mean']:.3f} | {v['r2_std']:.3f} | {v['mae_mean']:.3g} |"
        )
    lines.append(f"\n**Overall mean R^2 across 12 targets: {overall:.3f}**\n")
    lines.append(
        "\n*LESets exact (per-LE-graph DeepSets) was not applicable to our "
        "composition-only dataset; this is the simplified composition-graph variant "
        "that preserves the LESets architectural spirit (GNN -> "
        "fraction-weighted aggregation -> rho MLP).*\n"
    )
    with open(SUMMARY_PATH, "w") as f:
        f.write("\n".join(lines))

    print("\n" + "=" * 70)
    print(f"Overall mean R^2 across {len(TARGETS)} targets: {overall:.3f}")
    print(f"Results: {RESULTS_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    print(f"Wall time: {time.time()-overall_t0:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
