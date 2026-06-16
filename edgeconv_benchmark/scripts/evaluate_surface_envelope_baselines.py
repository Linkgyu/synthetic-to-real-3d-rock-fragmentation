from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.scene_dataset import balanced_edge_indices, edge_geom_features, load_npz_scene, load_scene_index  # noqa: E402
from src.fragmentation.psd import cumulative_psd, percentile_size  # noqa: E402
from src.fragmentation.surface_proxy import estimate_surface_proxy  # noqa: E402
from src.segmentation.components import components_from_edge_probabilities  # noqa: E402
from src.segmentation.metrics import clustering_scores  # noqa: E402


OUT_MODELS = ROOT / "outputs" / "models"
OUT_TABLES = ROOT / "outputs" / "tables"
OUT_FIGURES = ROOT / "outputs" / "figures"


def relabel_nonnegative(labels: np.ndarray, min_cluster_points: int = 10) -> np.ndarray:
    labels = np.asarray(labels, dtype=int).copy()
    valid = labels >= 0
    if np.any(valid):
        counts = np.bincount(labels[valid])
        small = np.isin(labels, np.flatnonzero(counts < min_cluster_points))
        labels[small] = -1
    out = np.full(len(labels), -1, dtype=int)
    valid = labels >= 0
    if np.any(valid):
        _, inv = np.unique(labels[valid], return_inverse=True)
        out[valid] = inv
    return out


def p80_from_labels(points_xyz: np.ndarray, labels: np.ndarray) -> float:
    sizes = estimate_surface_proxy(points_xyz, labels, min_points=10)
    if sizes.empty:
        return float("nan")
    psd = cumulative_psd(sizes["diameter_proxy_m"].to_numpy(), sizes["proxy_volume_m3"].to_numpy())
    return percentile_size(psd, 80.0)


def evaluate_labels(scene: dict, labels: np.ndarray, method: str, params: dict) -> dict:
    true_p80 = float(scene["ground_truth_P80_mm"][0])
    pred_p80 = p80_from_labels(scene["points_xyz"], labels)
    row = {
        "method": method,
        "predicted_P80_mm": pred_p80,
        "ground_truth_P80_mm": true_p80,
        "P80_error_pct": (pred_p80 - true_p80) / true_p80 * 100.0 if np.isfinite(pred_p80) else float("nan"),
        "abs_P80_error_pct": abs(pred_p80 - true_p80) / true_p80 * 100.0 if np.isfinite(pred_p80) else float("nan"),
        "abs_P80_error_mm": abs(pred_p80 - true_p80) if np.isfinite(pred_p80) else float("nan"),
        **clustering_scores(scene["instance_labels"], labels),
    }
    for key, value in params.items():
        row[key] = value
    return row


def region_growing_labels(
    scene: dict,
    radius_m: float,
    max_normal_angle_deg: float,
    max_curvature_delta: float,
    max_vertical_jump_m: float,
    min_cluster_points: int = 10,
) -> np.ndarray:
    points = np.asarray(scene["points_xyz"], dtype=float)
    normals = np.asarray(scene["normals"], dtype=float)
    curvature = np.asarray(scene["curvature"], dtype=float)
    pairs = np.array(list(cKDTree(points).query_pairs(radius_m)), dtype=np.int64)
    n = len(points)
    parent = np.arange(n, dtype=np.int64)
    rank = np.zeros(n, dtype=np.int8)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return int(x)

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    if len(pairs):
        i, j = pairs[:, 0], pairs[:, 1]
        normal_ok = np.abs(np.sum(normals[i] * normals[j], axis=1)) >= np.cos(np.deg2rad(max_normal_angle_deg))
        curvature_ok = np.abs(curvature[i] - curvature[j]) <= max_curvature_delta
        vertical_ok = np.abs(points[i, 2] - points[j, 2]) <= max_vertical_jump_m
        for a, b in pairs[normal_ok & curvature_ok & vertical_ok]:
            union(int(a), int(b))

    roots = np.array([find(i) for i in range(n)], dtype=np.int64)
    _, labels = np.unique(roots, return_inverse=True)
    return relabel_nonnegative(labels, min_cluster_points=min_cluster_points)


def simple_edge_score(scene: dict) -> np.ndarray:
    points = scene["points_xyz"]
    normals = scene["normals"]
    curvature = scene["curvature"]
    edges = scene["edges"]
    i, j = edges[:, 0], edges[:, 1]
    dist = np.linalg.norm(points[i] - points[j], axis=1)
    dz = np.abs(points[i, 2] - points[j, 2])
    normal_sim = np.abs(np.sum(normals[i] * normals[j], axis=1))
    curv_delta = np.abs(curvature[i] - curvature[j])
    dist_score = np.exp(-dist / 0.080)
    dz_score = np.exp(-dz / 0.055)
    curv_score = np.exp(-curv_delta / 0.060)
    return 0.42 * normal_sim + 0.28 * dist_score + 0.18 * dz_score + 0.12 * curv_score


def dbscan_labels(scene: dict, eps_m: float, min_samples: int, z_weight: float = 1.0) -> np.ndarray:
    points = np.asarray(scene["points_xyz"], dtype=float).copy()
    points[:, 2] *= float(z_weight)
    labels = DBSCAN(eps=float(eps_m), min_samples=int(min_samples), n_jobs=-1).fit_predict(points)
    return relabel_nonnegative(labels, min_cluster_points=10)


def collect_mlp_training_samples(train_rows: pd.DataFrame, max_edges_per_scene: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for row_idx, row in train_rows.reset_index(drop=True).iterrows():
        scene = load_npz_scene(row)
        idx = balanced_edge_indices(scene["edge_same_fragment"], max_edges=max_edges_per_scene, seed=seed + row_idx)
        xs.append(edge_geom_features(scene["edge_features"])[idx])
        ys.append(scene["edge_same_fragment"][idx].astype(int))
    return np.vstack(xs), np.concatenate(ys)


def train_mlp_baseline(train_rows: pd.DataFrame, args: argparse.Namespace) -> Pipeline:
    x, y = collect_mlp_training_samples(train_rows, max_edges_per_scene=args.mlp_edges_per_scene, seed=args.seed)
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(128, 96, 48),
                    activation="relu",
                    solver="adam",
                    alpha=1.0e-4,
                    batch_size=4096,
                    learning_rate_init=1.0e-3,
                    max_iter=args.mlp_max_iter,
                    random_state=args.seed,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=6,
                    verbose=False,
                ),
            ),
        ]
    )
    model.fit(x, y)
    prob = model.predict_proba(x)[:, 1]
    print(
        "MLP train sample metrics: "
        f"AP={average_precision_score(y, prob):.3f}, AUC={roc_auc_score(y, prob):.3f}, "
        f"n_edges={len(y)}",
        flush=True,
    )
    OUT_MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUT_MODELS / "surface_envelope_edge_affinity_mlp.joblib")
    return model


def evaluate_method_on_rows(method: str, rows: pd.DataFrame, params_list: list[dict], mlp_model: Pipeline | None = None) -> pd.DataFrame:
    out = []
    for scene_idx, row in rows.reset_index(drop=True).iterrows():
        scene = load_npz_scene(row)
        for params in params_list:
            if method == "DBSCAN":
                labels = dbscan_labels(scene, **params)
            elif method == "Region growing normals/curvature":
                labels = region_growing_labels(scene, **params)
            elif method == "Simple graph threshold":
                scores = simple_edge_score(scene)
                labels = components_from_edge_probabilities(
                    len(scene["points_xyz"]),
                    scene["edges"],
                    scores,
                    threshold=params["score_threshold"],
                    min_cluster_points=10,
                )
            elif method == "MLP edge-affinity":
                if mlp_model is None:
                    raise ValueError("MLP model required")
                probs = mlp_model.predict_proba(edge_geom_features(scene["edge_features"]))[:, 1]
                labels = components_from_edge_probabilities(
                    len(scene["points_xyz"]),
                    scene["edges"],
                    probs,
                    threshold=params["edge_threshold"],
                    min_cluster_points=10,
                )
            else:
                raise ValueError(method)
            result = evaluate_labels(scene, labels, method=method, params=params)
            result["scene_id"] = int(row["scene_id"])
            result["split"] = row["split"]
            out.append(result)
        print(f"{method}: {scene_idx + 1}/{len(rows)} scenes", flush=True)
    return pd.DataFrame(out)


def best_params_by_validation(df: pd.DataFrame) -> pd.DataFrame:
    param_cols = [c for c in df.columns if c not in {
        "method",
        "scene_id",
        "split",
        "predicted_P80_mm",
        "ground_truth_P80_mm",
        "P80_error_pct",
        "abs_P80_error_pct",
        "abs_P80_error_mm",
        "adjusted_rand_index",
        "normalized_mutual_info",
        "n_true_fragments",
        "n_predicted_clusters",
        "noise_fraction",
    }]
    grouped = (
        df.groupby(["method", *param_cols], dropna=False, as_index=False)
        .agg(
            mean_abs_P80_error_pct=("abs_P80_error_pct", "mean"),
            median_abs_P80_error_pct=("abs_P80_error_pct", "median"),
            mean_NMI=("normalized_mutual_info", "mean"),
            mean_ARI=("adjusted_rand_index", "mean"),
            mean_noise_fraction=("noise_fraction", "mean"),
        )
        .sort_values(["method", "mean_abs_P80_error_pct", "median_abs_P80_error_pct"])
    )
    return grouped.groupby("method", as_index=False).head(1).reset_index(drop=True)


def params_from_best(best_row: pd.Series) -> dict:
    skip = {
        "method",
        "mean_abs_P80_error_pct",
        "median_abs_P80_error_pct",
        "mean_NMI",
        "mean_ARI",
        "mean_noise_fraction",
    }
    params = {k: best_row[k] for k in best_row.index if k not in skip and pd.notna(best_row[k])}
    if "min_samples" in params:
        params["min_samples"] = int(params["min_samples"])
    for key in list(params):
        if isinstance(params[key], np.generic):
            params[key] = params[key].item()
    return params


def append_edgeconv_results(test_rows: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    edgeconv_path = OUT_TABLES / "edgeconv_test_results.csv"
    if not edgeconv_path.exists():
        return results
    edgeconv = pd.read_csv(edgeconv_path)
    edgeconv = edgeconv[edgeconv["variant"] == "edgeconv"].copy()
    edgeconv["method"] = "EdgeConv"
    edgeconv["P80_error_pct"] = (
        (edgeconv["predicted_P80_mm"] - edgeconv["ground_truth_P80_mm"])
        / edgeconv["ground_truth_P80_mm"]
        * 100.0
    )
    keep_cols = set(results.columns).union(edgeconv.columns)
    for col in keep_cols:
        if col not in results.columns:
            results[col] = np.nan
        if col not in edgeconv.columns:
            edgeconv[col] = np.nan
    return pd.concat([results, edgeconv[list(results.columns)]], ignore_index=True)


def write_summary_and_figure(test_results: pd.DataFrame) -> pd.DataFrame:
    summary = (
        test_results.groupby("method", as_index=False)
        .agg(
            n_scenes=("scene_id", "count"),
            mean_error_pct=("P80_error_pct", "mean"),
            mean_abs_P80_error_pct=("abs_P80_error_pct", "mean"),
            median_abs_P80_error_pct=("abs_P80_error_pct", "median"),
            max_abs_P80_error_pct=("abs_P80_error_pct", "max"),
            mean_abs_P80_error_mm=("abs_P80_error_mm", "mean"),
            mean_NMI=("normalized_mutual_info", "mean"),
            mean_ARI=("adjusted_rand_index", "mean"),
            mean_noise_fraction=("noise_fraction", "mean"),
        )
        .sort_values("mean_abs_P80_error_pct")
        .reset_index(drop=True)
    )
    summary.to_csv(OUT_TABLES / "surface_envelope_baseline_comparison_summary.csv", index=False)

    plot_df = summary.sort_values("mean_abs_P80_error_pct", ascending=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=180)
    colors = ["#2F80ED" if m == "EdgeConv" else "#6B7280" for m in plot_df["method"]]
    ax.barh(plot_df["method"], plot_df["mean_abs_P80_error_pct"], color=colors)
    ax.set_xlabel("Mean absolute P80 error [%]")
    ax.set_ylabel("")
    ax.grid(True, axis="x", color="#D8DEE8", linewidth=0.8)
    for idx, value in enumerate(plot_df["mean_abs_P80_error_pct"]):
        ax.text(value + 0.7, idx, f"{value:.2f}%", va="center", fontsize=9)
    ax.invert_yaxis()
    fig.tight_layout()
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIGURES / "04_surface_envelope_baseline_p80_comparison.png", bbox_inches="tight")
    plt.close(fig)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate surface-envelope synthetic baselines against EdgeConv.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mlp-edges-per-scene", type=int, default=10000)
    parser.add_argument("--mlp-max-iter", type=int, default=35)
    args = parser.parse_args()

    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)
    index = load_scene_index()
    if not index["scan_filter_version"].eq("angular_nearest_plus_xy_height_envelope").all():
        raise SystemExit("Regenerate the surface-envelope scene index before baseline evaluation.")
    train_rows = index[index["split"] == "train"].reset_index(drop=True)
    val_rows = index[index["split"] == "val"].reset_index(drop=True)
    test_rows = index[index["split"] == "test"].reset_index(drop=True)

    model_path = OUT_MODELS / "surface_envelope_edge_affinity_mlp.joblib"
    best_path = OUT_TABLES / "surface_envelope_baseline_selected_params.csv"
    validation_path = OUT_TABLES / "surface_envelope_baseline_validation_sweep.csv"

    if model_path.exists() and best_path.exists() and validation_path.exists():
        mlp = joblib.load(model_path)
        best_df = pd.read_csv(best_path)
        print("reusing cached MLP model and validation-selected baseline parameters", flush=True)
    else:
        mlp = train_mlp_baseline(train_rows, args)
        best_df = None
    method_grids = {
        "DBSCAN": [
            {"eps_m": eps, "min_samples": min_samples, "z_weight": z_weight}
            for eps, min_samples, z_weight in itertools.product([0.035, 0.050, 0.070, 0.095], [5, 10], [1.0, 1.6])
        ],
        "Region growing normals/curvature": [
            {
                "radius_m": radius,
                "max_normal_angle_deg": angle,
                "max_curvature_delta": curv,
                "max_vertical_jump_m": dz,
            }
            for radius, angle, curv, dz in itertools.product([0.045, 0.065, 0.085], [24.0, 38.0], [0.040, 0.075], [0.040, 0.070])
        ],
        "Simple graph threshold": [{"score_threshold": t} for t in np.linspace(0.55, 0.88, 12)],
        "MLP edge-affinity": [{"edge_threshold": t} for t in np.linspace(0.50, 0.95, 16)],
    }

    if best_df is None:
        validation_frames = []
        best_rows = []
        for method, grid in method_grids.items():
            val = evaluate_method_on_rows(method, val_rows, grid, mlp_model=mlp)
            validation_frames.append(val)
            best = best_params_by_validation(val).iloc[0]
            best_rows.append(best)
            print(f"best {method}: {params_from_best(best)}", flush=True)
        validation = pd.concat(validation_frames, ignore_index=True)
        validation.to_csv(validation_path, index=False)
        best_df = pd.DataFrame(best_rows)
        best_df.to_csv(best_path, index=False)

    test_frames = []
    for _, best in best_df.iterrows():
        method = best["method"]
        params = params_from_best(best)
        test_frames.append(evaluate_method_on_rows(method, test_rows, [params], mlp_model=mlp))
    test_results = pd.concat(test_frames, ignore_index=True)
    test_results = append_edgeconv_results(test_rows, test_results)
    test_results.to_csv(OUT_TABLES / "surface_envelope_baseline_test_results.csv", index=False)
    summary = write_summary_and_figure(test_results)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
