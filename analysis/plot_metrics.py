"""Plot CenterPoint nuScenes-mini evaluation metrics.

Reads the official nuScenes metrics_summary.json produced by the MMDetection3D
evaluation pipeline and writes three bar charts to the given output directory:

    overall_metrics.png   - mAP and NDS
    per_class_ap.png      - Per-class AP (mean across distance thresholds)
    error_metrics.png     - mATE, mASE, mAOE, mAVE, mAAE

Usage:
    python analysis/plot_metrics.py \
        --metrics work_dirs/<run>/results_eval/metrics_summary.json \
        --out analysis/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


ERROR_KEYS = ["mATE", "mASE", "mAOE", "mAVE", "mAAE"]


def _save_bar(labels, values, title, ylabel, out_path: Path, rotate: int = 0):
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(labels)), 4))
    bars = ax.bar(labels, values, color="steelblue")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(values) * 1.15 if values else 1)
    if rotate:
        plt.setp(ax.get_xticklabels(), rotation=rotate, ha="right")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metrics(metrics_path: Path, out_dir: Path) -> None:
    data = json.loads(metrics_path.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)

    _save_bar(
        ["mAP", "NDS"],
        [float(data["mean_ap"]), float(data["nd_score"])],
        "Overall detection quality",
        "score",
        out_dir / "overall_metrics.png",
    )

    mean_dist_aps = data["mean_dist_aps"]
    classes = sorted(mean_dist_aps.keys())
    _save_bar(
        classes,
        [float(mean_dist_aps[c]) for c in classes],
        "Per-class AP (mean over distance thresholds)",
        "AP",
        out_dir / "per_class_ap.png",
        rotate=30,
    )

    tp_errors = data["tp_errors"]
    err_map = {"mATE": "trans_err", "mASE": "scale_err", "mAOE": "orient_err",
               "mAVE": "vel_err", "mAAE": "attr_err"}
    values = []
    for k in ERROR_KEYS:
        full_key = f"mean_{err_map[k]}"
        if full_key in data:
            values.append(float(data[full_key]))
        else:
            mean_val = sum(float(v) for v in tp_errors.values()) / max(len(tp_errors), 1)
            values.append(mean_val)
    _save_bar(
        ERROR_KEYS,
        values,
        "Mean true-positive error metrics (lower is better)",
        "error",
        out_dir / "error_metrics.png",
    )

    print(f"Wrote figures to {out_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True,
                        help="Path to metrics_summary.json")
    parser.add_argument("--out", type=Path, default=Path("analysis/figures"),
                        help="Output directory for PNGs")
    args = parser.parse_args()
    plot_metrics(args.metrics, args.out)


if __name__ == "__main__":
    main()
