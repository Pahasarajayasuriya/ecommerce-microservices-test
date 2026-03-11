"""
plot_pr_curve.py
────────────────
Precision-Recall analysis for the zero-shot GraphCodeBERT baseline.

Reads clone_report.json (produced by extract_functions_and_embed.py)
and generates three publication-ready figures:

  Figure 1 — Precision-Recall curve  (PR curve across all thresholds)
  Figure 2 — Score distribution      (true clones vs non-clones as histogram)
  Figure 3 — F1 vs Threshold         (find optimal operating point)

Also prints a threshold sweep table to the console.

Usage:
    python plot_pr_curve.py                        # uses clone_report.json
    python plot_pr_curve.py my_report.json         # custom path
"""

import json
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── CONFIG ────────────────────────────────────────────────────────────────────
REPORT_FILE  = sys.argv[1] if len(sys.argv) > 1 else "clone_report.json"
OUTPUT_DIR   = "."           # where to save the PNG files

# Ground-truth clone pairs (function name substrings, order-independent).
# Update this list whenever you add new intentional clone groups.
GROUND_TRUTH_CLONES = [
    ("calculateOrderTotal",          "compute_cart_total"),           # A  JS↔PY
    ("applyDiscountCode",            "applyMemberDiscount"),          # B  JS↔Java
    ("validateOrderRequest",         "validate_restock_request"),     # C  JS↔PY
    ("get_transaction_page",         "paginateResults"),              # D  PY↔JS
    ("calculate_weighted_average_cost", "computeWeightedScore"),      # E  PY↔Java
    ("deliverWithRetry",             "retryWithBackoff"),             # F  Java↔Java
]

# ── COLOUR PALETTE ────────────────────────────────────────────────────────────
C_CLONE    = "#2196F3"    # blue  — true clones
C_NONCLONE = "#F44336"    # red   — non-clones
C_PR       = "#1565C0"    # dark blue — PR curve line
C_F1       = "#6A1B9A"    # purple — F1 curve line
C_THRESH   = "#FF6F00"    # amber — threshold markers
C_GRID     = "#E0E0E0"

STYLE = {
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        C_GRID,
    "grid.linewidth":    0.6,
    "figure.dpi":        150,
}
plt.rcParams.update(STYLE)
# ─────────────────────────────────────────────────────────────────────────────


def fn_name(key: str) -> str:
    """Extract bare function name from 'service/file::function_name'."""
    return key.split("::")[-1]


def is_ground_truth_clone(f1_key: str, f2_key: str) -> bool:
    """Return True if this pair is a known intentional clone."""
    n1, n2 = fn_name(f1_key), fn_name(f2_key)
    for a, b in GROUND_TRUTH_CLONES:
        if (a in n1 and b in n2) or (b in n1 and a in n2):
            return True
    return False


def load_pairs(path: str):
    """Load all scored pairs and tag each with ground-truth label."""
    with open(path) as f:
        report = json.load(f)

    pairs = []
    for p in report["all_pairs"]:
        label = is_ground_truth_clone(p["function_1"], p["function_2"])
        pairs.append({
            "score":   p["score"],
            "label":   label,           # True = real clone
            "lang_1":  p["lang_1"],
            "lang_2":  p["lang_2"],
            "f1":      fn_name(p["function_1"]),
            "f2":      fn_name(p["function_2"]),
        })

    n_pos = sum(1 for p in pairs if p["label"])
    n_neg = len(pairs) - n_pos
    print(f"Loaded {len(pairs)} pairs  |  "
          f"True clones: {n_pos}  |  Non-clones: {n_neg}")
    return pairs, report.get("config", {})


def threshold_sweep(pairs, thresholds):
    """
    For each threshold compute precision, recall, F1, TP, FP, FN.
    Returns list of dicts, one per threshold.
    """
    n_pos_total = sum(1 for p in pairs if p["label"])
    rows = []
    for t in thresholds:
        tp = sum(1 for p in pairs if p["score"] >= t and     p["label"])
        fp = sum(1 for p in pairs if p["score"] >= t and not p["label"])
        fn = sum(1 for p in pairs if p["score"] <  t and     p["label"])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall    = tp / n_pos_total if n_pos_total > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        rows.append({
            "threshold": round(t, 3),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision,
            "recall":    recall,
            "f1":        f1,
            "flagged":   tp + fp,
        })
    return rows


def print_sweep_table(rows):
    """Print a formatted threshold sweep table to stdout."""
    header = (f"{'Threshold':>10}  {'Precision':>10}  {'Recall':>8}  "
              f"{'F1':>8}  {'TP':>4}  {'FP':>4}  {'FN':>4}  {'Flagged':>7}")
    print("\n" + "─" * len(header))
    print(header)
    print("─" * len(header))
    for r in rows:
        marker = " ◄ best F1" if r.get("is_best_f1") else ""
        print(f"  {r['threshold']:>8.3f}  {r['precision']:>10.3f}  "
              f"{r['recall']:>8.3f}  {r['f1']:>8.3f}  "
              f"{r['tp']:>4d}  {r['fp']:>4d}  {r['fn']:>4d}  "
              f"{r['flagged']:>7d}{marker}")
    print("─" * len(header) + "\n")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — Precision-Recall Curve
# ══════════════════════════════════════════════════════════════════════════════

def plot_pr_curve(rows, config, out_path):
    precisions = [r["precision"] for r in rows]
    recalls    = [r["recall"]    for r in rows]
    thresholds = [r["threshold"] for r in rows]
    f1s        = [r["f1"]        for r in rows]

    best_idx = int(np.argmax(f1s))
    best_r   = rows[best_idx]

    # Annotate key threshold points on the curve
    annotate_at = [0.80, 0.85, 0.90, 0.95]

    fig, ax = plt.subplots(figsize=(7, 6))

    # Filled area under curve
    ax.fill_between(recalls, precisions, alpha=0.08, color=C_PR)

    # Main curve
    ax.plot(recalls, precisions, color=C_PR, linewidth=2.2,
            label="Precision-Recall (zero-shot GraphCodeBERT)")

    # Best F1 point
    ax.scatter([best_r["recall"]], [best_r["precision"]],
               s=110, zorder=5, color=C_F1,
               label=f"Best F1={best_r['f1']:.2f} @ τ={best_r['threshold']:.2f}")

    # Annotated threshold markers
    for t in annotate_at:
        closest = min(rows, key=lambda r: abs(r["threshold"] - t))
        ax.scatter([closest["recall"]], [closest["precision"]],
                   s=60, zorder=4, color=C_THRESH, marker="D")
        ax.annotate(f"τ={t:.2f}",
                    xy=(closest["recall"], closest["precision"]),
                    xytext=(6, -14), textcoords="offset points",
                    fontsize=8, color=C_THRESH)

    # Random classifier baseline
    n_pos   = rows[0]["tp"] + rows[0]["fn"]   # total positives (constant)
    n_total = n_pos + rows[0]["fp"] + rows[0]["fn"] + rows[0]["tp"]
    # rough baseline: precision = prevalence
    prevalence = n_pos / sum(1 for r in rows if r["threshold"] == rows[0]["threshold"]
                             for _ in [None]) if False else None
    # simpler: just draw horizontal line at positive rate
    # pos_rate placeholder — not used in plot

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve\nZero-shot GraphCodeBERT — Cross-service Clone Detection",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=9, loc="lower left")

    # Annotation box: model & dataset info
    info = (f"Model: {config.get('model','GraphCodeBERT').split('/')[-1]}\n"
            f"Corpus: 5 services, 29 functions\n"
            f"Ground truth: 6 clone pairs\n"
            f"Total pairs: {len(rows)}")
    ax.text(0.97, 0.97, info, transform=ax.transAxes,
            fontsize=7.5, verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=C_GRID, alpha=0.9))

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — Score Distribution (true clones vs non-clones)
# ══════════════════════════════════════════════════════════════════════════════

def plot_score_distribution(pairs, config, out_path):
    clone_scores    = [p["score"] for p in pairs if     p["label"]]
    nonclone_scores = [p["score"] for p in pairs if not p["label"]]

    fig, ax = plt.subplots(figsize=(8, 5))

    bins = np.linspace(0.60, 1.00, 33)

    ax.hist(nonclone_scores, bins=bins, color=C_NONCLONE, alpha=0.65,
            label=f"Non-clones  (n={len(nonclone_scores)})", edgecolor="white",
            linewidth=0.4)
    ax.hist(clone_scores,    bins=bins, color=C_CLONE,    alpha=0.80,
            label=f"True clones (n={len(clone_scores)})",    edgecolor="white",
            linewidth=0.4)

    # Mark each true clone individually
    for p in pairs:
        if p["label"]:
            ax.axvline(p["score"], color=C_CLONE, linewidth=1.2,
                       linestyle="--", alpha=0.5)
            ax.text(p["score"], ax.get_ylim()[1] * 0.02,
                    f"{p['f1'][:10]}", rotation=90,
                    fontsize=6, color=C_CLONE, va="bottom", ha="center")

    # Key threshold lines
    for t, label in [(0.80, "τ=0.80"), (0.90, "τ=0.90")]:
        ax.axvline(t, color=C_THRESH, linewidth=1.5, linestyle=":",
                   label=label)

    ax.set_xlabel("Cosine Similarity Score", fontsize=12)
    ax.set_ylabel("Number of Pairs", fontsize=12)
    ax.set_title("Score Distribution: True Clones vs Non-Clones\n"
                 "Zero-shot GraphCodeBERT — Cross-service Clone Detection",
                 fontsize=12, fontweight="bold", pad=12)
    ax.legend(fontsize=9)

    # Overlap annotation
    overlap_min = min(clone_scores) if clone_scores else 0
    overlap_max = max([s for s in nonclone_scores if s >= overlap_min],
                      default=overlap_min)
    if overlap_max > overlap_min:
        ax.axvspan(overlap_min, overlap_max, alpha=0.06, color="gray",
                   label="Overlap region")
        ax.text((overlap_min + overlap_max) / 2,
                ax.get_ylim()[1] * 0.85,
                "Overlap\nregion", fontsize=8, ha="center", color="gray")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — F1 / Precision / Recall vs Threshold
# ══════════════════════════════════════════════════════════════════════════════

def plot_f1_vs_threshold(rows, config, out_path):
    thresholds = [r["threshold"] for r in rows]
    precisions = [r["precision"] for r in rows]
    recalls    = [r["recall"]    for r in rows]
    f1s        = [r["f1"]        for r in rows]

    best_idx = int(np.argmax(f1s))
    best_t   = rows[best_idx]["threshold"]
    best_f1  = rows[best_idx]["f1"]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(thresholds, f1s,        color=C_F1,       linewidth=2.2,
            label="F1 score")
    ax.plot(thresholds, precisions, color=C_CLONE,    linewidth=1.8,
            linestyle="--", label="Precision")
    ax.plot(thresholds, recalls,    color=C_NONCLONE, linewidth=1.8,
            linestyle="-.", label="Recall")

    # Best F1 marker
    ax.axvline(best_t, color=C_F1, linewidth=1.2, linestyle=":",
               label=f"Best F1={best_f1:.2f} @ τ={best_t:.2f}")
    ax.scatter([best_t], [best_f1], s=90, zorder=5, color=C_F1)

    # Annotate the two thresholds used in the paper
    for t in [0.80, 0.90]:
        r = min(rows, key=lambda r: abs(r["threshold"] - t))
        ax.annotate(f"τ={t:.2f}\nP={r['precision']:.2f}\nR={r['recall']:.2f}",
                    xy=(t, r["f1"]),
                    xytext=(12, 8), textcoords="offset points",
                    fontsize=7.5, color=C_THRESH,
                    arrowprops=dict(arrowstyle="->", color=C_THRESH,
                                   lw=0.8))
        ax.scatter([t], [r["f1"]], s=55, zorder=4,
                   color=C_THRESH, marker="D")

    ax.set_xlabel("Similarity Threshold (τ)", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Precision, Recall & F1 vs Threshold\n"
                 "Zero-shot GraphCodeBERT — Cross-service Clone Detection",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlim(0.60, 1.00)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=9, loc="center left")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  COMBINED FIGURE (all three panels — for the paper)
# ══════════════════════════════════════════════════════════════════════════════

def plot_combined(pairs, rows, config, out_path):
    clone_scores    = [p["score"] for p in pairs if     p["label"]]
    nonclone_scores = [p["score"] for p in pairs if not p["label"]]
    precisions = [r["precision"] for r in rows]
    recalls    = [r["recall"]    for r in rows]
    thresholds = [r["threshold"] for r in rows]
    f1s        = [r["f1"]        for r in rows]
    best_idx   = int(np.argmax(f1s))
    best_r     = rows[best_idx]

    fig = plt.figure(figsize=(16, 5))
    gs  = GridSpec(1, 3, figure=fig, wspace=0.32)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # ── Panel A: Score Distribution ──────────────────────────────────────────
    bins = np.linspace(0.60, 1.00, 28)
    ax1.hist(nonclone_scores, bins=bins, color=C_NONCLONE, alpha=0.65,
             label=f"Non-clones (n={len(nonclone_scores)})",
             edgecolor="white", linewidth=0.4)
    ax1.hist(clone_scores,    bins=bins, color=C_CLONE,    alpha=0.80,
             label=f"True clones (n={len(clone_scores)})",
             edgecolor="white", linewidth=0.4)
    for t, ls in [(0.80, ":"), (0.90, "--")]:
        ax1.axvline(t, color=C_THRESH, linewidth=1.4, linestyle=ls,
                    label=f"τ={t:.2f}")
    ax1.set_xlabel("Cosine Similarity Score", fontsize=10)
    ax1.set_ylabel("Pair Count", fontsize=10)
    ax1.set_title("(a) Score Distribution", fontsize=10, fontweight="bold")
    ax1.legend(fontsize=7.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ── Panel B: Precision-Recall Curve ──────────────────────────────────────
    ax2.fill_between(recalls, precisions, alpha=0.08, color=C_PR)
    ax2.plot(recalls, precisions, color=C_PR, linewidth=2.2)
    ax2.scatter([best_r["recall"]], [best_r["precision"]],
                s=90, zorder=5, color=C_F1,
                label=f"Best F1={best_r['f1']:.2f} @ τ={best_r['threshold']:.2f}")
    for t in [0.80, 0.85, 0.90]:
        cr = min(rows, key=lambda r: abs(r["threshold"] - t))
        ax2.scatter([cr["recall"]], [cr["precision"]],
                    s=50, zorder=4, color=C_THRESH, marker="D")
        ax2.annotate(f"τ={t:.2f}",
                     xy=(cr["recall"], cr["precision"]),
                     xytext=(5, -13), textcoords="offset points",
                     fontsize=7, color=C_THRESH)
    ax2.set_xlabel("Recall", fontsize=10)
    ax2.set_ylabel("Precision", fontsize=10)
    ax2.set_title("(b) Precision-Recall Curve", fontsize=10, fontweight="bold")
    ax2.set_xlim(-0.02, 1.05); ax2.set_ylim(-0.02, 1.05)
    ax2.legend(fontsize=7.5, loc="lower left")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # ── Panel C: F1 / Precision / Recall vs Threshold ────────────────────────
    ax3.plot(thresholds, f1s,        color=C_F1,       linewidth=2.0,
             label="F1")
    ax3.plot(thresholds, precisions, color=C_CLONE,    linewidth=1.6,
             linestyle="--", label="Precision")
    ax3.plot(thresholds, recalls,    color=C_NONCLONE, linewidth=1.6,
             linestyle="-.", label="Recall")
    ax3.axvline(best_r["threshold"], color=C_F1, linewidth=1.1,
                linestyle=":", label=f"Best F1 @ τ={best_r['threshold']:.2f}")
    ax3.scatter([best_r["threshold"]], [best_r["f1"]], s=80, zorder=5, color=C_F1)
    ax3.set_xlabel("Threshold (τ)", fontsize=10)
    ax3.set_ylabel("Score", fontsize=10)
    ax3.set_title("(c) Metrics vs Threshold", fontsize=10, fontweight="bold")
    ax3.set_xlim(0.60, 1.00); ax3.set_ylim(-0.02, 1.05)
    ax3.legend(fontsize=7.5, loc="center left")
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    model_short = config.get("model", "GraphCodeBERT").split("/")[-1]
    fig.suptitle(
        f"Zero-shot {model_short} — Cross-service Semantic Clone Detection\n"
        f"Synthetic ecommerce corpus · 5 services · 29 functions · 6 ground-truth clone pairs",
        fontsize=11, fontweight="bold", y=1.03
    )

    plt.savefig(out_path, bbox_inches="tight", dpi=180)
    plt.close()
    print(f"  Saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Precision-Recall Analysis")
    print(f"  Input: {REPORT_FILE}")
    print("=" * 60)

    if not os.path.exists(REPORT_FILE):
        print(f"\nERROR: {REPORT_FILE} not found.")
        print("Run extract_functions_and_embed.py first to generate it.")
        sys.exit(1)

    # Load data
    pairs, config = load_pairs(REPORT_FILE)

    # Sweep thresholds from 0.60 to 0.99 in 0.01 steps
    thresholds = [round(t, 3) for t in np.arange(0.60, 1.00, 0.01)]
    rows = threshold_sweep(pairs, thresholds)

    # Mark best F1
    best_f1_val = max(r["f1"] for r in rows)
    for r in rows:
        r["is_best_f1"] = (r["f1"] == best_f1_val)

    # Print table (every 0.05 step for readability)
    table_rows = [r for r in rows if round(r["threshold"] * 100) % 5 == 0
                  or r["is_best_f1"]]
    print_sweep_table(table_rows)

    # Print key findings
    best_row = next(r for r in rows if r["is_best_f1"])
    row_080  = min(rows, key=lambda r: abs(r["threshold"] - 0.80))
    row_090  = min(rows, key=lambda r: abs(r["threshold"] - 0.90))

    print("KEY FINDINGS")
    print(f"  τ=0.80 → Precision={row_080['precision']:.3f}  "
          f"Recall={row_080['recall']:.3f}  F1={row_080['f1']:.3f}  "
          f"Flagged={row_080['flagged']}")
    print(f"  τ=0.90 → Precision={row_090['precision']:.3f}  "
          f"Recall={row_090['recall']:.3f}  F1={row_090['f1']:.3f}  "
          f"Flagged={row_090['flagged']}")
    print(f"  Best F1={best_row['f1']:.3f} @ τ={best_row['threshold']:.2f}  "
          f"(P={best_row['precision']:.3f}  R={best_row['recall']:.3f})")
    print()

    # Generate figures
    print("Generating figures...")
    plot_pr_curve(
        rows, config,
        os.path.join(OUTPUT_DIR, "fig1_pr_curve.png"))
    plot_score_distribution(
        pairs, config,
        os.path.join(OUTPUT_DIR, "fig2_score_distribution.png"))
    plot_f1_vs_threshold(
        rows, config,
        os.path.join(OUTPUT_DIR, "fig3_f1_vs_threshold.png"))
    plot_combined(
        pairs, rows, config,
        os.path.join(OUTPUT_DIR, "fig_combined.png"))

    print("\nDone. 4 figures saved.")
    print("  fig1_pr_curve.png          — use in Section 6.2 (PR analysis)")
    print("  fig2_score_distribution.png — use in Section 6.2 (overlap analysis)")
    print("  fig3_f1_vs_threshold.png   — use in Section 6.2 (threshold selection)")
    print("  fig_combined.png           — all three panels, one figure for the paper")


if __name__ == "__main__":
    main()