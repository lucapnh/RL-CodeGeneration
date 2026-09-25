"""
Compare RoboEval or EvalPlus (MBPP/MBPP+) results across training methods
(SFT, PPO, RLOO, ...) and plot them.

No project imports - reads only the result files eval/roboeval.py's save_results()
and eval/evalplus_pass_k.py's generate_evaluate() produce, so this runs standalone:
`python analysis/compare_results.py ...` (no -m needed).

RoboEval usage (pass@1 + outcome-breakdown chart):
    python analysis/compare_results.py \\
        --result "PPO=roboeval/Abgabe/Qwen-Abgabe/Qwen2.5-Coder-1.5B-Instruct/checkpoint-1200" \\
        --result "SFT=roboeval/Qwen/Qwen2.5-Coder-1.5B-Instruct/sft/checkpoint-1250" \\
        --out comparison.pdf

Each --result LABEL=PATH points to a directory containing pass1/result.csv and
error_breakdown/result.csv (exactly what eval/roboeval.py's save_results() writes
for a given --checkpoint).

EvalPlus usage (MBPP/MBPP+ pass@1 chart):
    python analysis/compare_results.py \\
        --evalplus-result "PPO=evalplus_passk/checkpoint-938" \\
        --evalplus-result "SFT=evalplus_passk/sft/1.5b/checkpoint-525" \\
        --out comparison.pdf

Each --evalplus-result LABEL=PATH points to a directory containing
evalplus_run_log.json (what eval/evalplus_pass_k.py's generate_evaluate() writes
for a given --checkpoint). Pass exactly one of --result / --evalplus-result (not
both - they produce differently-shaped charts). Plot order and color assignment
follow the order given, left to right, and are never reassigned based on which
methods happen to be passed.
"""
import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

# Categorical palette (method identity) - fixed order, not cycled based on input.
METHOD_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

# Status palette (outcome identity) - fixed, matches eval/roboeval.py::save_results'
# hardcoded error_breakdown categories exactly.
OUTCOME_ORDER = ["Success", "CompletionError", "RobotExecutionError", "PythonError"]
OUTCOME_COLORS = {
    "Success": "#0ca30c",              # good
    "CompletionError": "#fab219",      # warning
    "RobotExecutionError": "#ec835a",  # serious
    "PythonError": "#d03b3b",          # critical
}

INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def read_pass_rate(result_dir: Path) -> float:
    with open(result_dir / "pass1" / "result.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    values = [float(r["pass_1"]) for r in rows]
    return 100.0 * sum(values) / len(values)


def read_outcome_counts(result_dir: Path) -> dict:
    with open(result_dir / "error_breakdown" / "result.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {r["name"]: int(r["error_count"]) for r in rows}


def read_evalplus_pass_rates(result_dir: Path) -> dict:
    with open(result_dir / "evalplus_run_log.json", encoding="utf-8") as f:
        log = json.load(f)
    matches = dict(re.findall(r"(mbpp\+?) \(.*?\)\s*\npass@1:\t([\d.]+)", log))
    return {"MBPP": 100.0 * float(matches["mbpp"]), "MBPP+": 100.0 * float(matches["mbpp+"])}


def parse_results(args_list):
    results = {}
    for item in args_list:
        if "=" not in item:
            raise ValueError(f"--result must be LABEL=PATH, got: {item!r}")
        label, path = item.split("=", 1)
        results[label] = Path(path)
    return results


def plot_evalplus_comparison(results: dict, out: str, title: str):
    labels = list(results.keys())
    pass_rates = {label: read_evalplus_pass_rates(path) for label, path in results.items()}

    plt.rcParams.update({
        "font.family": "sans-serif",
        "text.color": INK,
        "axes.edgecolor": GRID,
        "axes.labelcolor": SECONDARY_INK,
        "xtick.color": SECONDARY_INK,
        "ytick.color": SECONDARY_INK,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
    })

    fig, ax = plt.subplots(figsize=(7.5, 5))
    fig.suptitle(title, fontsize=14, color=INK, fontweight="bold")

    metrics = ["MBPP", "MBPP+"]
    metric_colors = {"MBPP": METHOD_COLORS[0], "MBPP+": METHOD_COLORS[1]}
    x = range(len(labels))
    width = 0.35

    for i, metric in enumerate(metrics):
        offset = (i - 0.5) * width
        heights = [pass_rates[label][metric] for label in labels]
        positions = [xi + offset for xi in x]
        bars = ax.bar(positions, heights, width=width, color=metric_colors[metric],
                       edgecolor="white", linewidth=1, label=metric, zorder=3)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1f}%", (bar.get_x() + bar.get_width() / 2, height),
                        textcoords="offset points", xytext=(0, 4), ha="center",
                        fontsize=9, color=INK, fontweight="bold")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Pass@1 (%)")
    ax.set_ylim(0, max(v for rates in pass_rates.values() for v in rates.values()) * 1.25 + 5)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False, fontsize=9)

    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"Saved comparison figure to {out}")

    print("\n=== Summary ===")
    for label in labels:
        rates = pass_rates[label]
        print(f"{label}: MBPP pass@1={rates['MBPP']:.1f}%  MBPP+ pass@1={rates['MBPP+']:.1f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--result", action="append", default=[],
                         help="LABEL=PATH to a RoboEval result directory. Repeatable.")
    parser.add_argument("--evalplus-result", action="append", default=[],
                         help="LABEL=PATH to an EvalPlus (MBPP/MBPP+) result directory. Repeatable.")
    parser.add_argument("--out", default="comparison.pdf", help="Output figure path (extension picks the format, e.g. .pdf/.png/.svg).")
    parser.add_argument("--title", default=None, help="Figure title.")
    args = parser.parse_args()

    if bool(args.result) == bool(args.evalplus_result):
        raise ValueError("Pass exactly one of --result (RoboEval) or --evalplus-result (EvalPlus), not both/neither.")

    if args.evalplus_result:
        plot_evalplus_comparison(parse_results(args.evalplus_result), args.out,
                                  args.title or "MBPP / MBPP+ comparison")
        return

    results = parse_results(args.result)
    labels = list(results.keys())

    pass_rates = {label: read_pass_rate(path) for label, path in results.items()}
    outcome_counts = {label: read_outcome_counts(path) for label, path in results.items()}

    plt.rcParams.update({
        "font.family": "sans-serif",
        "text.color": INK,
        "axes.edgecolor": GRID,
        "axes.labelcolor": SECONDARY_INK,
        "xtick.color": SECONDARY_INK,
        "ytick.color": SECONDARY_INK,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(args.title or "RoboEval comparison", fontsize=14, color=INK, fontweight="bold")

    # --- Left: overall pass@1 rate per method ---
    colors = [METHOD_COLORS[i % len(METHOD_COLORS)] for i in range(len(labels))]
    bars = ax1.bar(labels, [pass_rates[l] for l in labels], color=colors, width=0.55,
                    edgecolor="white", linewidth=1, zorder=3)
    ax1.set_ylabel("Pass@1 (%)")
    ax1.set_title("RoboEval pass@1 rate", fontsize=11, color=INK)
    ax1.set_ylim(0, max(pass_rates.values()) * 1.35 + 5)
    ax1.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax1.set_axisbelow(True)
    for spine in ("top", "right"):
        ax1.spines[spine].set_visible(False)
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f"{height:.1f}%", (bar.get_x() + bar.get_width() / 2, height),
                     textcoords="offset points", xytext=(0, 4), ha="center",
                     fontsize=10, color=INK, fontweight="bold")

    # --- Right: outcome breakdown per method (stacked to 100%) ---
    bottoms = [0.0] * len(labels)
    for outcome in OUTCOME_ORDER:
        heights = []
        for label in labels:
            counts = outcome_counts[label]
            total = sum(counts.values())
            heights.append(100.0 * counts.get(outcome, 0) / total if total else 0.0)
        ax2.bar(labels, heights, bottom=bottoms, color=OUTCOME_COLORS[outcome], width=0.55,
                edgecolor="white", linewidth=1, label=outcome, zorder=3)
        for i, (h, b) in enumerate(zip(heights, bottoms)):
            if h >= 5:
                ax2.annotate(f"{h:.0f}%", (i, b + h / 2), ha="center", va="center",
                             fontsize=8.5, color="white", fontweight="bold")
        bottoms = [b + h for b, h in zip(bottoms, heights)]

    ax2.set_ylabel("Share of tasks (%)")
    ax2.set_title("Outcome breakdown", fontsize=11, color=INK)
    ax2.set_ylim(0, 100)
    ax2.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax2.set_axisbelow(True)
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=9)

    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    print(f"Saved comparison figure to {args.out}")

    print("\n=== Summary ===")
    for label in labels:
        counts = outcome_counts[label]
        total = sum(counts.values())
        breakdown = ", ".join(f"{k}={v} ({100 * v / total:.0f}%)" for k, v in counts.items())
        print(f"{label}: pass@1={pass_rates[label]:.1f}%  {breakdown}")


if __name__ == "__main__":
    main()
