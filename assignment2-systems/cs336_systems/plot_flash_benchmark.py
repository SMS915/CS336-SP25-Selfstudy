from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRIC_LABELS = {
    "forward_ms": "Forward Latency (ms)",
    "backward_ms": "Backward Latency (ms)",
    "end_to_end_ms": "End-to-End Latency (ms)",
}

IMPL_COLORS = {
    "regular_torch": "#1f77b4",
    "flash_torch": "#ff7f0e",
    "flash_triton": "#2ca02c",
}


def parse_csv_list(text: str) -> list[str]:
    return [piece.strip() for piece in text.split(",") if piece.strip()]


def normalize_status_label(status: str) -> str:
    if "OutOfResources" in status:
        return "OOM"
    if "CompilationError" in status:
        return "CompileErr"
    if "forward_only" in status:
        return "N/A"
    if "failed" in status:
        return "Failed"
    return status


def filter_dataframe(
    df: pd.DataFrame,
    impls: list[str] | None,
    dtypes: list[str] | None,
    seq_lengths: list[int] | None,
    d_models: list[int] | None,
) -> pd.DataFrame:
    out = df.copy()
    if impls:
        out = out[out["impl"].isin(impls)]
    if dtypes:
        out = out[out["dtype"].isin(dtypes)]
    if seq_lengths:
        out = out[out["seq_len"].isin(seq_lengths)]
    if d_models:
        out = out[out["d_model"].isin(d_models)]
    return out


def get_marker_height(subset: pd.DataFrame, metric: str) -> float:
    valid = subset[metric].dropna()
    if len(valid) == 0:
        return 1.0
    y_max = float(valid.max())
    y_min = float(valid.min())
    padding = max((y_max - y_min) * 0.12, y_max * 0.08, 0.02)
    return y_max + padding


def plot_metric(
    df: pd.DataFrame,
    metric: str,
    x_axis: str,
    out_dir: Path,
    causal: bool,
) -> Path:
    if x_axis not in {"seq_len", "d_model"}:
        raise ValueError(f"Unsupported x_axis: {x_axis}")

    facet_col = "d_model" if x_axis == "seq_len" else "seq_len"
    dtype_values = sorted(df["dtype"].unique().tolist())
    facet_values = sorted(df[facet_col].unique().tolist())
    nrows = max(1, len(dtype_values))
    ncols = max(1, len(facet_values))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.8 * ncols, 3.8 * nrows),
        squeeze=False,
    )

    for row_idx, dtype_value in enumerate(dtype_values):
        for col_idx, facet_value in enumerate(facet_values):
            ax = axes[row_idx][col_idx]
            subset = df[(df["dtype"] == dtype_value) & (df[facet_col] == facet_value)].copy()
            subset = subset.sort_values(x_axis)

            marker_height = get_marker_height(subset, metric)
            have_any_series = False

            for impl in sorted(subset["impl"].unique().tolist()):
                impl_df = subset[subset["impl"] == impl].copy()
                impl_df = impl_df.sort_values(x_axis)
                color = IMPL_COLORS.get(impl, None)

                valid_df = impl_df[impl_df[metric].notna()]
                invalid_df = impl_df[impl_df[metric].isna()]

                if not valid_df.empty:
                    ax.plot(
                        valid_df[x_axis],
                        valid_df[metric],
                        marker="o",
                        linewidth=2,
                        markersize=5,
                        label=impl,
                        color=color,
                    )
                    have_any_series = True

                if not invalid_df.empty:
                    ax.scatter(
                        invalid_df[x_axis],
                        [marker_height] * len(invalid_df),
                        marker="x",
                        s=60,
                        linewidths=2,
                        color=color,
                        label=None,
                    )
                    for _, row in invalid_df.iterrows():
                        ax.annotate(
                            normalize_status_label(str(row["status"])),
                            (row[x_axis], marker_height),
                            textcoords="offset points",
                            xytext=(0, 5),
                            ha="center",
                            fontsize=8,
                            color=color,
                        )

            if x_axis in {"seq_len", "d_model"}:
                ax.set_xscale("log", base=2)

            ax.set_title(f"dtype={dtype_value}, {facet_col}={facet_value}")
            ax.set_xlabel(x_axis)
            ax.set_ylabel(METRIC_LABELS[metric])
            ax.grid(True, alpha=0.25)

            if have_any_series:
                ax.legend()

    fig.suptitle(f"{metric} vs {x_axis} (causal={causal})", fontsize=14)
    fig.tight_layout()
    out_path = out_dir / f"{metric}_vs_{x_axis}3.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot flash attention benchmark curves from CSV.")
    parser.add_argument(
        "--csv_path",
        type=str,
        default="profiles/flash_benchmark.csv",
        help="Path to the benchmark CSV produced by flash_benchmarking.py",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="profiles/flash_plots",
        help="Directory where output plots will be written",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="forward_ms,backward_ms,end_to_end_ms",
        help="Comma-separated metrics to plot",
    )
    parser.add_argument(
        "--x_axis",
        type=str,
        default="seq_len",
        choices=["seq_len", "d_model"],
        help="Which setting to place on the x-axis",
    )
    parser.add_argument(
        "--impls",
        type=str,
        default="",
        help="Optional comma-separated list of implementations to keep",
    )
    parser.add_argument(
        "--dtypes",
        type=str,
        default="",
        help="Optional comma-separated list of dtypes to keep, e.g. float32,bfloat16",
    )
    parser.add_argument(
        "--seq_lengths",
        type=str,
        default="",
        help="Optional comma-separated list of seq lengths to keep",
    )
    parser.add_argument(
        "--d_models",
        type=str,
        default="",
        help="Optional comma-separated list of embedding dimensions to keep",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    impls = parse_csv_list(args.impls) if args.impls else None
    dtypes = parse_csv_list(args.dtypes) if args.dtypes else None
    seq_lengths = [int(x) for x in parse_csv_list(args.seq_lengths)] if args.seq_lengths else None
    d_models = [int(x) for x in parse_csv_list(args.d_models)] if args.d_models else None
    metrics = parse_csv_list(args.metrics)

    df = filter_dataframe(df, impls=impls, dtypes=dtypes, seq_lengths=seq_lengths, d_models=d_models)
    if df.empty:
        raise ValueError("No rows left after filtering.")

    causal_values = sorted(df["causal"].dropna().unique().tolist())
    causal_value = bool(causal_values[0]) if causal_values else True

    output_paths = []
    for metric in metrics:
        if metric not in METRIC_LABELS:
            raise ValueError(f"Unsupported metric: {metric}")
        output_paths.append(plot_metric(df, metric=metric, x_axis=args.x_axis, out_dir=out_dir, causal=causal_value))

    print("Saved plots:")
    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()
